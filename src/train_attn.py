import argparse
import copy
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from src.models.losses import FocalLoss, class_weights
from src.train import dev_score, mean_sd
from src.utils.metrics import bootstrap_ci, compute_metrics
from src.utils.seed import set_seed


class CrossAttention(nn.Module):
    """Multi-head cross-attention written from scratch (queries from x, keys/values from ctx)."""

    def __init__(self, d, heads, dropout):
        super().__init__()
        assert d % heads == 0
        self.h, self.dk = heads, d // heads
        self.q = nn.Linear(d, d)
        self.k = nn.Linear(d, d)
        self.v = nn.Linear(d, d)
        self.o = nn.Linear(d, d)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, ctx, ctx_mask):
        B, Lq, d = x.shape
        Lk = ctx.shape[1]
        q = self.q(x).view(B, Lq, self.h, self.dk).transpose(1, 2)
        k = self.k(ctx).view(B, Lk, self.h, self.dk).transpose(1, 2)
        v = self.v(ctx).view(B, Lk, self.h, self.dk).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.dk)
        scores = scores.masked_fill(~ctx_mask[:, None, None, :], float("-inf"))
        attn = self.drop(torch.softmax(scores, dim=-1))
        out = (attn @ v).transpose(1, 2).reshape(B, Lq, d)
        return self.o(out)


class CrossBlock(nn.Module):
    def __init__(self, d, heads, dropout):
        super().__init__()
        self.attn = CrossAttention(d, heads, dropout)
        self.n1 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Dropout(dropout), nn.Linear(2 * d, d))
        self.n2 = nn.LayerNorm(d)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, ctx, ctx_mask):
        x = self.n1(x + self.drop(self.attn(x, ctx, ctx_mask)))
        return self.n2(x + self.drop(self.ff(x)))


def masked_mean(x, mask):
    m = mask.unsqueeze(-1).float()
    return (x * m).sum(1) / m.sum(1).clamp(min=1)


class CrossModalNet(nn.Module):
    def __init__(self, d_in=768, d=256, heads=4, layers=1, dropout=0.3, cross=True, n_classes=7):
        super().__init__()
        self.cross = cross

        def proj():
            return nn.Sequential(nn.LayerNorm(d_in), nn.Linear(d_in, d), nn.GELU(), nn.Dropout(dropout))

        self.pa, self.pt = proj(), proj()
        if cross:
            self.a_from_t = nn.ModuleList([CrossBlock(d, heads, dropout) for _ in range(layers)])
            self.t_from_a = nn.ModuleList([CrossBlock(d, heads, dropout) for _ in range(layers)])
        self.head = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Dropout(dropout), nn.Linear(d, n_classes))

    def forward(self, xa, ma, xt, mt):
        a, t = self.pa(xa), self.pt(xt)
        if self.cross:
            for ba, bt in zip(self.a_from_t, self.t_from_a):
                a, t = ba(a, t, mt), bt(t, a, ma)
        return self.head(torch.cat([masked_mean(a, ma), masked_mean(t, mt)], dim=-1))


def load_seq(feat_dir, split):
    out = {}
    for mod in ("audio", "text"):
        d = np.load(Path(feat_dir) / f"{split}_{mod}.npz")
        feats, lens, y = d["feats"], d["lens"], d["labels"]
        off = np.concatenate([[0], np.cumsum(lens)])
        out[mod] = [feats[off[i]:off[i + 1]] for i in range(len(lens))]
        out["y"] = y.astype(np.int64)
    return out


def make_batch(S, idx, device):
    la = torch.tensor([len(S["audio"][i]) for i in idx])
    lt = torch.tensor([len(S["text"][i]) for i in idx])
    xa = torch.zeros(len(idx), int(la.max()), 768)
    xt = torch.zeros(len(idx), int(lt.max()), 768)
    for j, i in enumerate(idx):
        xa[j, :la[j]] = torch.from_numpy(S["audio"][i].astype(np.float32))
        xt[j, :lt[j]] = torch.from_numpy(S["text"][i].astype(np.float32))
    ma = torch.arange(xa.shape[1])[None, :] < la[:, None]
    mt = torch.arange(xt.shape[1])[None, :] < lt[:, None]
    y = torch.from_numpy(S["y"][idx])
    return xa.to(device), ma.to(device), xt.to(device), mt.to(device), y.to(device)


def predict(model, S, device, bs=64):
    model.eval()
    n, out = len(S["y"]), []
    with torch.no_grad():
        for i in range(0, n, bs):
            xa, ma, xt, mt, _ = make_batch(S, np.arange(i, min(i + bs, n)), device)
            out.append(model(xa, ma, xt, mt).argmax(1).cpu())
    return torch.cat(out).numpy()


def run_one(seed, a, D, device, name):
    set_seed(seed)
    y_tr, y_dv, y_te = D["train"]["y"], D["dev"]["y"], D["test"]["y"]
    model = CrossModalNet(d=a.d, heads=a.heads, layers=a.layers, dropout=a.dropout,
                          cross=bool(a.cross)).to(device)
    w = class_weights(y_tr).to(device) if a.loss in ("wce", "focal_w") else None
    crit = nn.CrossEntropyLoss(weight=w) if a.loss in ("ce", "wce") else FocalLoss(a.gamma, weight=w)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    g = torch.Generator().manual_seed(seed)

    best, best_state, best_ep, bad = -1.0, None, 0, 0
    t0 = time.time()
    for ep in range(1, a.epochs + 1):
        model.train()
        perm = torch.randperm(len(y_tr), generator=g).numpy()
        for i in range(0, len(perm), a.bs):
            xa, ma, xt, mt, y = make_batch(D["train"], perm[i:i + a.bs], device)
            opt.zero_grad()
            crit(model(xa, ma, xt, mt), y).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        s = dev_score(y_dv, predict(model, D["dev"], device))
        if s > best:
            best, best_state, best_ep, bad = s, copy.deepcopy(model.state_dict()), ep, 0
        else:
            bad += 1
            if bad >= a.patience:
                break
    train_sec = time.time() - t0

    model.load_state_dict(best_state)
    ckpt = Path("checkpoints") / f"{name}_seed{seed}.pt"
    ckpt.parent.mkdir(exist_ok=True)
    torch.save(best_state, ckpt)

    if device == "cuda":
        torch.cuda.synchronize()
    t1 = time.time()
    pred = predict(model, D["test"], device)  # test used ONCE, with the dev-selected model
    if device == "cuda":
        torch.cuda.synchronize()
    infer_ms = 1000 * (time.time() - t1) / len(y_te)

    res = compute_metrics(y_te, pred)
    res["weighted_f1_ci95"] = bootstrap_ci(y_te, pred, "weighted")
    res["macro_f1_ci95"] = bootstrap_ci(y_te, pred, "macro")
    pdir = Path("experiments/results/preds")
    pdir.mkdir(parents=True, exist_ok=True)
    np.save(pdir / f"{name}_seed{seed}.npy", pred)
    return {"seed": seed, "best_epoch": best_ep, "dev_score": best, "test": res,
            "train_sec": round(train_sec, 1), "infer_ms_per_sample": round(infer_ms, 4),
            "params": sum(p.numel() for p in model.parameters()),
            "ckpt_mb": round(os.path.getsize(ckpt) / 1e6, 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default=r"C:\meld\features")
    ap.add_argument("--cross", type=int, default=1)
    ap.add_argument("--loss", default="ce", choices=["ce", "wce", "focal", "focal_w"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--wd", type=float, default=1e-2)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--gamma", type=float, default=2.0)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    name = f"attn_{'cross' if a.cross else 'nocross'}_{a.loss}{a.tag}"
    D = {s: load_seq(a.features, s) for s in ["train", "dev", "test"]}
    assert (D["train"]["y"].shape[0] == len(D["train"]["audio"]) == len(D["train"]["text"]))

    Path("configs").mkdir(exist_ok=True)
    Path(f"configs/{name}.json").write_text(json.dumps(vars(a), indent=2))

    runs = []
    for seed in a.seeds:
        r = run_one(seed, a, D, device, name)
        runs.append(r)
        t = r["test"]
        print(f"[{name}] seed={seed} ep={r['best_epoch']} dev={r['dev_score']:.4f} | TEST "
              f"wF1={t['weighted_f1']:.4f} mF1={t['macro_f1']:.4f} acc={t['accuracy']:.4f} "
              f"| train {r['train_sec']}s")

    summary = {k: mean_sd([r["test"][k] for r in runs]) for k in ["weighted_f1", "macro_f1", "accuracy"]}
    pc = np.array([r["test"]["per_class_f1"] for r in runs])
    summary["per_class_f1_mean"] = pc.mean(0).tolist()
    summary["per_class_f1_sd"] = (pc.std(0, ddof=1) if len(runs) > 1 else np.zeros(pc.shape[1])).tolist()
    cost = {"params": runs[0]["params"], "ckpt_mb": runs[0]["ckpt_mb"],
            "train_sec_mean": mean_sd([r["train_sec"] for r in runs])[0],
            "infer_ms_per_sample": mean_sd([r["infer_ms_per_sample"] for r in runs])[0],
            "hardware": torch.cuda.get_device_name(0) if device == "cuda" else "cpu"}
    print(f"==== {name}: wF1 {summary['weighted_f1'][0]:.4f} +- {summary['weighted_f1'][1]:.4f} | "
          f"mF1 {summary['macro_f1'][0]:.4f} +- {summary['macro_f1'][1]:.4f} | "
          f"acc {summary['accuracy'][0]:.4f} +- {summary['accuracy'][1]:.4f}")
    Path("experiments/results").mkdir(parents=True, exist_ok=True)
    Path(f"experiments/results/{name}.json").write_text(
        json.dumps({"name": name, "config": vars(a), "summary": summary, "cost": cost, "runs": runs}, indent=2))


if __name__ == "__main__":
    main()
