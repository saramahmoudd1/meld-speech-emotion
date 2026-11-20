import argparse
import copy
import json
import os
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from src.models.losses import FocalLoss, class_weights
from src.train import dev_score, mean_sd
from src.utils.metrics import bootstrap_ci, compute_metrics
from src.utils.seed import set_seed

TOK = re.compile(r"[a-z0-9']+|[^\sa-z0-9']")  # words + punctuation (? and ! carry emotion)
MAX_TOK = 64
N_MELS = 64


def clean_text(t):
    t = re.sub(r"\([^)]*\)", " ", str(t))
    return " ".join(t.split())


def load_data(manifest, feat_dir):
    m = pd.read_csv(manifest)
    m["text"] = m["text"].fillna("").map(clean_text)
    st = np.load(Path(feat_dir) / "logmel_stats.npz")
    mean, std = st["mean"], st["std"]

    toks = {}
    for s in ["train", "dev", "test"]:
        toks[s] = [TOK.findall(t.lower()) for t in m[m.split == s].text]
    c = Counter(t for sent in toks["train"] for t in sent)  # vocabulary from TRAIN only
    itos = ["<pad>", "<unk>"] + [w for w, n in c.most_common() if n >= 2]
    stoi = {w: i for i, w in enumerate(itos)}

    D = {}
    for s in ["train", "dev", "test"]:
        d = np.load(Path(feat_dir) / f"logmel_{s}.npz")
        feats, lens, y = d["feats"], d["lens"], d["labels"].astype(np.int64)
        assert (y == m[m.split == s].label_id.values).all(), "audio/manifest misaligned"
        off = np.concatenate([[0], np.cumsum(lens)])
        A = [((feats[off[i]:off[i + 1]].astype(np.float32) - mean) / std).astype(np.float16)
             for i in range(len(lens))]
        T = [[stoi.get(t, 1) for t in sent[:MAX_TOK]] or [1] for sent in toks[s]]
        D[s] = (A, T, y)
    return D, len(itos)


def batch(S, idx, device):
    A, T, y = S
    la = torch.tensor([len(A[i]) for i in idx])
    lt = torch.tensor([len(T[i]) for i in idx])
    xa = torch.zeros(len(idx), int(la.max()), N_MELS)
    xt = torch.zeros(len(idx), int(lt.max()), dtype=torch.long)
    for j, i in enumerate(idx):
        xa[j, :la[j]] = torch.from_numpy(A[i].astype(np.float32))
        xt[j, :lt[j]] = torch.tensor(T[i])
    return xa.to(device), la, xt.to(device), lt, torch.from_numpy(y[idx]).to(device)


class SeqEncoder(nn.Module):
    """BiGRU followed by masked mean pooling over time."""

    def __init__(self, d_in, hidden):
        super().__init__()
        self.gru = nn.GRU(d_in, hidden, batch_first=True, bidirectional=True)

    def forward(self, x, lens):
        packed = pack_padded_sequence(x, lens.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.gru(packed)
        out, _ = pad_packed_sequence(out, batch_first=True, total_length=x.shape[1])
        mask = (torch.arange(x.shape[1], device=x.device)[None, :] < lens.to(x.device)[:, None]).unsqueeze(-1)
        return (out * mask).sum(1) / mask.sum(1)


class GRUFusion(nn.Module):
    def __init__(self, modality, vocab, emb=128, hidden=128, dropout=0.3, n_classes=7):
        super().__init__()
        self.use_a = modality in ("audio", "fusion")
        self.use_t = modality in ("text", "fusion")
        if self.use_a:
            self.a = SeqEncoder(N_MELS, hidden)
        if self.use_t:
            self.emb = nn.Embedding(vocab, emb, padding_idx=0)  # trained from scratch
            self.t = SeqEncoder(emb, hidden)
        self.drop = nn.Dropout(dropout)
        k = 2 * hidden * (int(self.use_a) + int(self.use_t))
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(k, hidden), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(hidden, n_classes))

    def forward(self, xa, la, xt, lt):
        parts = []
        if self.use_a:
            parts.append(self.a(xa, la))
        if self.use_t:
            parts.append(self.t(self.drop(self.emb(xt)), lt))
        return self.head(torch.cat(parts, dim=-1))


def predict(model, S, device, bs=128):
    model.eval()
    n, out = len(S[2]), []
    with torch.no_grad():
        for i in range(0, n, bs):
            xa, la, xt, lt, _ = batch(S, np.arange(i, min(i + bs, n)), device)
            out.append(model(xa, la, xt, lt).argmax(1).cpu())
    return torch.cat(out).numpy()


def run_one(seed, a, D, vocab, device, name):
    set_seed(seed)
    y_tr, y_dv, y_te = D["train"][2], D["dev"][2], D["test"][2]
    model = GRUFusion(a.modality, vocab, a.emb, a.hidden, a.dropout).to(device)

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
            xa, la, xt, lt, y = batch(D["train"], perm[i:i + a.bs], device)
            opt.zero_grad()
            crit(model(xa, la, xt, lt), y).backward()
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
    ap.add_argument("--manifest", default=r"C:\meld\processed\manifest.csv")
    ap.add_argument("--features", default=r"C:\meld\features")
    ap.add_argument("--modality", default="fusion", choices=["audio", "text", "fusion"])
    ap.add_argument("--loss", default="ce", choices=["ce", "wce", "focal", "focal_w"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--wd", type=float, default=1e-2)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--emb", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--gamma", type=float, default=2.0)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    name = f"gru_{a.modality}_{a.loss}{a.tag}"
    D, vocab = load_data(a.manifest, a.features)
    print("vocab size:", vocab, "| device:", device)

    Path("configs").mkdir(exist_ok=True)
    Path(f"configs/{name}.json").write_text(json.dumps(vars(a), indent=2))

    runs = []
    for seed in a.seeds:
        r = run_one(seed, a, D, vocab, device, name)
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
