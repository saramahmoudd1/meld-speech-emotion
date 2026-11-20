import argparse
import copy
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.datasets.meld_features import load_pooled
from src.models.losses import FocalLoss, class_weights
from src.models.mlp import FusionMLP
from src.utils.metrics import bootstrap_ci, compute_metrics
from src.utils.seed import set_seed


def predict(model, xa, xt, device, bs=512):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(xa), bs):
            out.append(model(xa[i:i + bs].to(device), xt[i:i + bs].to(device)).argmax(1).cpu())
    return torch.cat(out).numpy()


def dev_score(y, p):
    m = compute_metrics(y, p)
    return 0.5 * (m["weighted_f1"] + m["macro_f1"])


def run_one(seed, a, data, device, name):
    set_seed(seed)
    (xa_tr, xt_tr, y_tr), (xa_dv, xt_dv, y_dv), (xa_te, xt_te, y_te) = data
    model = FusionMLP(a.modality, hidden=a.hidden, dropout=a.dropout).to(device)

    w = class_weights(y_tr).to(device) if a.loss in ("wce", "focal_w") else None
    if a.loss in ("ce", "wce"):
        crit = nn.CrossEntropyLoss(weight=w)
    else:
        crit = FocalLoss(a.gamma, weight=w)

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    g = torch.Generator().manual_seed(seed)
    loader = DataLoader(TensorDataset(xa_tr, xt_tr, torch.from_numpy(y_tr)),
                        batch_size=a.bs, shuffle=True, generator=g)

    best, best_state, best_ep, bad = -1.0, None, 0, 0
    t0 = time.time()
    for ep in range(1, a.epochs + 1):
        model.train()
        for xa, xt, y in loader:
            xa, xt, y = xa.to(device), xt.to(device), y.to(device)
            opt.zero_grad()
            crit(model(xa, xt), y).backward()
            opt.step()
        s = dev_score(y_dv, predict(model, xa_dv, xt_dv, device))
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

    # test set is used ONCE, with the model selected on dev
    if device == "cuda":
        torch.cuda.synchronize()
    t1 = time.time()
    pred = predict(model, xa_te, xt_te, device)
    if device == "cuda":
        torch.cuda.synchronize()
    infer_ms = 1000 * (time.time() - t1) / len(y_te)

    res = compute_metrics(y_te, pred)
    res["weighted_f1_ci95"] = bootstrap_ci(y_te, pred, "weighted")
    res["macro_f1_ci95"] = bootstrap_ci(y_te, pred, "macro")
    pdir = Path("experiments/results/preds")
    pdir.mkdir(parents=True, exist_ok=True)
    np.save(pdir / f"{name}_seed{seed}.npy", pred)

    return {
        "seed": seed, "best_epoch": best_ep, "dev_score": best, "test": res,
        "train_sec": round(train_sec, 1), "infer_ms_per_sample": round(infer_ms, 4),
        "params": sum(p.numel() for p in model.parameters()),
        "ckpt_mb": round(os.path.getsize(ckpt) / 1e6, 3),
    }


def mean_sd(vals):
    v = np.asarray(vals, dtype=float)
    return float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default=r"C:\meld\features")
    ap.add_argument("--modality", default="fusion", choices=["audio", "text", "fusion"])
    ap.add_argument("--loss", default="wce", choices=["ce", "wce", "focal", "focal_w"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-2)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--gamma", type=float, default=2.0)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    name = f"mlp_{a.modality}_{a.loss}{a.tag}"

    data = []
    for s in ["train", "dev", "test"]:
        xa, xt, y = load_pooled(a.features, s)
        data.append((torch.from_numpy(xa), torch.from_numpy(xt), y))

    Path("configs").mkdir(exist_ok=True)
    Path(f"configs/{name}.json").write_text(json.dumps(vars(a), indent=2))

    runs = []
    for seed in a.seeds:
        r = run_one(seed, a, data, device, name)
        runs.append(r)
        t = r["test"]
        print(f"[{name}] seed={seed} ep={r['best_epoch']} dev={r['dev_score']:.4f} | TEST "
              f"wF1={t['weighted_f1']:.4f} mF1={t['macro_f1']:.4f} acc={t['accuracy']:.4f}")

    summary = {k: mean_sd([r["test"][k] for r in runs]) for k in ["weighted_f1", "macro_f1", "accuracy"]}
    pc = np.array([r["test"]["per_class_f1"] for r in runs])
    summary["per_class_f1_mean"] = pc.mean(0).tolist()
    summary["per_class_f1_sd"] = (pc.std(0, ddof=1) if len(runs) > 1 else np.zeros(pc.shape[1])).tolist()
    cost = {
        "params": runs[0]["params"], "ckpt_mb": runs[0]["ckpt_mb"],
        "train_sec_mean": mean_sd([r["train_sec"] for r in runs])[0],
        "infer_ms_per_sample": mean_sd([r["infer_ms_per_sample"] for r in runs])[0],
        "hardware": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
    }
    print(f"==== {name}: wF1 {summary['weighted_f1'][0]:.4f} +- {summary['weighted_f1'][1]:.4f} | "
          f"mF1 {summary['macro_f1'][0]:.4f} +- {summary['macro_f1'][1]:.4f} | "
          f"acc {summary['accuracy'][0]:.4f} +- {summary['accuracy'][1]:.4f}")

    Path("experiments/results").mkdir(parents=True, exist_ok=True)
    out = {"name": name, "config": vars(a), "summary": summary, "cost": cost, "runs": runs}
    Path(f"experiments/results/{name}.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
