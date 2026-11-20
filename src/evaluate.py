import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

LABELS = ["neutral", "joy", "sadness", "anger", "surprise", "fear", "disgust"]
RES = Path("experiments/results")


def load_preds(name):
    files = sorted((RES / "preds").glob(f"{name}_seed*.npy"))
    return np.stack([np.load(f) for f in files])


def seed_avg(preds, y, idx, avg):
    return float(np.mean([f1_score(y[idx], p[idx], average=avg, zero_division=0) for p in preds]))


def paired_boot(pa, pb, y, avg, n_boot=1000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(y)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs.append(seed_avg(pa, y, idx, avg) - seed_avg(pb, y, idx, avg))
    d = np.array(diffs)
    return float(d.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--compare", nargs="*", default=[], help="pairs like modelA:modelB")
    ap.add_argument("--features", default=r"C:\meld\features")
    a = ap.parse_args()

    y = np.load(Path(a.features) / "test_text.npz")["labels"]

    rows, pcs = [], []
    for name in a.models:
        r = json.loads((RES / f"{name}.json").read_text())
        s = r["summary"]
        rows.append({
            "model": name,
            "weighted_f1": f"{s['weighted_f1'][0]:.3f} +- {s['weighted_f1'][1]:.3f}",
            "macro_f1": f"{s['macro_f1'][0]:.3f} +- {s['macro_f1'][1]:.3f}",
            "accuracy": f"{s['accuracy'][0]:.3f} +- {s['accuracy'][1]:.3f}",
        })
        pc = {"model": name}
        pc.update({l: round(v, 3) for l, v in zip(LABELS, s["per_class_f1_mean"])})
        pcs.append(pc)

    main_t, pc_t = pd.DataFrame(rows), pd.DataFrame(pcs)
    print("\n== Main results (test, mean +- SD over seeds) ==")
    print(main_t.to_string(index=False))
    print("\n== Per-class F1 (test, mean over seeds) ==")
    print(pc_t.to_string(index=False))
    main_t.to_csv(RES / "main_table.csv", index=False)
    pc_t.to_csv(RES / "per_class_table.csv", index=False)

    if a.compare:
        print("\n== Paired bootstrap, difference A - B (95% CI) ==")
    for pair in a.compare:
        na, nb = pair.split(":")
        pa, pb = load_preds(na), load_preds(nb)
        for avg in ["weighted", "macro"]:
            m, lo, hi = paired_boot(pa, pb, y, avg)
            verdict = "SIGNIFICANT" if (lo > 0 or hi < 0) else "not significant"
            print(f"{na} vs {nb} | {avg}-F1: diff={m:+.4f} CI=[{lo:+.4f}, {hi:+.4f}] -> {verdict}")


if __name__ == "__main__":
    main()
