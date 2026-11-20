import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

LABELS = ["neutral", "joy", "sadness", "anger", "surprise", "fear", "disgust"]
RES = Path("experiments/results")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlp_fusion_ce")
    ap.add_argument("--text_model", default="mlp_text_ce")
    ap.add_argument("--audio_model", default="mlp_audio_ce")
    ap.add_argument("--seed", type=int, default=43)
    ap.add_argument("--per_class", type=int, default=5)
    a = ap.parse_args()

    m = pd.read_csv(RES / "manifest_with_stats.csv")
    te = m[m.split == "test"].reset_index(drop=True)
    y = te.label_id.values

    def P(n):
        return np.load(RES / "preds" / f"{n}_seed{a.seed}.npy")

    p, pt, pa = P(a.model), P(a.text_model), P(a.audio_model)
    assert len(p) == len(te) == len(pt) == len(pa)

    te["true"] = [LABELS[i] for i in y]
    te["pred"] = [LABELS[i] for i in p]
    te["text_model_pred"] = [LABELS[i] for i in pt]
    te["audio_model_pred"] = [LABELS[i] for i in pa]
    te["err"] = p != y
    te["is_neutral"] = y == 0
    err = te[te.err]
    print(f"\nModel {a.model} (seed {a.seed}): {len(err)} errors out of {len(te)} ({100 * len(err) / len(te):.1f}%)")

    print("\n== Error types ==")
    types = {"emotion -> predicted neutral": (y != 0) & (p == 0),
             "neutral -> predicted emotion": (y == 0) & (p != 0),
             "emotion -> other emotion": (y != 0) & (p != 0) & (p != y)}
    for k, v in types.items():
        print(f"{k}: {int(v.sum())}")

    print("\n== Top 10 confusions (true -> predicted) ==")
    print(pd.crosstab(err.true, err.pred).stack().sort_values(ascending=False).head(10).to_string())

    print("\n== Error rate and neutral share by audio duration ==")
    print(te.groupby(pd.cut(te.duration, [0, 1, 2, 4, 8, 400]), observed=True)
            .agg(error_rate=("err", "mean"), neutral_share=("is_neutral", "mean"), n=("err", "size")).round(3).to_string())
    print("\n== Error rate and neutral share by words in utterance ==")
    print(te.groupby(pd.cut(te.n_words, [0, 2, 5, 10, 100]), observed=True)
            .agg(error_rate=("err", "mean"), neutral_share=("is_neutral", "mean"), n=("err", "size")).round(3).to_string())

    print("\n== Does fusion rescue single-modality mistakes? ==")
    te["text_ok"], te["audio_ok"] = pt == y, pa == y
    print(te.groupby(["text_ok", "audio_ok"])["err"].agg(fusion_error_rate="mean", n="size").round(3).to_string())

    print("\n== Error rate per frequent speaker ==")
    sp = te.groupby("speaker")["err"].agg(["mean", "size"])
    print(sp[sp["size"] >= 50].sort_values("mean").round(3).to_string())

    cm = confusion_matrix(y, p, labels=list(range(7)), normalize="true")
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", xticklabels=LABELS, yticklabels=LABELS)
    plt.xlabel("predicted")
    plt.ylabel("true")
    plt.title(f"Confusion matrix (row-normalised) - {a.model}")
    plt.tight_layout()
    Path("experiments/figures").mkdir(parents=True, exist_ok=True)
    plt.savefig(f"experiments/figures/confusion_{a.model}.png", dpi=150)
    plt.close()

    parts = [g.sample(min(len(g), a.per_class), random_state=0) for _, g in err.groupby("true")]
    ex = pd.concat(parts)
    ex = ex[["dialogue_id", "utterance_id", "text", "speaker", "true", "pred", "text_model_pred",
             "audio_model_pred", "duration", "n_words", "path"]].copy()
    ex["category"] = ""
    ex["likely_cause"] = ""
    ex.to_csv(RES / "error_examples.csv", index=False, encoding="utf-8-sig")
    print(f"\nSaved {len(ex)} examples to {RES / 'error_examples.csv'}")


if __name__ == "__main__":
    main()
