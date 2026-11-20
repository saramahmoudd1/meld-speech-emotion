import argparse
from pathlib import Path

import pandas as pd
import soundfile as sf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

LABELS = ["neutral", "joy", "sadness", "anger", "surprise", "fear", "disgust"]
SPLIT_ORDER = ["train", "dev", "test"]


def audio_duration(path):
    try:
        info = sf.info(path)
        return info.frames / info.samplerate
    except Exception:
        return float("nan")


def main(manifest_path, results_dir, figures_dir):
    results_dir, figures_dir = Path(results_dir), Path(figures_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    m = pd.read_csv(manifest_path)
    m["text"] = m["text"].fillna("")
    m["duration"] = m["path"].map(audio_duration)
    m["n_words"] = m["text"].str.split().str.len()

    sizes = m.groupby("split").size().reindex(SPLIT_ORDER)
    print("\n== Split sizes ==\n", sizes)

    counts = (m.groupby(["split", "label"]).size().unstack(fill_value=0)
                .reindex(index=SPLIT_ORDER, columns=LABELS, fill_value=0))
    pct = counts.div(counts.sum(axis=1), axis=0).mul(100).round(1)
    print("\n== Class counts ==\n", counts)
    print("\n== Class % ==\n", pct)
    counts.to_csv(results_dir / "eda_class_counts.csv")
    pct.to_csv(results_dir / "eda_class_percent.csv")

    stats = m.groupby("split")[["duration", "n_words"]].agg(["mean", "median", "min", "max"]).round(2)
    print("\n== Duration (sec) and words ==\n", stats)
    stats.to_csv(results_dir / "eda_length_stats.csv")

    print(f"\nUnreadable audio: {m['duration'].isna().sum()} | shorter than 0.5s: {(m['duration'] < 0.5).sum()} | empty text: {(m['n_words'] == 0).sum()}")
    print("Duration 95th/99th percentile:", m["duration"].quantile([0.95, 0.99]).round(2).tolist())

    top = m[m.split == "train"]["speaker"].value_counts().head(10)
    print("\n== Top 10 speakers (train) ==\n", top)
    train_sp = set(m[m.split == "train"]["speaker"])
    for s in ["dev", "test"]:
        sp = set(m[m.split == s]["speaker"])
        print(f"{s}: {len(sp)} speakers, {len(sp & train_sp)} also appear in train")
    top.to_csv(results_dir / "eda_top_speakers.csv")

    tr = counts.loc["train"]
    print(f"\nImbalance (train): max/min = {tr.max()}/{tr.min()} = {tr.max()/tr.min():.1f}x")

    plt.figure(figsize=(9, 4))
    sns.barplot(data=pct.reset_index().melt(id_vars="split", var_name="emotion", value_name="percent"),
                x="emotion", y="percent", hue="split", hue_order=SPLIT_ORDER)
    plt.title("MELD class distribution per split (%)")
    plt.tight_layout()
    plt.savefig(figures_dir / "eda_class_distribution.png", dpi=150)
    plt.close()

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    sns.histplot(m["duration"].dropna().clip(upper=m["duration"].quantile(0.99)), bins=50, ax=ax[0])
    ax[0].set_title("Audio duration (sec)")
    sns.histplot(m["n_words"].clip(upper=m["n_words"].quantile(0.99)), bins=40, ax=ax[1])
    ax[1].set_title("Words per utterance")
    plt.tight_layout()
    plt.savefig(figures_dir / "eda_lengths.png", dpi=150)
    plt.close()

    m.to_csv(results_dir / "manifest_with_stats.csv", index=False)
    print("\nSaved tables to", results_dir, "and figures to", figures_dir)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--results_dir", default="experiments/results")
    ap.add_argument("--figures_dir", default="experiments/figures")
    a = ap.parse_args()
    main(a.manifest, a.results_dir, a.figures_dir)
