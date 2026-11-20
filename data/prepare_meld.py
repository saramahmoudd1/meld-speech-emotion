import argparse
import subprocess
from pathlib import Path

import pandas as pd
from tqdm import tqdm

SPLITS = {
    "train": ("train_sent_emo.csv", "train_splits"),
    "dev":   ("dev_sent_emo.csv",   "dev_splits_complete"),
    "test":  ("test_sent_emo.csv",  "output_repeated_splits_test"),
}
LABELS = ["neutral", "joy", "sadness", "anger", "surprise", "fear", "disgust"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}


def clean_text(s):
    """Fix broken apostrophes/quotes stored as control characters in MELD."""
    s = str(s)
    s = (s.replace("\x92", "'").replace("\x91", "'")
          .replace("\x93", '"').replace("\x94", '"')
          .replace("\x96", "-").replace("\x97", "-")
          .replace("\x85", "...").replace("\xa0", " "))
    s = s.replace("\u00c2", "")
    return " ".join(s.split())


def extract_wav(mp4, wav):
    if wav.exists() and wav.stat().st_size > 1000:
        return True
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(mp4),
           "-vn", "-ac", "1", "-ar", "16000", str(wav)]
    r = subprocess.run(cmd)
    return r.returncode == 0 and wav.exists() and wav.stat().st_size > 1000


def main(root, out, limit):
    root, out = Path(root), Path(out)
    wav_dir = out / "wav"
    wav_dir.mkdir(parents=True, exist_ok=True)

    rows, bad = [], []
    for split, (csv_name, vid_dir) in SPLITS.items():
        df = pd.read_csv(root / csv_name, encoding="utf-8")
        if limit:
            df = df.head(limit)
        (wav_dir / split).mkdir(parents=True, exist_ok=True)

        for _, r in tqdm(df.iterrows(), total=len(df), desc=split):
            name = f"dia{r['Dialogue_ID']}_utt{r['Utterance_ID']}"
            mp4 = root / vid_dir / f"{name}.mp4"
            wav = wav_dir / split / f"{name}.wav"

            if not mp4.exists() or not extract_wav(mp4, wav):
                bad.append((split, name))
                continue

            emo = str(r["Emotion"]).strip().lower()
            rows.append({
                "split": split,
                "dialogue_id": int(r["Dialogue_ID"]),
                "utterance_id": int(r["Utterance_ID"]),
                "path": str(wav),
                "text": clean_text(r["Utterance"]),
                "speaker": r["Speaker"],
                "label": emo,
                "label_id": LABEL2ID[emo],
            })

    manifest = pd.DataFrame(rows)
    manifest.to_csv(out / "manifest.csv", index=False, encoding="utf-8")
    pd.DataFrame(bad, columns=["split", "clip"]).to_csv(out / "skipped.csv", index=False)

    print("\nKept:", len(manifest), "| Skipped:", len(bad))
    print(manifest.groupby("split").size())
    print(manifest.groupby(["split", "label"]).size().unstack(fill_value=0))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default="data/processed")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    main(a.root, a.out, a.limit)
