import argparse
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from joblib import Parallel, delayed
from tqdm import tqdm

SR = 16000
MAX_SEC = 12
N_MELS = 64
N_FFT = 800   # 50 ms
HOP = 320     # 20 ms -> 50 frames/sec, same rate as WavLM


def logmel(path):
    y, _ = sf.read(path, frames=SR * MAX_SEC, dtype="float32")
    if y.ndim > 1:
        y = y.mean(axis=1)
    if len(y) < N_FFT:
        y = np.pad(y, (0, N_FFT - len(y)))
    m = librosa.feature.melspectrogram(y=y, sr=SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS)
    return np.log(m + 1e-6).T.astype(np.float32)  # (T, N_MELS)


def main(a):
    m = pd.read_csv(a.manifest)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for split in ["train", "dev", "test"]:
        df = m[m.split == split].reset_index(drop=True)
        mels = Parallel(n_jobs=-2)(delayed(logmel)(p) for p in tqdm(df.path, desc=split))
        lens = np.array([len(x) for x in mels])
        feats = np.concatenate(mels)
        if split == "train":  # normalisation statistics from the TRAIN split only
            np.savez(out / "logmel_stats.npz", mean=feats.mean(0), std=feats.std(0) + 1e-5)
        np.savez(out / f"logmel_{split}.npz", feats=feats.astype(np.float16), lens=lens,
                 labels=df.label_id.values)
        print(split, "n =", len(df), "| frames =", feats.shape, "| mean len =", round(lens.mean(), 1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out_dir", default=r"C:\meld\features")
    main(ap.parse_args())
