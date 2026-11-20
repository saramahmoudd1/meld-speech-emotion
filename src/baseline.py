import argparse
import json
import re
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from joblib import Parallel, delayed
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from src.utils.metrics import bootstrap_ci, compute_metrics
from src.utils.seed import set_seed

SR = 16000
MAX_SEC = 12
MIN_SAMPLES = SR  # pad very short clips to 1 second
N_MFCC = 40


def clean_text(t):
    t = re.sub(r"\([^)]*\)", " ", str(t))  # remove stage directions in parentheses
    return " ".join(t.split())


def mfcc_stats(path):
    y, _ = sf.read(path, frames=SR * MAX_SEC, dtype="float32")
    if y.ndim > 1:
        y = y.mean(axis=1)
    if len(y) < MIN_SAMPLES:
        y = np.pad(y, (0, MIN_SAMPLES - len(y)))
    mf = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=N_MFCC)
    d = librosa.feature.delta(mf, width=5, mode="nearest")
    return np.concatenate([mf.mean(1), mf.std(1), d.mean(1), d.std(1)])


def audio_features(df, cache):
    if cache.exists():
        return np.load(cache)
    rows = Parallel(n_jobs=-2)(delayed(mfcc_stats)(p) for p in tqdm(df.path, desc=f"MFCC {cache.stem}"))
    X = np.stack(rows)
    np.save(cache, X)
    return X


def fit_eval(Xtr, ytr, Xdv, ydv, Xte, yte, name, seed, out):
    best = None
    for C in [0.01, 0.1, 1, 10]:
        clf = LogisticRegression(C=C, class_weight="balanced", max_iter=2000, random_state=seed)
        clf.fit(Xtr, ytr)
        f1 = compute_metrics(ydv, clf.predict(Xdv))["weighted_f1"]
        print(f"  [{name}] C={C:<5} dev weighted-F1 = {f1:.4f}")
        if best is None or f1 > best[0]:
            best = (f1, C, clf)
    _, C, clf = best
    pred = clf.predict(Xte)
    res = compute_metrics(yte, pred)
    res["weighted_f1_ci95"] = bootstrap_ci(yte, pred)
    res["best_C"] = C
    res["dev_weighted_f1"] = best[0]
    print(f"== {name}: best C={C} | TEST weighted-F1={res['weighted_f1']:.4f} "
          f"macro-F1={res['macro_f1']:.4f} acc={res['accuracy']:.4f}")
    (out / f"baseline_{name}.json").write_text(json.dumps(res, indent=2))
    return res


def main(manifest, cache_dir, out_dir, seed):
    set_seed(seed)
    cache_dir, out_dir = Path(cache_dir), Path(out_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    m = pd.read_csv(manifest)
    m["text"] = m["text"].fillna("").map(clean_text)
    tr, dv, te = (m[m.split == s].reset_index(drop=True) for s in ["train", "dev", "test"])
    ytr, ydv, yte = tr.label_id.values, dv.label_id.values, te.label_id.values

    A = {s: audio_features(d, cache_dir / f"mfcc_{s}.npy") for s, d in [("train", tr), ("dev", dv), ("test", te)]}
    sc = StandardScaler().fit(A["train"])
    A = {k: sc.transform(v) for k, v in A.items()}

    tf = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, lowercase=True)
    T = {"train": tf.fit_transform(tr.text), "dev": tf.transform(dv.text), "test": tf.transform(te.text)}

    fit_eval(A["train"], ytr, A["dev"], ydv, A["test"], yte, "audio", seed, out_dir)
    fit_eval(T["train"], ytr, T["dev"], ydv, T["test"], yte, "text", seed, out_dir)

    F = {k: hstack([T[k], csr_matrix(A[k])]).tocsr() for k in A}
    fit_eval(F["train"], ytr, F["dev"], ydv, F["test"], yte, "fusion", seed, out_dir)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--cache_dir", default=r"C:\meld\features")
    ap.add_argument("--out_dir", default="experiments/results")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    main(a.manifest, a.cache_dir, a.out_dir, a.seed)
