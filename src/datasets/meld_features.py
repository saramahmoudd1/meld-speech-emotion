from pathlib import Path

import numpy as np


def _pool(npz_path):
    """Mean-pool frame/token features of each utterance."""
    d = np.load(npz_path)
    feats, lens, y = d["feats"], d["lens"], d["labels"]
    off = np.concatenate([[0], np.cumsum(lens)])
    X = np.stack([feats[off[i]:off[i + 1]].astype(np.float32).mean(axis=0) for i in range(len(lens))])
    return X, y.astype(np.int64)


def load_pooled(feature_dir, split):
    """Returns (audio_vec, text_vec, labels) for a split. Cached after first call."""
    fd = Path(feature_dir)
    cache = fd / f"pooled_{split}.npz"
    if cache.exists():
        d = np.load(cache)
        return d["audio"], d["text"], d["labels"]
    xa, ya = _pool(fd / f"{split}_audio.npz")
    xt, yt = _pool(fd / f"{split}_text.npz")
    assert (ya == yt).all(), "audio/text labels are misaligned"
    np.savez(cache, audio=xa, text=xt, labels=ya)
    return xa, xt, ya
