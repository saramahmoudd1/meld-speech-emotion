import argparse
import os
import re
import subprocess
import tempfile

import numpy as np
import soundfile as sf
import torch
from transformers import AutoFeatureExtractor, AutoModel, AutoTokenizer

from src.models.mlp import FusionMLP

LABELS = ["neutral", "joy", "sadness", "anger", "surprise", "fear", "disgust"]
SR, MAX_SEC, MIN_SAMPLES, HOP = 16000, 12, 16000, 320
MODEL_NAME = "mlp_fusion_ce_tuned"


def load_audio(path):
    """Decode any audio file to 16 kHz mono with ffmpeg; keep at most 12 s."""
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    try:
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", path, "-vn", "-ac", "1",
                        "-ar", "16000", tmp.name], check=True)
        y, _ = sf.read(tmp.name, frames=SR * MAX_SEC, dtype="float32")
    finally:
        os.remove(tmp.name)
    return y.mean(axis=1) if y.ndim > 1 else y


class EmotionDemo:
    def __init__(self, ckpt=f"checkpoints/{MODEL_NAME}_seed43.pt", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.fe = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base-plus")
        self.wavlm = AutoModel.from_pretrained("microsoft/wavlm-base-plus").to(self.device).eval()
        self.tok = AutoTokenizer.from_pretrained("roberta-base")
        self.roberta = AutoModel.from_pretrained("roberta-base").to(self.device).eval()
        self.head = FusionMLP("fusion", hidden=256, dropout=0.5)
        self.head.load_state_dict(torch.load(ckpt, map_location="cpu"))
        self.head.to(self.device).eval()

    @torch.no_grad()
    def audio_vec(self, path):
        y = load_audio(path)
        real = len(y)
        if real < MIN_SAMPLES:
            y = np.pad(y, (0, MIN_SAMPLES - real))
        x = self.fe(y, sampling_rate=SR, return_tensors="pt").input_values.to(self.device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=(self.device == "cuda")):
            hs = self.wavlm(x, output_hidden_states=True).hidden_states
        h = torch.stack(hs[1:]).float().mean(0)[0]
        n = max(1, min(h.shape[0], real // HOP))
        return h[:n].float().mean(0, keepdim=True)

    @torch.no_grad()
    def text_vec(self, text):
        t = " ".join(re.sub(r"\([^)]*\)", " ", str(text)).split())
        enc = self.tok([t], padding=True, truncation=True, max_length=64, return_tensors="pt").to(self.device)
        h = self.roberta(**enc).last_hidden_state[0]
        n = int(enc.attention_mask[0].sum())
        return h[:n].float().mean(0, keepdim=True)

    @torch.no_grad()
    def predict(self, audio_path, text):
        probs = torch.softmax(self.head(self.audio_vec(audio_path), self.text_vec(text)), dim=-1)[0].cpu().numpy()
        return {l: float(p) for l, p in zip(LABELS, probs)}


def self_test(n, seed, manifest):
    import pandas as pd
    te = pd.read_csv(manifest)
    te = te[te.split == "test"].reset_index(drop=True).head(n)
    saved = np.load(f"experiments/results/preds/{MODEL_NAME}_seed{seed}.npy")[:n]
    demo = EmotionDemo(ckpt=f"checkpoints/{MODEL_NAME}_seed{seed}.pt")
    agree = 0
    for i, r in te.iterrows():
        p = demo.predict(r.path, r.text)
        top = max(p, key=p.get)
        agree += int(LABELS.index(top) == saved[i])
        print(f"{r.label:9s} -> {top:9s} ({p[top]:.2f}) | {str(r.text)[:60]}")
    print(f"\nAgreement with saved test predictions: {agree}/{len(te)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio")
    ap.add_argument("--text")
    ap.add_argument("--seed", type=int, default=43)
    ap.add_argument("--self_test", type=int, default=0)
    ap.add_argument("--manifest", default=r"C:\meld\processed\manifest.csv")
    a = ap.parse_args()
    if a.self_test:
        self_test(a.self_test, a.seed, a.manifest)
    else:
        d = EmotionDemo(ckpt=f"checkpoints/{MODEL_NAME}_seed{a.seed}.pt")
        for k, v in sorted(d.predict(a.audio, a.text).items(), key=lambda kv: -kv[1]):
            print(f"{k:9s} {v:.3f}")
