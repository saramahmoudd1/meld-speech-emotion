import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from tqdm import tqdm
from transformers import AutoFeatureExtractor, AutoModel, AutoTokenizer

SR = 16000
MAX_SEC = 12
MIN_SAMPLES = SR
HOP = 320  # WavLM frame stride in samples (20 ms)


def clean_text(t):
    t = re.sub(r"\([^)]*\)", " ", str(t))
    return " ".join(t.split())


@torch.no_grad()
def extract_audio(df, model, fe, device, layer_mode):
    feats, lens = [], []
    for p in tqdm(df.path, desc="audio"):
        y, _ = sf.read(p, frames=SR * MAX_SEC, dtype="float32")
        if y.ndim > 1:
            y = y.mean(axis=1)
        real = len(y)
        if real < MIN_SAMPLES:
            y = np.pad(y, (0, MIN_SAMPLES - real))
        x = fe(y, sampling_rate=SR, return_tensors="pt").input_values.to(device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=(device == "cuda")):
            hs = model(x, output_hidden_states=True).hidden_states
        if layer_mode == "last":
            h = hs[-1][0]
        else:  # uniform average of transformer layers 1..12
            h = torch.stack(hs[1:]).float().mean(0)[0]
        n = max(1, min(h.shape[0], real // HOP))  # drop frames that come only from padding
        feats.append(h[:n].float().cpu().numpy().astype(np.float16))
        lens.append(n)
    return np.concatenate(feats), np.array(lens)


@torch.no_grad()
def extract_text(df, model, tok, device, bs=64):
    texts = df.text.fillna("").map(clean_text).tolist()
    feats, lens = [], []
    for i in tqdm(range(0, len(texts), bs), desc="text"):
        enc = tok(texts[i:i + bs], padding=True, truncation=True, max_length=64,
                  return_tensors="pt").to(device)
        h = model(**enc).last_hidden_state
        for j in range(h.shape[0]):
            n = int(enc.attention_mask[j].sum())
            feats.append(h[j, :n].float().cpu().numpy().astype(np.float16))
            lens.append(n)
    return np.concatenate(feats), np.array(lens)


def main(a):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device, torch.cuda.get_device_name(0) if device == "cuda" else "")
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    m = pd.read_csv(a.manifest)
    fe = AutoFeatureExtractor.from_pretrained(a.audio_model)
    amodel = AutoModel.from_pretrained(a.audio_model).to(device).eval()
    tok = AutoTokenizer.from_pretrained(a.text_model)
    tmodel = AutoModel.from_pretrained(a.text_model).to(device).eval()

    cost = {
        "device": device,
        "audio_model": a.audio_model,
        "audio_params_M": round(sum(p.numel() for p in amodel.parameters()) / 1e6, 1),
        "text_model": a.text_model,
        "text_params_M": round(sum(p.numel() for p in tmodel.parameters()) / 1e6, 1),
        "audio_layer_mode": a.audio_layer,
    }

    for split in ["train", "dev", "test"]:
        df = m[m.split == split].reset_index(drop=True)
        if a.limit:
            df = df.head(a.limit)
        y = df.label_id.values

        t0 = time.time()
        af, al = extract_audio(df, amodel, fe, device, a.audio_layer)
        t_audio = time.time() - t0
        np.savez(out / f"{split}_audio.npz", feats=af, lens=al, labels=y)

        t0 = time.time()
        tf_, tl = extract_text(df, tmodel, tok, device)
        t_text = time.time() - t0
        np.savez(out / f"{split}_text.npz", feats=tf_, lens=tl, labels=y)

        print(f"{split}: n={len(df)} audio frames={af.shape} text tokens={tf_.shape} "
              f"| audio {t_audio:.0f}s ({1000 * t_audio / len(df):.1f} ms/clip) "
              f"| text {t_text:.0f}s ({1000 * t_text / len(df):.1f} ms/clip)")
        cost[f"{split}_audio_sec"] = round(t_audio, 1)
        cost[f"{split}_text_sec"] = round(t_text, 1)

    (out / "extraction_cost.json").write_text(json.dumps(cost, indent=2))
    print(json.dumps(cost, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out_dir", default=r"C:\meld\features")
    ap.add_argument("--audio_model", default="microsoft/wavlm-base-plus")
    ap.add_argument("--text_model", default="roberta-base")
    ap.add_argument("--audio_layer", default="uniform", choices=["uniform", "last"])
    ap.add_argument("--limit", type=int, default=0)
    main(ap.parse_args())
