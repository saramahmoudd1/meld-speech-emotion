import argparse
import json
from pathlib import Path

import pandas as pd

RES = Path("experiments/results")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--features", default=r"C:\meld\features")
    a = ap.parse_args()

    ex = json.loads((Path(a.features) / "extraction_cost.json").read_text())
    n_test = 2610
    audio_ms = 1000 * ex["test_audio_sec"] / n_test   # includes audio loading
    text_ms = 1000 * ex["test_text_sec"] / n_test     # batched, so slightly optimistic

    rows = []
    for name in a.models:
        r = json.loads((RES / f"{name}.json").read_text())
        c = r["cost"]
        pretrained = name.startswith(("mlp_", "attn_"))
        use_a, use_t = "_text_" not in name, "_audio_" not in name
        params, ms = c["params"], c["infer_ms_per_sample"]
        if pretrained:
            if use_a:
                params += ex["audio_params_M"] * 1e6
                ms += audio_ms
            if use_t:
                params += ex["text_params_M"] * 1e6
                ms += text_ms
        rows.append({"model": name, "pretrained_encoders": pretrained,
                     "trainable_params": c["params"], "total_params_M": round(params / 1e6, 2),
                     "checkpoint_MB": c["ckpt_mb"], "train_sec_per_run": round(c["train_sec_mean"], 1),
                     "head_infer_ms": round(c["infer_ms_per_sample"], 4),
                     "end_to_end_infer_ms": round(ms, 2), "hardware": c["hardware"]})

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    print(f"\nOne-time feature extraction (WavLM + RoBERTa, all splits): "
          f"{sum(v for k, v in ex.items() if k.endswith('_sec')):.0f} s on {ex['device']}")
    df.to_csv(RES / "cost_table.csv", index=False)


if __name__ == "__main__":
    main()
