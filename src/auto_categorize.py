import argparse
from pathlib import Path

import numpy as np
import pandas as pd

LABELS = ["neutral", "joy", "sadness", "anger", "surprise", "fear", "disgust"]
RES = Path("experiments/results")

CAUSE = {
    "Rare class missed (fear/disgust)":
        "Hypothesis: fear and disgust have only about 270 training examples each (2.7% of train), "
        "so the model rarely learns to predict them.",
    "Very short utterance (1-2 words)":
        "Hypothesis: almost no lexical content, so the emotion must come from tone; "
        "the audio is the weaker modality and the evidence is limited.",
    "Emotion predicted as neutral":
        "Hypothesis: neutral is 47% of the training data, so uncertain or mildly expressed "
        "emotions drift towards the majority class.",
    "Neutral predicted as an emotion":
        "Hypothesis: neutral sentence with emotional wording or intonation; the boundary between "
        "neutral and mild emotion is subjective and no dialogue context is available.",
    "Confusion between two emotions":
        "Hypothesis: emotions with similar valence or arousal are hard to separate from an "
        "isolated utterance, and the labels are subjective.",
}


def categorize(r):
    if r.true in ("fear", "disgust"):
        return "Rare class missed (fear/disgust)"
    if r.n_words <= 2:
        return "Very short utterance (1-2 words)"
    if r.pred == "neutral":
        return "Emotion predicted as neutral"
    if r.true == "neutral":
        return "Neutral predicted as an emotion"
    return "Confusion between two emotions"


def modality(r):
    t_ok, a_ok = r.text_model_pred == r.true, r.audio_model_pred == r.true
    if t_ok and a_ok:
        return "text-only right, audio-only right (fusion wrong)"
    if t_ok:
        return "text-only right, audio-only wrong"
    if a_ok:
        return "audio-only right, text-only wrong"
    return "text-only wrong, audio-only wrong"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlp_fusion_ce")
    ap.add_argument("--text_model", default="mlp_text_ce")
    ap.add_argument("--audio_model", default="mlp_audio_ce")
    ap.add_argument("--seed", type=int, default=43)
    a = ap.parse_args()

    m = pd.read_csv(RES / "manifest_with_stats.csv")
    te = m[m.split == "test"].reset_index(drop=True)
    y = te.label_id.values

    def P(n):
        return np.load(RES / "preds" / f"{n}_seed{a.seed}.npy")

    te["true"] = [LABELS[i] for i in y]
    te["pred"] = [LABELS[i] for i in P(a.model)]
    te["text_model_pred"] = [LABELS[i] for i in P(a.text_model)]
    te["audio_model_pred"] = [LABELS[i] for i in P(a.audio_model)]

    err = te[te.pred != te.true].copy()
    err["category"] = err.apply(categorize, axis=1)
    err["modality_pattern"] = err.apply(modality, axis=1)
    err["likely_cause"] = err.apply(
        lambda r: (f"Confusion {r.true} -> {r.pred}. " if r.category.startswith("Confusion") else "")
        + CAUSE[r.category], axis=1)
    print(f"Errors: {len(err)} of {len(te)} (model {a.model}, seed {a.seed})")

    cols = ["dialogue_id", "utterance_id", "text", "speaker", "true", "pred", "text_model_pred",
            "audio_model_pred", "n_words", "duration", "category", "modality_pattern", "likely_cause"]
    err[cols].to_csv(RES / "errors_auto_full.csv", index=False, encoding="utf-8-sig")

    rows = []
    for cat, g in err.groupby("category"):
        e = g.sample(1, random_state=0).iloc[0]
        rows.append({"Error type": cat, "Count": len(g), "% of errors": round(100 * len(g) / len(err), 1),
                     "Example": f'"{str(e.text)[:70]}" (true {e.true}, predicted {e.pred})',
                     "Likely cause (hypothesis)": CAUSE[cat]})
    t6 = pd.DataFrame(rows).sort_values("Count", ascending=False)
    t6.to_csv(RES / "error_table6_auto.csv", index=False, encoding="utf-8-sig")
    print("\n== Table 6 (all test errors) ==")
    print(t6[["Error type", "Count", "% of errors", "Example"]].to_string(index=False))

    print("\n== Modality pattern of the errors ==")
    print(err.modality_pattern.value_counts().to_string())
    print("\n== Category x true class ==")
    print(pd.crosstab(err.category, err.true).to_string())

    ex_path = RES / "error_examples.csv"
    if ex_path.exists():
        ex = pd.read_csv(ex_path, encoding="utf-8-sig").drop(columns=["category", "likely_cause"], errors="ignore")
        ex = ex.merge(err[["dialogue_id", "utterance_id", "category", "modality_pattern", "likely_cause"]],
                      on=["dialogue_id", "utterance_id"], how="left")
        ex.to_csv(RES / "error_examples_filled.csv", index=False, encoding="utf-8-sig")
        print(f"\nFilled {len(ex)} examples -> {RES / 'error_examples_filled.csv'}")


if __name__ == "__main__":
    main()
