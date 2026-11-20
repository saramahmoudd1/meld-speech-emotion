import json
from pathlib import Path

import pandas as pd

RES = Path("experiments/results")
FAMILIES = ["mlp_fusion_ce", "gru_fusion_ce", "attn_cross_ce"]

rows = []
for fam in FAMILIES:
    files = [RES / f"{fam}.json"] + sorted(RES.glob(f"{fam}_*.json"))
    for f in files:
        if not f.exists():
            continue
        r = json.loads(f.read_text())
        run = next(x for x in r["runs"] if x["seed"] == 42)
        c = r["config"]
        rows.append({"family": fam, "run": r["name"], "lr": c["lr"], "dropout": c["dropout"],
                     "dev_score": round(run["dev_score"], 4),
                     "test_wF1": round(run["test"]["weighted_f1"], 4),
                     "test_mF1": round(run["test"]["macro_f1"], 4)})

df = pd.DataFrame(rows)
for fam in FAMILIES:
    d = df[df.family == fam].sort_values("dev_score", ascending=False)
    print(f"\n== {fam} (seed 42, ranked by DEV score) ==")
    print(d.drop(columns="family").to_string(index=False))
    print("best by dev:", d.iloc[0].run)
df.to_csv(RES / "hpo_study.csv", index=False)
