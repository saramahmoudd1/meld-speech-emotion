import argparse
import json
from pathlib import Path

import pandas as pd

from src.utils.metrics import bootstrap_ci, compute_metrics

ap = argparse.ArgumentParser()
ap.add_argument("--manifest", required=True)
ap.add_argument("--out", default="experiments/results/majority_baseline.json")
a = ap.parse_args()

m = pd.read_csv(a.manifest)
train = m[m.split == "train"]
test = m[m.split == "test"]
majority = int(train.label_id.mode()[0])
pred = [majority] * len(test)

res = compute_metrics(test.label_id.values, pred)
res["weighted_f1_ci95"] = bootstrap_ci(test.label_id.values, pred)
print(json.dumps(res, indent=2))

Path(a.out).parent.mkdir(parents=True, exist_ok=True)
Path(a.out).write_text(json.dumps(res, indent=2))
