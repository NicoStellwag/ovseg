"""
Helper that prints out relevant metrics for pseudo ground truth
evaluations.
"""

import json
from pathlib import Path

BASE_PATH = "/mnt/hdd/ncut_eval"
METRICS = [
    "class_agnostic_val_ap",
    "class_agnostic_val_ap_25",
    "class_agnostic_val_ap_50",
    "val_mean_ap",
    "val_mean_ap_25",
    "val_mean_ap_50",
]
result_files = Path(BASE_PATH).rglob("*scannet_eval_results.json")

all_res = {}
for rf in sorted(list(result_files)):
    print(rf.parent.name)
    with open(rf, "r") as f:
        res = json.load(f)
    for m in METRICS:
        print("\t", f"{m+':':<25} {res[m]:.4f}")
