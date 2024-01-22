"""
Helper that prints out relevant metrics for pseudo ground truth
evaluations.
"""

import json
from pathlib import Path

BASE_PATH = "/mnt/hdd/ncut_eval"
METRICS = [
    "val_class_agnostic_ap",
    "val_class_agnostic_ap_25",
    "val_class_agnostic_ap_50",
    "val_mean_ap",
    "val_mean_ap_25",
    "val_mean_ap_50",
    "val_mean_head_ap",
    "val_mean_common_ap",
    "val_mean_tail_ap",
]

result_files = []
base_dir = Path(BASE_PATH)
for d in base_dir.iterdir():
    if "gt" in d.name:
        continue
    if not d.is_dir():
        continue
    res_d = d / "results"
    if not res_d.exists():
        continue
    res_f = res_d / "scannet_eval_results.json"
    if not res_f.exists():
        continue
    result_files.append(res_f)

bests = {m: ("", 0) for m in METRICS}
for rf in sorted(list(result_files)):
    name = rf.parent.parent.name
    print(name)
    with open(rf, "r") as f:
        res = json.load(f)
    for m in METRICS:
        if bests[m][1] < res[m]:
            bests[m] = (name, res[m])
        print("\t", f"{m+':':<25} {res[m]:.4f}")

print("Bests:")
for k, v in bests.items():
    print(f"{k:<30}{v[0]:<40}{v[1]:.4f}")
