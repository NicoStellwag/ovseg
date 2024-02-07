"""
Helper that prints out relevant metrics for pseudo ground truth
evaluations.
"""

import json
from pathlib import Path
import re

BASE_PATH = "/mnt/hdd/ncut_eval"
METRICS = [
    "val_class_agnostic_ap_25",
    "val_class_agnostic_ap_50",
    "val_class_agnostic_ap",
    "val_mean_ap_25",
    "val_mean_ap_50",
    "val_mean_ap",
]
ANALYSIS_METRICS = [
    "val_class_agnostic_ap",
    "val_class_agnostic_ap_25",
    # "val_class_agnostic_ap_50",
    "val_mean_ap",
    "val_mean_ap_25",
    # "val_mean_ap_50",
    # "val_mean_head_ap",
    # "val_mean_common_ap",
    # "val_mean_tail_ap",
]


def recursive_mean(d):
    for k, v in d.items():
        if isinstance(v, list):
            d[k] = sum(v) / max(len(v), 1)
        if isinstance(v, dict):
            recursive_mean(v)


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
analysis = {
    m: {
        "method": {"it": [], "sc": []},
        "dim": {"full": [], "reduced": []},
        "3dfeats": {"yes": [], "no": []},
        "sc_filter": {"no": [], "yes": []},
        "it_tau": {"0." + str(t): [] for t in range(50, 95, 5)},
    }
    for m in ANALYSIS_METRICS
}

latex_table_rows = []

for rf in sorted(list(result_files)):
    name = rf.parent.parent.name
    print(name)
    with open(rf, "r") as f:
        res = json.load(f)
    for m in METRICS:
        print("\t", f"{m+':':<25} {res[m]:.4f}")

        # best
        if bests[m][1] < res[m]:
            bests[m] = (name, res[m])

        # analysis
        if m in ANALYSIS_METRICS:
            if "iterative" in name:
                analysis[m]["method"]["it"].append(res[m])
            else:
                analysis[m]["method"]["sc"].append(res[m])

            if "fulldim" in name:
                analysis[m]["dim"]["full"].append(res[m])
            else:
                analysis[m]["dim"]["reduced"].append(res[m])

            if "2d3d" in name:
                analysis[m]["3dfeats"]["yes"].append(res[m])
            else:
                analysis[m]["3dfeats"]["no"].append(res[m])

            if "iterative" not in name:
                if "nofiltering" in name:
                    analysis[m]["sc_filter"]["no"].append(res[m])
                else:
                    analysis[m]["sc_filter"]["yes"].append(res[m])

            if "iterative" in name:
                pattern = r"\d{1}\.\d{2}"
                tau = re.findall(pattern, name)[0]
                analysis[m]["it_tau"][tau].append(res[m])

    # latex table
    if "iterative" in name:
        percents = [f"{res[m] * 100:.2f}" for m in METRICS]
        in_math_envs = [f"\\( {p} \\)" for p in percents]
        res_line_section = " & ".join(in_math_envs)
        tau = "\\( " + name[15:19] + " \\)"
        reduce_2d = "\\cmark" if "reduceddim" in name else "\\xmark"
        take_3d = "\\cmark" if "2d3d" in name else "\\xmark"
        latex_table_rows.append(
            f"{tau} & {reduce_2d} & {take_3d} & {res_line_section} \\\\"
        )


print()
print("Bests:")
for k, v in bests.items():
    print(f"{k:<30}{v[0]:<40}{v[1]:.4f}")

print()
recursive_mean(analysis)
print("Analysis (means over hyperparameters):")
print(json.dumps(analysis, indent=4))

print()
print("Latex table rows:")
for r in latex_table_rows:
    print(r)
