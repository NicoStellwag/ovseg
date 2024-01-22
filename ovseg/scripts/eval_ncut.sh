# source this script to run in the correct conda env!!!

export OMP_NUM_THREADS=3

EVAL_DIR="/mnt/hdd/ncut_eval"
SUBDIR_PATTERN="*gt" # "*" for all subdirs

for subdir in "$EVAL_DIR"/$SUBDIR_PATTERN; do
    if [ -d "$subdir" ]; then
        nice -n 15 python ovseg/evaluation/eval_pseudo_ground_truth.py \
        "ncut.eval.base_dir=${subdir}"
    fi
done