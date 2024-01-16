# source this script to run in the correct conda env!!!

# no idea why this won't work

export OMP_NUM_THREADS=3

# spectral clustering + elbow
nice -n 15 python ovseg/evaluation/eval_pseudo_ground_truth.py \
"ncut.eval=lseg_spectral_clustering"

# iterative ncut
for tau in $(LANG=en_US seq 50 5 90)
do
    nice -n 15 python ovseg/evaluation/eval_pseudo_ground_truth.py \
    "ncut.eval=lseg_iterative_${tau}"
done