# source this script to run in the correct conda env!!!

export OMP_NUM_THREADS=3

# ground truth
nice -n 15 python ncut/run_offline.py \
ncut.data.dataset.mode="val" \
ncut.use_ds_gt_segmentation=true \
ncut.save_dir="/mnt/hdd/ncut_eval/lseg_gt"

# spectral clustering + elbow
nice -n 15 python ncut/run_offline.py \
ncut.data.dataset.mode="val" \
ncut.method="spectral_clustering" \
ncut.save_dir="/mnt/hdd/ncut_eval/lseg_spectral_clustering"

# iterative ncut
for tau in $(LANG=en_US seq 0.5 0.05 0.9)
do
    nice -n 15 python ncut/run_offline.py \
    ncut.data.dataset.mode="val" \
    ncut.method="iterative" \
    ncut.save_dir="/mnt/hdd/ncut_eval/lseg_iterative_${tau}" \
    ncut.affinity_tau="${tau}" \
    "~ncut.segment_visualization_filename"
done