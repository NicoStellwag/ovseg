#!/bin/bash
conda activate mask3d_cuda113

# ground truth
python ncut/run_offline.py \
ncut.data.dataset.mode="val" \
ncut.use_ds_gt_segmentation=true \
ncut.save_dir="/mnt/hdd/ncut_eval/lseg_gt"

# spectral clustering + elbow
python ncut/run_offline.py \
ncut.data.dataset.mode="val" \
ncut.method="spectral_clustering" \
ncut.save_dir="/mnt/hdd/ncut_eval/lseg_spectral_clustering"

# iterative ncut
for tau in $(seq 0.5 0.05 0.9)
do
    python ncut/run_offline.py \
    ncut.data.dataset.mode="val" \
    ncut.method="iterative" \
    ncut.save_dir="/mnt/hdd/ncut_eval/lseg_iterative_${tau}" \
    ncut.affinity_tau="${tau}"
done