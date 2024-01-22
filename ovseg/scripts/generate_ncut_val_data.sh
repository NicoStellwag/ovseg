# source this script to run in the correct conda env!!!
# exprects the following defaults in config:
# use_3d_feats: true
# dim_reduce_2d: -1

export OMP_NUM_THREADS=3

# ground truth
python ncut/run_offline.py \
ncut.data.dataset.mode="val" \
ncut.use_ds_gt_segmentation=true \
ncut.save_dir="/mnt/hdd/ncut_eval/lseg_gt"

# spectral clustering, no dim reduction, 3d feats, no filtering
nice -n 15 python ncut/run_offline.py \
ncut.data.dataset.mode="val" \
ncut.method="spectral_clustering" \
ncut.save_dir="/mnt/hdd/ncut_eval/lseg_sc_fulldim_2d3d_nofiltering"

# spectral clustering, dim reduction, 3d feats, no filtering
nice -n 15 python ncut/run_offline.py \
ncut.data.dataset.mode="val" \
ncut.method="spectral_clustering" \
ncut.save_dir="/mnt/hdd/ncut_eval/lseg_sc_reduceddim_2d3d_nofiltering" \
ncut.dim_reduce_2d=128

# spectral clustering, no dim reduction, no 3d feats, no filtering
nice -n 15 python ncut/run_offline.py \
ncut.data.dataset.mode="val" \
ncut.method="spectral_clustering" \
ncut.save_dir="/mnt/hdd/ncut_eval/lseg_sc_fulldim_2d_nofiltering" \
ncut.use_3d_feats="false"

# spectral clustering, dim reduction, no 3d feats, no filtering
nice -n 15 python ncut/run_offline.py \
ncut.data.dataset.mode="val" \
ncut.method="spectral_clustering" \
ncut.save_dir="/mnt/hdd/ncut_eval/lseg_sc_reduceddim_2d_nofiltering" \
ncut.dim_reduce_2d=128 \
ncut.use_3d_feats="false"

# spectral clustering, no dim reduction, 3d feats, filtering
nice -n 15 python ncut/run_offline.py \
ncut.data.dataset.mode="val" \
ncut.method="spectral_clustering" \
ncut.save_dir="/mnt/hdd/ncut_eval/lseg_sc_fulldim_2d3d_filtering" \
ncut.spectral_filter_instances="true"

# spectral clustering, dim reduction, 3d feats, filtering
nice -n 15 python ncut/run_offline.py \
ncut.data.dataset.mode="val" \
ncut.method="spectral_clustering" \
ncut.save_dir="/mnt/hdd/ncut_eval/lseg_sc_reduceddim_2d3d_filtering" \
ncut.dim_reduce_2d=128 \
ncut.spectral_filter_instances="true"

# spectral clustering, no dim reduction, no 3d feats, filtering
nice -n 15 python ncut/run_offline.py \
ncut.data.dataset.mode="val" \
ncut.method="spectral_clustering" \
ncut.save_dir="/mnt/hdd/ncut_eval/lseg_sc_fulldim_2d_filtering" \
ncut.use_3d_feats="false" \
ncut.spectral_filter_instances="true"

# spectral clustering, dim reduction, no 3d feats, filtering
nice -n 15 python ncut/run_offline.py \
ncut.data.dataset.mode="val" \
ncut.method="spectral_clustering" \
ncut.save_dir="/mnt/hdd/ncut_eval/lseg_sc_reduceddim_2d_filtering" \
ncut.dim_reduce_2d=128 \
ncut.use_3d_feats="false" \
ncut.spectral_filter_instances="true"

# iterative ncut
for tau in $(LANG=en_US seq 0.5 0.05 0.9)
do

    # iterative, no dim reduction, 3d feats
    nice -n 15 python ncut/run_offline.py \
    ncut.data.dataset.mode="val" \
    ncut.method="iterative" \
    ncut.save_dir="/mnt/hdd/ncut_eval/lseg_iterative_${tau}_fulldim_2d3d" \
    ncut.affinity_tau="${tau}"

    # iterative, dim reduction, 3d feats
    nice -n 15 python ncut/run_offline.py \
    ncut.data.dataset.mode="val" \
    ncut.method="iterative" \
    ncut.save_dir="/mnt/hdd/ncut_eval/lseg_iterative_${tau}_reduceddim_2d3d" \
    ncut.affinity_tau="${tau}" \
    ncut.dim_reduce_2d=128

    # iterative, no dim reduction, no 3d feats
    nice -n 15 python ncut/run_offline.py \
    ncut.data.dataset.mode="val" \
    ncut.method="iterative" \
    ncut.save_dir="/mnt/hdd/ncut_eval/lseg_iterative_${tau}_fulldim_2d" \
    ncut.affinity_tau="${tau}" \
    ncut.use_3d_feats="false"

    # iterative, dim reduction, no 3d feats
    nice -n 15 python ncut/run_offline.py \
    ncut.data.dataset.mode="val" \
    ncut.method="iterative" \
    ncut.save_dir="/mnt/hdd/ncut_eval/lseg_iterative_${tau}_reducedim_2d" \
    ncut.affinity_tau="${tau}" \
    ncut.dim_reduce_2d=128 \
    ncut.use_3d_feats="false"

done