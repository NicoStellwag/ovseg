export OMP_NUM_THREADS=3
python ./ncut/run_offline.py ncut.use_ds_gt_segmentation="false" ncut.data.dataset.mode=train
python ./ncut/run_offline.py ncut.use_ds_gt_segmentation="true" ncut.data.dataset.mode=val