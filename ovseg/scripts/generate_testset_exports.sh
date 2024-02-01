export OMP_NUM_THREADS=3

CP_1="/home/stellwag/dev/ovseg/saved/train_i1_lseg_iterative_070_fulldim_2d3d/epoch\=899-val_class_agnostic_ap\=0.262.ckpt"
CP_2="/home/luebberstedt/ovseg/saved/19-47-04_30-01-24/epoch\=1024-val_class_agnostic_ap\=0.242.ckpt"
CP_3="/home/luebberstedt/ovseg/saved/43-48-13_30-01-24/epoch\=1099-val_class_agnostic_ap\=0.258.ckpt"
CP_4="/home/luebberstedt/ovseg/saved/44-33-22_30-01-24/epoch\=1149-val_class_agnostic_ap\=0.272.ckpt"
DBSCAN_EPS=0.95

python main_openvocab_instance_segmentation.py \
"general.experiment_name=i1_test_export" \
"general.checkpoint=${CP_1}" \
"general.train_mode=false" \
"general.save_visualizations=true" \
"data.test_mode=test" \
"general.export=true" \
"general.export_threshold=0.0" \
"general.export_root_path=/mnt/hdd/viz_poster/model_exports"

python main_openvocab_instance_segmentation.py \
"general.experiment_name=i2_test_export" \
"general.checkpoint=${CP_2}" \
"general.train_mode=false" \
"general.save_visualizations=true" \
"data.test_mode=test" \
"general.export=true" \
"general.export_threshold=0.0" \
"general.export_root_path=/mnt/hdd/viz_poster/model_exports"

python main_openvocab_instance_segmentation.py \
"general.experiment_name=i3_test_export" \
"general.checkpoint=${CP_3}" \
"general.train_mode=false" \
"general.save_visualizations=true" \
"data.test_mode=test" \
"general.export=true" \
"general.export_threshold=0.0" \
"general.export_root_path=/mnt/hdd/viz_poster/model_exports"

python main_openvocab_instance_segmentation.py \
"general.experiment_name=i4_test_export" \
"general.checkpoint=${CP_4}" \
"general.train_mode=false" \
"general.save_visualizations=true" \
"data.test_mode=test" \
"general.export=true" \
"general.export_threshold=0.0" \
"general.export_root_path=/mnt/hdd/viz_poster/model_exports"
