# source this for correct conda env!

export OMP_NUM_THREADS=3

# # escape = with backslashes!
# CHECKPOINT_PATH="/home/stellwag/dev/ovseg/saved/fixed_losses_feature_dropout_all_instances/epoch\=549-val_mean_ap_50\=0.004.ckpt"
# EXPERIMENT_NAME="eval_it1_fixed_losses_feature_dropout_all_instances"
CHECKPOINT_PATH1="/home/stellwag/dev/ovseg/saved/2nd_it1/epoch\=349-val_class_agnostic_ap\=0.074.ckpt"
CHECKPOINT_PATH2="/home/luebberstedt/ovseg/saved/cycle_2_last-epoch.ckpt"
CHECKPOINT_PATH3="/home/luebberstedt/ovseg/saved/cycle_3_last-epoch.ckpt"
CHECKPOINT_PATH4="/home/luebberstedt/ovseg/saved/cycle_4_last-epoch.ckpt"
DBSCAN_EPS=0.95
EXP_NAME_BASE="eval_v2_"

# python main_openvocab_instance_segmentation.py \
# "general.experiment_name=${EXPERIMENT_NAME}" \
# "general.checkpoint=${CHECKPOINT_PATH}" \
# "general.train_mode=false" \
# "general.use_dbscan=true" \
# "general.dbscan_eps=${DBSCAN_EPS}" \
# "general.save_visualizations=true"

nice -n 15 python main_openvocab_instance_segmentation.py \
"general.experiment_name=${EXP_NAME_BASE}1" \
"general.checkpoint=${CHECKPOINT_PATH1}" \
"general.train_mode=false" \
"general.use_dbscan=true" \
"general.dbscan_eps=${DBSCAN_EPS}" \
"general.save_visualizations=true"

nice -n 15 python main_openvocab_instance_segmentation.py \
"general.experiment_name=${EXP_NAME_BASE}2" \
"general.checkpoint=${CHECKPOINT_PATH2}" \
"general.train_mode=false" \
"general.use_dbscan=true" \
"general.dbscan_eps=${DBSCAN_EPS}" \
"general.save_visualizations=true"

nice -n 15 python main_openvocab_instance_segmentation.py \
"general.experiment_name=${EXP_NAME_BASE}3" \
"general.checkpoint=${CHECKPOINT_PATH3}" \
"general.train_mode=false" \
"general.use_dbscan=true" \
"general.dbscan_eps=${DBSCAN_EPS}" \
"general.save_visualizations=true"

nice -n 15 python main_openvocab_instance_segmentation.py \
"general.experiment_name=${EXP_NAME_BASE}4" \
"general.checkpoint=${CHECKPOINT_PATH4}" \
"general.train_mode=false" \
"general.use_dbscan=true" \
"general.dbscan_eps=${DBSCAN_EPS}" \
"general.save_visualizations=true"