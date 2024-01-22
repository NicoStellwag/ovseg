# source this for correct conda env!

export OMP_NUM_THREADS=3

# escape = with backslashes!
CHECKPOINT_PATH="/home/stellwag/dev/ovseg/saved/fixed_losses_feature_dropout_all_instances/epoch\=549-val_mean_ap_50\=0.004.ckpt"
EXPERIMENT_NAME="eval_it1_fixed_losses_feature_dropout_all_instances"
DBSCAN_EPS=0.95

python main_openvocab_instance_segmentation.py \
"general.experiment_name=${EXPERIMENT_NAME}" \
"general.checkpoint=${CHECKPOINT_PATH}" \
"general.train_mode=false" \
"general.use_dbscan=true" \
"general.dbscan_eps=${DBSCAN_EPS}" \
"general.save_visualizations=true"