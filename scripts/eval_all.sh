# Init all the experiment names to put the results
#eval_0="general.experiment_name=freemask_CSC_val general.checkpoint=true"
#eval_1='general.experiment_name=cutler3d_CSC_droploss_0.01 general.checkpoint=true'
#eval_2='general.experiment_name=freemask_dino_droploss_0.01 general.checkpoint=true'
#eval_3='general.experiment_name=freemask_lseg_droploss_0.1 general.checkpoint=true'

# Run all self-train experiments
# eval_0="general.experiment_name=cutler3d_DINO_self_train_2 general.checkpoint=true"
# eval_1='general.experiment_name=cutler3d_DINO_CSC_self_train_2 general.checkpoint=true'
eval_2='general.experiment_name=cutler3d_CSC_self_train_3 general.checkpoint=true'
#eval_3='general.experiment_name=freemask_lseg_self_train_1 general.checkpoint=true'

# Necessary params for standard evaluation
export EVAL_PARAMS="general.project_name=mask3d general.train_mode=false general.eval_on_segments=true data.test_batch_size=1 general.num_targets=3 data/datasets=freemask data/collation_functions=freemask_voxelize_collate general.save_visualizations=false logging=offline"
export DATA_PARAMS="data.test_dataset.data_dir=data/processed/scannet_freemask_oracle"  # scannet_freemask_oracle, arkit_freemask


# Parameters if we want to export for self train
export PHASE="data.test_dataset.mode=train_validation"  # train, validation, test, train_validation
export FREEMASK_PARAMS="general.filter_out_instances=true general.save_for_freemask=true"

# Run everything, let's gooo
# python main_instance_segmentation.py ${EVAL_PARAMS} ${DATA_PARAMS} ${PHASE} ${FREEMASK_PARAMS} ${eval_0}
# python main_instance_segmentation.py ${EVAL_PARAMS} ${DATA_PARAMS} ${PHASE} ${FREEMASK_PARAMS} ${eval_1}
python main_instance_segmentation.py ${EVAL_PARAMS} ${DATA_PARAMS} ${PHASE} ${FREEMASK_PARAMS} ${eval_2}
# python main_instance_segmentation.py ${EVAL_PARAMS} ${DATA_PARAMS} ${PHASE} ${FREEMASK_PARAMS} ${eval_3}