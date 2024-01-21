import colorsys
import math
from typing import List, Tuple
import hydra
from omegaconf import DictConfig
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
import pyviz3d.visualizer as vis
import random
import json
import functools
from tqdm import tqdm

CALC_BBOX_METRICS = False


@functools.lru_cache(20)
def get_evenly_distributed_colors(
    count: int,
) -> List[Tuple[np.uint8, np.uint8, np.uint8]]:
    # lru cache caches color tuples
    HSV_tuples = [(x / count, 1.0, 1.0) for x in range(count)]
    random.shuffle(HSV_tuples)
    return list(
        map(
            lambda x: (np.array(colorsys.hsv_to_rgb(*x)) * 255).astype(np.uint8),
            HSV_tuples,
        )
    )


# taken from trainer class
def save_visualizations(
    target_full,
    full_res_coords,
    sorted_masks,
    sort_classes,
    file_name,
    original_colors,
    original_normals,
    ds,
    save_base_dir,
    point_size=20,
):
    full_res_coords -= full_res_coords.mean(axis=0)

    gt_pcd_pos = []
    gt_pcd_normals = []
    gt_pcd_color = []
    gt_inst_pcd_color = []
    gt_boxes = []

    if "labels" in target_full:
        instances_colors = torch.from_numpy(
            np.vstack(get_evenly_distributed_colors(target_full["labels"].shape[0]))
        )
        for instance_counter, (label, mask) in enumerate(
            zip(target_full["labels"], target_full["masks"])
        ):
            if label == 255:
                continue

            mask_tmp = mask.detach().cpu().numpy()
            mask_coords = full_res_coords[mask_tmp.astype(bool), :]

            if len(mask_coords) == 0:
                continue

            gt_pcd_pos.append(mask_coords)
            mask_coords_min = full_res_coords[mask_tmp.astype(bool), :].min(axis=0)
            mask_coords_max = full_res_coords[mask_tmp.astype(bool), :].max(axis=0)
            size = mask_coords_max - mask_coords_min
            mask_coords_middle = mask_coords_min + size / 2

            gt_boxes.append(
                {
                    "position": mask_coords_middle,
                    "size": size,
                    "color": ds.map2color([label])[0],
                }
            )

            gt_pcd_color.append(
                ds.map2color([label]).repeat(gt_pcd_pos[-1].shape[0], 1)
            )
            gt_inst_pcd_color.append(
                instances_colors[instance_counter % len(instances_colors)]
                .unsqueeze(0)
                .repeat(gt_pcd_pos[-1].shape[0], 1)
            )

            gt_pcd_normals.append(original_normals[mask_tmp.astype(bool), :])

        gt_pcd_pos = np.concatenate(gt_pcd_pos)
        gt_pcd_normals = np.concatenate(gt_pcd_normals)
        gt_pcd_color = np.concatenate(gt_pcd_color)
        gt_inst_pcd_color = np.concatenate(gt_inst_pcd_color)

    v = vis.Visualizer()

    v.add_points(
        "RGB Input",
        full_res_coords,
        colors=original_colors,
        normals=original_normals,
        visible=True,
        point_size=point_size,
    )

    if "labels" in target_full:
        v.add_points(
            "Semantics (GT)",
            gt_pcd_pos,
            colors=gt_pcd_color,
            normals=gt_pcd_normals,
            alpha=0.8,
            visible=False,
            point_size=point_size,
        )
        v.add_points(
            "Instances (GT)",
            gt_pcd_pos,
            colors=gt_inst_pcd_color,
            normals=gt_pcd_normals,
            alpha=0.8,
            visible=False,
            point_size=point_size,
        )

    pred_coords = []
    pred_normals = []
    pred_sem_color = []
    pred_inst_color = []

    for did in range(len(sorted_masks)):
        instances_colors = torch.from_numpy(
            np.vstack(get_evenly_distributed_colors(max(1, sorted_masks[did].shape[1])))
        )

        for i in reversed(range(sorted_masks[did].shape[1])):
            coords = full_res_coords[sorted_masks[did][:, i].astype(bool), :]

            mask_coords = full_res_coords[sorted_masks[did][:, i].astype(bool), :]
            mask_normals = original_normals[sorted_masks[did][:, i].astype(bool), :]

            label = sort_classes[did][i]

            if len(mask_coords) == 0:
                continue

            pred_coords.append(mask_coords)
            pred_normals.append(mask_normals)

            pred_sem_color.append(ds.map2color([label]).repeat(mask_coords.shape[0], 1))

            pred_inst_color.append(
                instances_colors[i % len(instances_colors)]
                .unsqueeze(0)
                .repeat(mask_coords.shape[0], 1)
            )

        if len(pred_coords) > 0:
            pred_coords = np.concatenate(pred_coords)
            pred_normals = np.concatenate(pred_normals)
            pred_sem_color = np.concatenate(pred_sem_color)
            pred_inst_color = np.concatenate(pred_inst_color)

            v.add_points(
                "Semantics (Mask3D)",
                pred_coords,
                colors=pred_sem_color,
                normals=pred_normals,
                visible=False,
                alpha=0.8,
                point_size=point_size,
            )
            v.add_points(
                "Instances (Mask3D)",
                pred_coords,
                colors=pred_inst_color,
                normals=pred_normals,
                visible=False,
                alpha=0.8,
                point_size=point_size,
            )

    v.save(os.path.join(save_base_dir, file_name), verbose=False)


def get_predicted_class_ids(target_ncut, label_offset, ds_gt, ds_ncut, device, cfg):
    ncut_instance_feats = target_ncut[0]["instance_feats"]
    ncut_instance_feats = ncut_instance_feats.to(device)
    ncut_instance_feats = F.normalize(ncut_instance_feats, p=2, dim=-1)

    all_labels = np.arange(cfg.data.num_labels - label_offset)
    all_labels[0] = -1  # chair fix
    all_class_ids = ds_gt._remap_model_output(all_labels + label_offset)
    all_text_label_feats = ds_ncut.map2features(
        all_class_ids, ncut_instance_feats.device
    )

    cos_sims = ncut_instance_feats @ all_text_label_feats.T
    pred_classes = cos_sims.argmax(dim=-1)
    pred_classes[pred_classes == 0] = -1  # chair fix
    pred_class_ids = ds_gt._remap_model_output(pred_classes.cpu() + label_offset)
    return pred_class_ids


def get_bbox_gt(gt_target_full_res, orig_coords):
    bbox_data = []
    for inst_id in range(gt_target_full_res["masks"].shape[0]):
        if gt_target_full_res["labels"][inst_id].item() == 255:
            continue

        obj_coords = orig_coords[
            gt_target_full_res["masks"][inst_id, :].cpu().detach().numpy().astype(bool),
            :,
        ]
        if obj_coords.shape[0] > 0:
            obj_center = obj_coords.mean(axis=0)
            obj_axis_length = obj_coords.max(axis=0) - obj_coords.min(axis=0)

            bbox = np.concatenate((obj_center, obj_axis_length))
            bbox_data.append(
                (
                    gt_target_full_res["labels"][inst_id].item(),
                    bbox,
                )
            )
    return bbox_data


def get_bbox_ncut(ncut_target_full_res, orig_coords, pred_class_ids):
    bbox_data = []
    for inst_id in range(ncut_target_full_res["masks"].shape[0]):
        obj_coords = orig_coords[
            ncut_target_full_res["masks"][inst_id, :]
            .cpu()
            .detach()
            .numpy()
            .astype(bool),
            :,
        ]
        if obj_coords.shape[0] > 0:
            obj_center = obj_coords.mean(axis=0)
            obj_axis_length = obj_coords.max(axis=0) - obj_coords.min(axis=0)

            bbox = np.concatenate((obj_center, obj_axis_length))
            bbox_data.append((pred_class_ids[inst_id], bbox, 1.0))  # fake score of 1.0
    return bbox_data


def semantic_evaluation(ap_results, preds, gt_data_path, pred_path, ds_gt, log_prefix):
    from ovseg.benchmark.openvocab_evaluate_semantic_instance import evaluate
    from datasets.scannet200.scannet200_splits import (
        HEAD_CATS_SCANNET_200,
        TAIL_CATS_SCANNET_200,
        COMMON_CATS_SCANNET_200,
        VALID_CLASS_IDS_200_VALIDATION,
    )

    evaluate(
        preds,
        gt_data_path,
        pred_path,
        dataset=ds_gt.dataset_name,
    )
    with open(pred_path, "r") as fin:
        head_results, common_results, tail_results = [], [], []
        for line_id, line in enumerate(fin):
            if line_id == 0:
                # ignore header
                continue
            (
                class_name,
                class_id,
                ap,
                ap_50,
                ap_25,
                ar,
                ar_50,
                ar_25,
            ) = line.strip().split(",")

            if class_name in VALID_CLASS_IDS_200_VALIDATION:
                ap_results[f"{log_prefix}_{class_name}_val_ap"] = float(ap)
                ap_results[f"{log_prefix}_{class_name}_val_ap_50"] = float(ap_50)
                ap_results[f"{log_prefix}_{class_name}_val_ap_25"] = float(ap_25)

                if class_name in HEAD_CATS_SCANNET_200:
                    head_results.append(
                        np.array((float(ap), float(ap_50), float(ap_25)))
                    )
                elif class_name in COMMON_CATS_SCANNET_200:
                    common_results.append(
                        np.array((float(ap), float(ap_50), float(ap_25)))
                    )
                elif class_name in TAIL_CATS_SCANNET_200:
                    tail_results.append(
                        np.array((float(ap), float(ap_50), float(ap_25)))
                    )
                else:
                    assert False, "class not known!"
    head_results = np.stack(head_results)
    common_results = np.stack(common_results)
    tail_results = np.stack(tail_results)
    mean_tail_results = np.nanmean(tail_results, axis=0)
    mean_common_results = np.nanmean(common_results, axis=0)
    mean_head_results = np.nanmean(head_results, axis=0)
    ap_results[f"{log_prefix}_mean_tail_ap_25"] = mean_tail_results[0]
    ap_results[f"{log_prefix}_mean_common_ap_25"] = mean_common_results[0]
    ap_results[f"{log_prefix}_mean_head_ap_25"] = mean_head_results[0]
    ap_results[f"{log_prefix}_mean_tail_ap_50"] = mean_tail_results[1]
    ap_results[f"{log_prefix}_mean_common_ap_50"] = mean_common_results[1]
    ap_results[f"{log_prefix}_mean_head_ap_50"] = mean_head_results[1]
    ap_results[f"{log_prefix}_mean_tail_ap_25"] = mean_tail_results[2]
    ap_results[f"{log_prefix}_mean_common_ap_25"] = mean_common_results[2]
    ap_results[f"{log_prefix}_mean_head_ap_25"] = mean_head_results[2]
    overall_ap_results = np.nanmean(
        np.vstack((head_results, common_results, tail_results)),
        axis=0,
    )
    ap_results[f"{log_prefix}_mean_ap"] = overall_ap_results[0]
    ap_results[f"{log_prefix}_mean_ap_50"] = overall_ap_results[1]
    ap_results[f"{log_prefix}_mean_ap_25"] = overall_ap_results[2]
    ap_results = {
        key: 0.0 if math.isnan(score) else score for key, score in ap_results.items()
    }
    return ap_results


def class_agnostic_evaluation(
    ap_results, preds, gt_data_path, pred_path, ds_gt, log_prefix
):
    """
    DO NOT CALL BEFORE REGULAR SEMANTIC EVALUATION, WILL MODIFY PREDS INPLACE
    """
    from ovseg.benchmark.openvocab_evaluate_semantic_instance import evaluate
    from benchmark import util_3d

    # set all predicted and gt classes to the same value to become class agnostic
    gt_dict = {}
    for k, v in preds.items():
        n_points, n_instances = v["pred_masks"].shape
        v["pred_classes"] = np.full(
            shape=(n_instances,), fill_value=2  # first valid class id
        )
        gt_file = os.path.join(gt_data_path, k + ".txt")
        gt_ids = util_3d.load_ids(
            gt_file
        )  # instances are encoded in lower 3 digits, semantic classes in upper
        gt_dict[k] = (gt_ids % 1000) + 2000  # first valid class id

    evaluate(
        preds,
        gt_data_path,
        pred_path,
        dataset=ds_gt.dataset_name,
        gt_dict=gt_dict,
    )
    with open(pred_path, "r") as fin:
        it = iter(fin)
        next(it)  # skip header
        line = next(it)
        (
            _,
            _,
            ap,
            ap_50,
            ap_25,
            ar,
            ar_50,
            ar_25,
        ) = line.strip().split(",")
        ap_results[f"{log_prefix}_ap"] = float(ap)
        ap_results[f"{log_prefix}_ap_50"] = float(ap_50)
        ap_results[f"{log_prefix}_ap_25"] = float(ap_25)
    return ap_results


@hydra.main(
    config_path="../../conf", config_name="config_base_instance_segmentation.yaml"
)
def main(cfg: DictConfig):
    hydra_dir = os.getcwd()
    os.chdir(hydra.utils.get_original_cwd())
    sys.path.append(hydra.utils.get_original_cwd())

    from utils.votenet_utils.eval_det import eval_det

    ds_gt = hydra.utils.instantiate(cfg.ncut.eval.gt_data.dataset)
    ds_ncut = hydra.utils.instantiate(cfg.ncut.eval.ncut_data.dataset)
    c_fn_gt = hydra.utils.instantiate(
        cfg.data.validation_collation_original,
        filter_out_classes=ds_gt.filter_out_classes,
    )
    c_fn_ncut = hydra.utils.instantiate(
        cfg.data.validation_collation, filter_out_classes=ds_ncut.filter_out_classes
    )
    loader_gt = hydra.utils.instantiate(
        cfg.ncut.eval.gt_data.dataloader,
        dataset=ds_gt,
        collate_fn=c_fn_gt,
    )
    loader_ncut = hydra.utils.instantiate(
        cfg.ncut.eval.ncut_data.dataloader,
        dataset=ds_ncut,
        collate_fn=c_fn_ncut,
    )
    assert len(loader_gt) == len(loader_ncut), "different number of scenes in datasets"

    device = torch.device("cuda:0")

    label_offset = ds_gt.label_offset

    print("Loading predictions...")
    bbox_gt = {}
    bbox_ncut = {}
    preds = {}
    for i, (batch_gt, batch_ncut) in tqdm(
        enumerate(zip(loader_gt, loader_ncut)), total=len(loader_gt)
    ):
        data_gt, _, file_names = batch_gt
        data_ncut, target_ncut, _ = batch_ncut

        pred_class_ids = get_predicted_class_ids(
            target_ncut, label_offset, ds_gt, ds_ncut, device, cfg
        )

        # remap gt labels
        gt_target_full_res = data_gt.target_full[0]
        gt_target_full_res["labels"][gt_target_full_res["labels"] == 0] = -1
        gt_target_full_res["labels"] = ds_gt._remap_model_output(
            gt_target_full_res["labels"].cpu() + label_offset
        )

        # visualize
        orig_coords = data_gt.original_coordinates[0]
        pred_masks = data_ncut.target_full[0]["masks"].T.numpy()
        file_name = file_names[0]
        orig_colors = data_gt.original_colors[0]
        orig_normals = data_gt.original_normals[0]
        save_visualizations(
            target_full=gt_target_full_res,
            full_res_coords=orig_coords,
            sorted_masks=[pred_masks],  # [np(n_points_full_res, n_pred_instances)]
            sort_classes=[pred_class_ids],  # [np(n_pred_instances)] - class ids
            file_name=file_name,  # str (dirname => without html ending)
            original_colors=orig_colors,  # np(n_points_full, 3)
            original_normals=orig_normals,  # np(n_points_full, 3)
            ds=ds_gt,
            save_base_dir=cfg.ncut.eval.res_dir,
        )

        # save for metric computation
        ncut_target_full_res = data_ncut.target_full[0]
        if CALC_BBOX_METRICS:
            # gt bounding boxes
            bbox_gt[file_name] = get_bbox_gt(
                gt_target_full_res, orig_coords
            )  # [(class_id, (bbox_center, bb_axis_len)), ...]

            # pseudo gt bounding boxes
            bbox_ncut[file_name] = get_bbox_ncut(
                ncut_target_full_res, orig_coords, pred_class_ids
            )  # [(class_id, (bbox_center, bb_axis_len)), ...]
        # full res pseudo gt (real gt read from dataset by official evaluation function)
        n_inst = ncut_target_full_res["masks"].shape[0]
        preds[file_name] = {
            "pred_masks": pred_masks,
            "pred_scores": np.full(fill_value=1.0, shape=(n_inst)),
            "pred_classes": pred_class_ids,
        }

    log_prefix = f"val"
    ap_results = {}

    # bounding box based evaluation
    if CALC_BBOX_METRICS:
        (
            box_rec_25,  # {class_id: np(n_inst_class)}
            box_prec_25,  # {class_id: np(n_inst_class)}
            box_ap_25,  # {class_id: float}
        ) = eval_det(bbox_ncut, bbox_gt, ovthresh=0.25, use_07_metric=False)
        (
            box_rec_50,
            box_prec_50,
            box_ap_50,
        ) = eval_det(bbox_ncut, bbox_gt, ovthresh=0.5, use_07_metric=False)
        mean_box_ap_25 = sum([v for k, v in box_ap_25.items()]) / len(
            box_ap_25.keys()
        )  # float
        mean_box_ap_50 = sum([v for k, v in box_ap_50.items()]) / len(box_ap_50.keys())
        ap_results[f"{log_prefix}_mean_box_ap_25"] = mean_box_ap_25
        ap_results[f"{log_prefix}_mean_box_ap_50"] = mean_box_ap_50
        for class_id in box_ap_50.keys():
            class_name = ds_gt.label_info[class_id]["name"]
            ap_results[f"{log_prefix}_{class_name}_val_box_ap_50"] = box_ap_50[class_id]
        for class_id in box_ap_25.keys():
            class_name = ds_gt.label_info[class_id]["name"]
            ap_results[f"{log_prefix}_{class_name}_val_box_ap_25"] = box_ap_25[class_id]

    # official scannet evaluation
    base_path = cfg.ncut.eval.res_dir
    gt_data_path = f"{ds_gt.data_dir[0]}/instance_gt/{ds_gt.mode}"
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    try:
        semantic_pred_path = f"{base_path}/tmp_output_semantic.txt"
        ap_results = semantic_evaluation(  # modifies ap_results inplace
            ap_results, preds, gt_data_path, semantic_pred_path, ds_gt, log_prefix
        )
        class_agnostic_pred_path = f"{base_path}/tmp_output_class_agnostic.txt"
        ap_results = class_agnostic_evaluation(  # modifies ap_results inplace
            ap_results,
            preds,
            gt_data_path,
            class_agnostic_pred_path,
            ds_gt,
            "val_class_agnostic",
        )
    except (IndexError, OSError) as e:
        print("NO SCORES!!!")
        ap_results[f"{log_prefix}_mean_ap"] = 0.0
        ap_results[f"{log_prefix}_mean_ap_50"] = 0.0
        ap_results[f"{log_prefix}_mean_ap_25"] = 0.0

    formatted_ap_results = json.dumps(ap_results, sort_keys=True, indent=4)
    with open(os.path.join(base_path, "scannet_eval_results.json"), "w") as fl:
        fl.write(formatted_ap_results)
    print(formatted_ap_results)

    print()
    print("=====================")
    print("Results saved under:", base_path)


if __name__ == "__main__":
    main()
