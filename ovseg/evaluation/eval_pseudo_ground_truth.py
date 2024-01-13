import colorsys
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
import functools


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

    v.save(os.path.join(save_base_dir, file_name))


@hydra.main(
    config_path="../../conf", config_name="config_base_instance_segmentation.yaml"
)
def main(cfg: DictConfig):
    hydra_dir = os.getcwd()
    os.chdir(hydra.utils.get_original_cwd())
    sys.path.append(hydra.utils.get_original_cwd())

    c_fn_gt = hydra.utils.instantiate(cfg.data.validation_collation_original)
    c_fn_ncut = hydra.utils.instantiate(cfg.data.validation_collation)
    ds_gt = hydra.utils.instantiate(cfg.ncut.eval.gt_data.dataset)
    ds_ncut = hydra.utils.instantiate(cfg.ncut.eval.ncut_data.dataset)
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

    device = torch.device("cuda:0")

    label_offset = ds_gt.label_offset

    for batch_gt, batch_ncut in zip(loader_gt, loader_ncut):
        data_gt, target_gt, filenames_gt = batch_gt
        data_ncut, target_ncut, _ = batch_ncut

        # compute predicted classes
        ncut_instance_feats = target_ncut[0]["instance_feats"]
        ncut_instance_feats = ncut_instance_feats.to(device)
        ncut_instance_feats = F.normalize(ncut_instance_feats, p=2, dim=-1)

        all_labels = np.arange(cfg.data.num_labels - label_offset)
        all_labels[0] = -1  # chair fix
        all_class_ids = ds_ncut._remap_model_output(all_labels + label_offset)
        all_text_label_feats = ds_ncut.map2features(
            all_class_ids, ncut_instance_feats.device
        )

        cos_sims = ncut_instance_feats @ all_text_label_feats.T
        pred_classes = cos_sims.argmax(dim=-1)
        pred_classes[pred_classes == 0] = -1  # chair fix
        pred_class_ids = ds_ncut._remap_model_output(pred_classes.cpu() + label_offset)

        # print("all_labels", all_labels)
        # print("all_class_ids", all_class_ids)
        # print("pred_classes", pred_classes)
        # print("pred class ids", pred_class_ids)

        # visualize
        gt_target_full_res = data_gt.target_full[0]
        gt_target_full_res["labels"][gt_target_full_res["labels"] == 0] = -1
        gt_target_full_res["labels"] = ds_gt._remap_model_output(
            gt_target_full_res["labels"].cpu() + label_offset
        )
        orig_coords = data_gt.original_coordinates[0]
        pred_masks = data_ncut.target_full[0]["masks"].T.numpy()
        file_name = filenames_gt[0]
        orig_colors = data_gt.original_colors[0]
        orig_normals = data_gt.original_normals[0]
        save_visualizations(
            target_full=gt_target_full_res,
            full_res_coords=orig_coords,
            sorted_masks=[pred_masks],  # [np(n_points_full_res, n_pred_instances)]
            sort_classes=[pred_class_ids],  # [np(n_pred_instances)] - class ids
            file_name=file_name,  # str (witout html ending)
            original_colors=orig_colors,  # np(n_points_full, 3)
            original_normals=orig_normals,  # np(n_points_full, 3)
            ds=ds_gt,
            save_base_dir=cfg.ncut.eval.res_dir,
        )

        # remap gt labels

        # generate bounding boxes for ncut

        # generate bounding boxes for gt
        exit(0)


if __name__ == "__main__":
    main()
