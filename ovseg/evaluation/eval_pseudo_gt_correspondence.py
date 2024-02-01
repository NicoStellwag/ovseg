import clip
from pathlib import Path
import hydra
from omegaconf import DictConfig
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

NCUT_EVAL_DIR = "/mnt/hdd/ncut_eval/lseg_iterative_0.70_fulldim_2d3d"
QUERIES = [
    "mock"
    # "a fridge in a scene",
    # "a couch in a scene",
    # "a chair in a scene",
    # "a shelf in a scene",
    # "a kitchen counter in a scene",
    # "a washing machine in a scene",
    # "a trashcan in a scene",
    # "a desk in a scene",
    # "a computer in a scene",
    # "a window in a scene",
    # "a door in a scene",
    # "a computer in a scene",
    # "a toilet in a scene",
    # "a sink in a scene",
    # "stairs in a scene",
    # "washing machines in a scene",
    # "a painting on the wall",
]
TOP_K = 3
SAVE_DIR = "/mnt/hdd/viz_poster/segments"
FIRST_N_SCENES = 10


def get_predicted_class_ids(
    target_ncut, label_offset, topk, ds_gt, ds_ncut, device, cfg
):
    """
    returns top k classes np(n_inst, topk) for the featues
    the topk dimension is ordered descending w.r.t. corresponding cosine sims
    (col 0 contains the most likely class, col topk the least likely among the top k)
    """
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
    _, pred_classes = cos_sims.topk(k=topk, dim=-1, sorted=True)
    pred_classes[pred_classes == 0] = -1  # chair fix
    pred_classes = pred_classes.cpu().numpy()
    pred_classes_flat = pred_classes.flatten()
    pred_class_ids_flat = ds_gt._remap_model_output(pred_classes_flat + label_offset)
    pred_class_ids = pred_class_ids_flat.reshape(pred_classes.shape)
    return pred_class_ids


@hydra.main(
    config_path="../../conf", config_name="config_base_instance_segmentation.yaml"
)
def main(cfg: DictConfig):
    hydra_dir = os.getcwd()
    os.chdir(hydra.utils.get_original_cwd())
    sys.path.append(hydra.utils.get_original_cwd())

    from ovseg.evaluation.trainer_visualization import (
        save_visualizations,
        get_evenly_distributed_colors,
    )

    Path(SAVE_DIR).mkdir(parents=True, exist_ok=True)

    # compute query features
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    clip_model, _ = clip.load("ViT-B/32", device=device)
    queries = clip.tokenize(QUERIES).to(device)
    with torch.no_grad():
        query_feats = clip_model.encode_text(queries)
    query_feats = F.normalize(query_feats, p=2, dim=-1)
    query_feats = query_feats.cpu().numpy()  # np(n_queries, dim_feature)

    ds = hydra.utils.instantiate(cfg.ncut.eval.gt_data.dataset)
    collate_fn = hydra.utils.instantiate(
        cfg.data.test_collation,
        filter_out_classes=ds.filter_out_classes,
    )
    loader = hydra.utils.instantiate(
        cfg.ncut.eval.gt_data.dataloader,
        dataset=ds,
        collate_fn=collate_fn,
    )

    if FIRST_N_SCENES:
        total = min(len(loader), FIRST_N_SCENES)
    else:
        total = len(loader)
    for i, batch in tqdm(
        enumerate(loader),
        total=total,
    ):
        if i == FIRST_N_SCENES:
            break

        data_gt, target_gt, file_names = batch

        file_name = file_names[0]
        target_full = data_gt.target_full[0]
        orig_coords = data_gt.original_coordinates[0]
        orig_colors = data_gt.original_colors[0]
        orig_normals = data_gt.original_normals[0]

        # prepare stuff for visualization
        # pred_class_ids = get_predicted_class_ids(
        #     target_ncut,
        #     ds.label_offset,
        #     1,
        #     ds,
        #     ds,
        #     device,
        #     cfg,
        # )  # np(n_inst, topk)
        scan_dir = Path(NCUT_EVAL_DIR) / file_name
        sorted_masks = np.load(scan_dir / "pointwise_instances.npy").astype(
            bool
        )  # np(n_points, n_inst)
        sorted_feats = np.load(scan_dir / "instance_labels.npy")  # np(n_inst, feat_dim)
        sorted_classes = np.full(shape=(sorted_masks.shape[1]), fill_value=3)
        # target_full["labels"][target_full["labels"] == 0] = -1
        # target_full["labels"] = ds._remap_model_output(
        #     target_full["labels"].cpu() + ds.label_offset
        # )

        # compute query correspondences
        predictions = []
        n_inst = len(sorted_classes)
        for i in range(n_inst):
            pred_feature = sorted_feats[i, :]
            pred_feature = F.normalize(
                torch.from_numpy(pred_feature), p=2, dim=0
            ).numpy()

            cos_sims = pred_feature @ query_feats.T  # np(n_queries,)
            correspondences = (cos_sims + 1) / 2

            predictions.append(
                {
                    "mask": sorted_masks[:, i],
                    "query_correspondences": correspondences,
                }
            )

        # generate red heatmaps from correspondences
        query_heatmaps = {}
        for query in QUERIES:
            query_heatmaps[query] = {
                "coord_mask": np.full(
                    shape=(orig_coords.shape[0],), fill_value=True, dtype=bool
                ),
                "colors": np.zeros(shape=(orig_coords.shape[0], 3)),
                "normals": orig_normals.copy(),
            }
        for pred in reversed(predictions):
            for i, query in enumerate(QUERIES):
                mask = pred["mask"]
                exp_scale = lambda x: (np.exp(10 * x) - 1) / (np.exp(10) - 1)
                query_heatmaps[query]["colors"][mask, 0] = (
                    exp_scale(pred["query_correspondences"][i]) * 255
                )

        # generate overlays with top k corresponding instances
        if TOP_K:
            topk_queries = {}
            for query, query_viz in query_heatmaps.items():
                top_k_vals, _ = torch.topk(
                    torch.from_numpy(query_viz["colors"][:, 0]).unique(),
                    k=TOP_K,
                )
                top_k_vals = top_k_vals.numpy()
                top_k_instance_masks = []
                for val in top_k_vals:
                    top_k_instance_masks.append(
                        (query_viz["colors"][:, 0] == val).astype(int)
                    )
                full_mask = np.stack(top_k_instance_masks, axis=0).sum(0) > 0
                k = f"top_{TOP_K}_{query}"
                topk_queries[k] = {
                    "coord_mask": full_mask,
                    "colors": query_viz["colors"][full_mask],
                    "normals": query_viz["normals"][full_mask],
                }
                topk_queries[k]["colors"][:, 0] = 255.0
                topk_queries[k]["colors"][:, 1:] = 0.0
            query_heatmaps.update(topk_queries)

        # generate segment overlay
        point2segment = target_full["point2segment"]
        segment_cols = np.array(
            get_evenly_distributed_colors(len(np.unique(point2segment)))
        )
        point_segment_cols = segment_cols[point2segment]
        query_heatmaps["segments"] = {
            "coord_mask": np.full(
                shape=(orig_coords.shape[0],), fill_value=True, dtype=bool
            ),
            "colors": point_segment_cols,
            "normals": orig_normals,
        }

        # save to file
        save_visualizations(
            target_full=target_full,
            full_res_coords=orig_coords,
            sorted_masks=[sorted_masks],  # [np(n_points_full_res, n_pred_instances)]
            sort_classes=[sorted_classes],  # [np(n_pred_instances)] - class ids
            file_name=file_name,  # str (dirname => without html ending)
            original_colors=orig_colors,  # np(n_points_full, 3)
            original_normals=orig_normals,  # np(n_points_full, 3)
            ds=ds,
            save_base_dir=SAVE_DIR,
            additional_overlays=query_heatmaps,
        )


if __name__ == "__main__":
    main()
