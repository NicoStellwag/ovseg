import hydra
import os
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import clip

EXPORT_DIR = (
    "/home/stellwag/dev/ovseg/eval_output/instance_evaluation_test_2_49/decoder_-1"
)
QUERIES = ["a chair in a scene", "a table in a scene"]
TOP_K = 3
SAVE_DIR = "/mnt/hdd/corr_viz/"


@hydra.main(
    config_path="../../conf", config_name="config_base_instance_segmentation.yaml"
)
def main(cfg):
    hydra_dir = os.getcwd()
    os.chdir(hydra.utils.get_original_cwd())
    sys.path.append(hydra.utils.get_original_cwd())

    from ovseg.evaluation.trainer_visualization import save_visualizations

    Path(SAVE_DIR).mkdir(parents=True, exist_ok=True)

    # compute query features
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    clip_model, _ = clip.load("ViT-B/32", device=device)
    queries = clip.tokenize(QUERIES).to(device)
    with torch.no_grad():
        query_feats = clip_model.encode_text(queries)
    query_feats = F.normalize(query_feats, p=2, dim=-1)
    query_feats = query_feats.cpu().numpy()  # np(n_queries, dim_feature)

    ds = hydra.utils.instantiate(cfg.data.validation_dataset)
    collate_fn = hydra.utils.instantiate(cfg.data.validation_collation)
    loader = hydra.utils.instantiate(
        cfg.data.validation_dataloader, ds, collate_fn=collate_fn
    )
    for batch in loader:
        data, target, file_names = batch

        file_name = file_names[0]
        target_full = data.target_full[0]
        orig_coords = data.original_coordinates[0]
        orig_colors = data.original_colors[0]
        orig_normals = data.original_normals[0]

        export_dir = Path(EXPORT_DIR)
        export_file = export_dir / (str(file_name) + ".txt")

        if not export_file.exists:
            print(f"WARNING: no exports for {file_name}")
            continue

        predictions = []  # order: scores descending
        with open(export_file, "r") as fl:
            lines = fl.readlines()
        for line in lines:
            rel_mask_file, pred_class_id, score = line.strip().split(" ")
            mask_file = export_dir / rel_mask_file
            feature_file = export_dir / rel_mask_file.replace(".txt", "_feature.npy")
            pred_feature = np.load(feature_file)
            pred_feature = F.normalize(
                torch.from_numpy(pred_feature), p=2, dim=0
            ).numpy()

            # compute correspondence to queries
            cos_sims = pred_feature @ query_feats.T  # np(n_queries)
            correspondences = (cos_sims + 1) / 2

            predictions.append(
                {
                    "mask": np.loadtxt(mask_file).astype(bool),  # np(n_points,)
                    "query_correspondences": correspondences,  # np(n_queries,)
                    "class_id": int(pred_class_id),
                    "score": float(score),
                }
            )

        # prepare stuff for istance visualization
        sorted_masks = np.hstack([p["mask"][:, None] for p in predictions])
        sorted_classes = np.array([p["class_id"] for p in predictions])
        target_full["labels"][target_full["labels"] == 0] = -1
        target_full["labels"] = ds._remap_model_output(
            target_full["labels"].cpu() + ds.label_offset
        )

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
                query_heatmaps[query]["colors"][mask, 0] = pred[
                    "query_correspondences"
                ][i]
        topk_queries = {}
        for query, query_viz in query_heatmaps.items():
            # map the red colors to blue -> red heatmap
            query_viz["colors"][:, 0] = query_viz["colors"][:, 0] * 255

            # additionally create top k correspondences overlays
            if TOP_K:
                top_k_vals, _ = torch.topk(
                    torch.from_numpy(query_viz["colors"][:, 0].flatten()).unique(),
                    k=TOP_K,
                )
                top_k_vals = top_k_vals.numpy()
                top_k_instance_masks = []
                for val in top_k_vals:
                    top_k_instance_masks.append(
                        (query_viz["colors"][:, 0] == val).astype(
                            int
                        )  # check the red channel
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

        save_visualizations(
            target_full=target_full,
            full_res_coords=orig_coords,
            sorted_masks=[sorted_masks],
            sort_classes=[sorted_classes],
            file_name=file_name,
            original_colors=orig_colors,
            original_normals=orig_normals,
            ds=ds,
            save_base_dir=SAVE_DIR,
            additional_overlays=query_heatmaps,
        )

        exit(0)


if __name__ == "__main__":
    main()
