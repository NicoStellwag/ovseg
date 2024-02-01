import hydra
from sklearn.decomposition import PCA
import os
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import clip
from tqdm import tqdm

MODEL_EXPORT_DIR = (
    "/mnt/hdd/viz_poster/model_exports/instance_evaluation_i4_test_export"
)
QUERIES = [
    "a fridge in a scene",
    "a couch in a scene",
    "a chair in a scene",
    "a shelf in a scene",
    "a kitchen counter in a scene",
    "a washing machine in a scene",
    "a trashcan in a scene",
    "a desk in a scene",
    "a computer in a scene",
    "a window in a scene",
    "a door in a scene",
    "a computer in a scene",
    "a toilet in a scene",
    "a sink in a scene",
    "stairs in a scene",
    "washing machines in a scene",
    "a painting on the wall",
]
TOP_K = 3
SAVE_DIR = "/mnt/hdd/viz_poster/model_i4"
FIRST_N_SCENES = 10
FEATURE_CLOUDS_DIR_2D = "/mnt/hdd/ncut_features/2d/lseg"
FEATURE_CLOUDS_DIR_3D = "/mnt/hdd/ncut_features/3d/csc"


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

    ds = hydra.utils.instantiate(cfg.data.test_dataset)
    collate_fn = hydra.utils.instantiate(cfg.data.test_collation)
    loader = hydra.utils.instantiate(
        cfg.data.test_dataloader, ds, collate_fn=collate_fn
    )
    if FIRST_N_SCENES:
        total = min(len(loader), FIRST_N_SCENES)
    else:
        total = len(loader)
    for i, batch in tqdm(enumerate(loader), total=total):
        if i == FIRST_N_SCENES:
            break

        data, target, file_names = batch

        file_name = file_names[0]
        target_full = data.target_full[0]
        orig_coords = data.original_coordinates[0]
        orig_colors = data.original_colors[0]
        orig_normals = data.original_normals[0]

        export_dir = Path(MODEL_EXPORT_DIR)
        export_file = export_dir / (str(file_name) + ".txt")

        # load model predictions and compute query correspondences
        if not export_file.exists:
            print(f"WARNING: no exports for {file_name}")
            continue
        predictions = []  # order: scores descending
        with open(export_file, "r") as fl:
            lines = fl.readlines()
        for line in lines:
            rel_mask_file, score = line.strip().split(" ")
            mask_file = export_dir / rel_mask_file.replace(".npy", ".txt")
            feature_file = export_dir / rel_mask_file.replace(".npy", "_feature.npy")
            pred_feature = np.load(feature_file)
            pred_feature = F.normalize(
                torch.from_numpy(pred_feature), p=2, dim=0
            ).numpy()

            cos_sims = pred_feature @ query_feats.T  # np(n_queries,)
            correspondences = (cos_sims + 1) / 2

            predictions.append(
                {
                    "mask": np.loadtxt(mask_file).astype(bool),  # np(n_points,)
                    "query_correspondences": correspondences,  # np(n_queries,)
                    "score": float(score),
                }
            )

        # prepare stuff for istance visualization
        sorted_masks = np.hstack([p["mask"][:, None] for p in predictions])
        sorted_classes = np.array([3 for p in predictions])
        # target_full["labels"][target_full["labels"] == 0] = -1
        # target_full["labels"] = ds._remap_model_output(
        #     target_full["labels"].cpu() + ds.label_offset
        # )

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
                    torch.from_numpy(query_viz["colors"][:, 0].flatten()).unique(),
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

        # add lseg feat viz if specified
        if FEATURE_CLOUDS_DIR_2D:
            feats_file = Path(FEATURE_CLOUDS_DIR_2D) / file_name / "lseg_feats.npy"
            if not feats_file.exists:
                print("WARNING: file specified but does not exist:", feats_file)
            else:
                feats = np.load(feats_file)
                pca = PCA(n_components=3)
                reduced_feats = pca.fit_transform(feats)
                reduced_feats = (
                    (reduced_feats - reduced_feats.min()) / reduced_feats.max() * 255
                )
                query_heatmaps["2d_feats"] = {
                    "coord_mask": np.full(
                        shape=(orig_coords.shape[0],), fill_value=True, dtype=bool
                    ),
                    "colors": reduced_feats,
                    "normals": orig_normals,
                }

        # add csc feat viz if specified
        if FEATURE_CLOUDS_DIR_3D:
            feats_file = Path(FEATURE_CLOUDS_DIR_3D) / file_name / "csc_feats.npy"
            if not feats_file.exists:
                print("WARNING: file specified but does not exist:", feats_file)
            else:
                feats = np.load(feats_file)
                pca = PCA(n_components=3)
                reduced_feats = pca.fit_transform(feats)
                reduced_feats = (
                    (reduced_feats - reduced_feats.min()) / reduced_feats.max() * 255
                )
                query_heatmaps["3d_feats"] = {
                    "coord_mask": np.full(
                        shape=(orig_coords.shape[0],), fill_value=True, dtype=bool
                    ),
                    "colors": reduced_feats,
                    "normals": orig_normals,
                }

        # save to file
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


if __name__ == "__main__":
    main()
