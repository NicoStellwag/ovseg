import hydra
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
import os
import numpy as np
import sys
import clip


@hydra.main(config_path="../conf", config_name="config_base_instance_segmentation.yaml")
def main(cfg: DictConfig):
    hydra_dir = os.getcwd()
    os.chdir(hydra.utils.get_original_cwd())
    sys.path.append(hydra.utils.get_original_cwd())

    from ncut.visualize import visualize_3d_feats, generate_random_color
    from ovseg.feature_dim_reduction.savable_pca import SavablePCA

    # get a sample
    c_fn = hydra.utils.instantiate(cfg.data.validation_collation)
    ds = hydra.utils.instantiate(cfg.data.validation_dataset)
    loader = hydra.utils.instantiate(
        cfg.data.validation_dataloader,
        dataset=ds,
        collate_fn=c_fn,
    )
    data, target, filenames = next(iter(loader))
    orig_labels = data.original_labels[0]
    feats = data.original_colors[0]
    coords = data.original_coordinates[0]
    voxel_to_full_res = data.inverse_maps[0]
    labels = target[0]["labels"].numpy()
    instance_feats = target[0]["instance_feats"].numpy()
    instance_masks = target[0]["masks"].numpy()
    n_points = len(orig_labels)
    n_instances = len(labels)
    # ds = hydra.utils.instantiate(cfg.data.validation_dataset)
    # (
    #     coords,
    #     feats,
    #     labels,
    #     clip_vecs,
    #     scene,
    #     raw_color,
    #     raw_normals,
    #     raw_coords,
    #     idx,
    # ) = ds[0]

    # # make sense of label remapping process
    # labels = np.arange(198) # num classes - label_offset (ignore floor, wall)
    # labels[0] = -1  # remap first logit to chair instead of floor
    # ids = ds._remap_model_output(labels + 2)
    # print("labels")
    # print(labels)
    # print("ids")
    # print(ids)
    # print("label info keys")
    # print(list(ds.label_info.keys()))
    # print(len(labels), len(list(ds.label_info.keys())))

    # viz augmented scene
    aug_colors = feats[:, :3]
    visualize_3d_feats(
        coords, aug_colors, os.path.join(hydra_dir, "augmented_scene.html")
    )

    # viz instance labels of augmented scene
    remapped_labels = ds._remap_model_output(labels + ds.label_offset)  # instance wise
    text_labels = [ds._labels[i]["name"] for i in remapped_labels]  # instance wise
    instance_col_map = np.stack([generate_random_color() for _ in range(n_instances)])
    point_cols = np.zeros(shape=(n_points, 3))
    point_texts = np.full(shape=(n_points,), fill_value="", dtype=object)
    for i in range(n_instances):
        mask = instance_masks[i][voxel_to_full_res]
        point_cols[mask] = instance_col_map[i]
        point_texts[mask] = text_labels[i] + "_" + str(i)
    visualize_3d_feats(
        coords,
        point_cols,
        os.path.join(hydra_dir, "instances.html"),
        hovertext=point_texts.tolist(),
    )

    # viz correspondence without dim reduction
    query_string = "tv"
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    clip_model, _ = clip.load("ViT-B/32", device=device)
    text = clip.tokenize([query_string]).to(device)
    with torch.no_grad():
        query_feat = clip_model.encode_text(text)
    query_feat = F.normalize(query_feat, p=2, dim=-1).cpu().numpy()
    instance_feats = F.normalize(torch.from_numpy(instance_feats), p=2, dim=-1).numpy()
    cos_sim = ((query_feat * instance_feats).sum(-1) + 1) / 2
    cos_cols = np.zeros(shape=(n_points, 3))
    for i in range(n_instances):
        mask = instance_masks[i][voxel_to_full_res]
        cos_cols[mask] = cos_sim[i]
    visualize_3d_feats(
        coords, cos_cols, os.path.join(hydra_dir, "correspondence_full_dim.html")
    )

    # viz correspondence with dim reduction
    dim_red = SavablePCA.from_file(cfg.data.feature_dim_reduction_path)
    instance_feats = dim_red.transform(instance_feats)
    instance_feats = F.normalize(torch.from_numpy(instance_feats), p=2, dim=-1).numpy()
    query_feat = dim_red.transform(query_feat)
    query_feat = F.normalize(torch.from_numpy(query_feat), p=2, dim=-1).numpy()
    cos_sim = ((query_feat * instance_feats).sum(-1) + 1) / 2
    cos_cols = np.zeros(shape=(n_points, 3))
    for i in range(n_instances):
        mask = instance_masks[i][voxel_to_full_res]
        cos_cols[mask] = cos_sim[i]
    visualize_3d_feats(
        coords, cos_cols, os.path.join(hydra_dir, "correspondence_lower_dim.html")
    )

    # compare cos sim to correct label with that of random other ones
    label_feats = ds.map2features(remapped_labels, "cpu")  # already normalized
    cos_sims = (label_feats * torch.from_numpy(instance_feats)).sum(-1)
    random_cos_sims = torch.zeros((n_instances,))
    for i in range(10):
        random_label_feats = ds.randfeats(len(remapped_labels), "cpu")
        random_cos_sims += (random_label_feats * torch.from_numpy(instance_feats)).sum(
            -1
        )
    random_cos_sims /= 10
    for i in range(n_instances):
        print(
            i,
            text_labels[i],
            "correct",
            torch.round(cos_sims[i], decimals=3).item(),
            "random",
            torch.round(random_cos_sims[i], decimals=3).item(),
        )
    print()
    print("matching mean", cos_sims.mean())
    print("random mean", random_cos_sims.mean())


if __name__ == "__main__":
    main()
