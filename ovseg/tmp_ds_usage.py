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

    # c_fn = hydra.utils.instantiate(cfg.data.validation_collation)
    # ds = hydra.utils.instantiate(cfg.data.validation_dataset)
    # loader = hydra.utils.instantiate(
    #     cfg.data.validation_dataloader,
    #     dataset=ds,
    #     collate_fn=c_fn,
    # )

    ds = hydra.utils.instantiate(cfg.data.validation_dataset)
    (
        coords,
        feats,
        labels,
        clip_vecs,
        scene,
        raw_color,
        raw_normals,
        raw_coords,
        idx,
    ) = ds[0]

    # data, target, filenames = next(iter(loader))
    # point_labels = data.original_labels[0]
    # feats = data.original_colors[0]
    # coords = data.original_coordinates[0]
    # labels = target[0]["labels"]

    from ncut.visualize import visualize_3d_feats, generate_random_color
    from ovseg.feature_dim_reduction.savable_pca import SavablePCA

    # viz augmented scene
    aug_colors = feats[:, :3]
    visualize_3d_feats(
        coords, aug_colors, os.path.join(hydra_dir, "augmented_scene.html")
    )

    # viz istance labels of augmented scene
    instance_col_map = np.stack(
        [generate_random_color() for _ in range(clip_vecs.shape[0])]
    )
    point_cols = instance_col_map[labels[:, 1].astype(int)]
    point_cols[labels[:, 1] == -1] = np.array([0.0, 0.0, 0.0])
    visualize_3d_feats(coords, point_cols, os.path.join(hydra_dir, "instances.html"))

    # viz correspondence without dim reduction
    # => this works fine, so the clip instance aggregation works
    query_string = "tv"
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    clip_model, _ = clip.load("ViT-B/32", device=device)
    text = clip.tokenize([query_string]).to(device)
    with torch.no_grad():
        f = clip_model.encode_text(text)
    f = F.normalize(f, p=2, dim=-1).cpu().numpy()
    clip_vecs = F.normalize(torch.from_numpy(clip_vecs), p=2, dim=-1).numpy()
    cos_sim = ((f * clip_vecs).sum(-1) + 1) / 2
    cos_cols = np.array([[cos_sim[i.item()]] * 3 for i in labels[:, 1]])
    visualize_3d_feats(
        coords, cos_cols, os.path.join(hydra_dir, "correspondence_full_dim.html")
    )

    # viz correspondence with dim reduction
    dim_red = SavablePCA.from_file(cfg.data.feature_dim_reduction_path)
    clip_vecs = dim_red.transform(clip_vecs)
    clip_vecs = F.normalize(torch.from_numpy(clip_vecs), p=2, dim=-1).numpy()
    f = dim_red.transform(f)
    f = F.normalize(torch.from_numpy(f), p=2, dim=-1).numpy()
    cos_sim = ((f * clip_vecs).sum(-1) + 1) / 2
    cos_cols = np.array([[cos_sim[i.item()]] * 3 for i in labels[:, 1]])
    visualize_3d_feats(
        coords, cos_cols, os.path.join(hydra_dir, "correspondence_lower_dim.html")
    )


if __name__ == "__main__":
    main()
