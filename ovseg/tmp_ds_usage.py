import hydra
from omegaconf import DictConfig
import os
import numpy as np
import sys

@hydra.main(
    config_path="../conf", config_name="config_base_instance_segmentation.yaml"
)
def main(cfg: DictConfig):
    hydra_dir = os.getcwd()
    os.chdir(hydra.utils.get_original_cwd())
    sys.path.append(hydra.utils.get_original_cwd())

    ds = hydra.utils.instantiate(cfg.data.train_dataset)
    (
        coords,
        feats,
        labels,
        clip_vecs,
        scene,
        raw_color,
        raw_normals,
        raw_coords,
        idx
    ) = ds[0]

    from ncut.visualize import visualize_3d_feats, generate_random_color
    # viz augmented scene
    aug_colors = feats[:, :3]
    visualize_3d_feats(coords, aug_colors, os.path.join(hydra_dir, "augmented_scene.html"))

    # viz istance labels of augmented scene
    instance_col_map = np.stack([generate_random_color() for _ in range(clip_vecs.shape[0])])
    point_cols = instance_col_map[labels[:, 1].astype(int)]
    point_cols[labels[:, 1] == -1] = np.array([0., 0., 0.])
    visualize_3d_feats(coords, point_cols, os.path.join(hydra_dir, "instances.html"))

if __name__ == "__main__":
    main()