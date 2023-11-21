import sys
from pathlib import Path
import hydra
import torch
import torch.nn as nn
import torchvision.transforms.functional as F
import open3d as o3d
import numpy as np
import MinkowskiEngine as ME
from omegaconf import DictConfig


def get_feature_extractor(cfg_fe3d: DictConfig) -> nn.Module:
    """
    Instantiates a minkowski resunet and loads pretrained weights.
    """
    model = hydra.utils.instantiate(cfg_fe3d.model)
    model.load_state_dict(torch.load(cfg_fe3d.model.pretrained_weights)["state_dict"])
    return model


def load_data(cfg_data):
    """
    feats, coords
    SparseTensor(feats, coords)
    """
    mesh_files = [
        str(i) for i in Path(cfg_data.dataset_dir).rglob(cfg_data.mesh_filename_pattern)
    ]
    for mesh_file in mesh_files[:1]:
        scene_mesh = o3d.io.read_triangle_mesh(mesh_file)
        coords = np.array(scene_mesh.vertices)
        coords = np.hstack(
            [coords, np.zeros(coords.shape[0])[:, None]]
        )  # ME.SparseTensor adds the batch dim as last dim in coords when loaded through dataloader
        colors = np.array(scene_mesh.vertex_colors)
        yield F.to_tensor(colors)[0].type(torch.float), F.to_tensor(coords)[0].type(
            torch.int
        )


@hydra.main(
    config_path="../../conf", config_name="config_base_instance_segmentation.yaml"
)
def main(cfg: DictConfig):
    # necessary for hydra object instantiation when script is not in project root
    sys.path.append(hydra.utils.get_original_cwd())

    device = torch.device("cuda:0")

    model = get_feature_extractor(cfg.ncut.feature_extraction_3d)
    model = model.eval().to(device)

    for colors, coords in load_data(cfg.ncut.feature_extraction_3d.data):
        colors, coords = colors.to(device), coords.to(device)
        x = ME.SparseTensor(features=colors, coordinates=coords)
        _, feature_maps = model(x)
        print(feature_maps)


if __name__ == "__main__":
    main()
