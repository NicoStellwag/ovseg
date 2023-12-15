import sys
import os
import hydra
import torch
import torch.nn as nn
from scipy.spatial import KDTree
import numpy as np
import MinkowskiEngine as ME
from omegaconf import DictConfig


def get_feature_extractor(cfg_fe3d: DictConfig, device) -> nn.Module:
    """
    Instantiates a minkowski resunet and loads pretrained weights.
    """
    model = hydra.utils.instantiate(cfg_fe3d.model)
    model.load_state_dict(torch.load(cfg_fe3d.model.pretrained_weights)["state_dict"])
    return model.eval().to(device)


def associate_features_to_original_coords(
    y_hat: ME.SparseTensor, cfg_data: DictConfig, original_coords
):
    """
    Transforming the ply file to a minkowski engine sparse tensor is lossy
    because several points map to the same voxel.
    This function assigns the nearest neighbor's features to each of the original points.
    """
    low_res_coords = y_hat.C[:, 1:].cpu().numpy()  # slice off batch dim
    high_res_coords = original_coords / cfg_data.dataset.voxel_size
    kdtree = KDTree(low_res_coords)
    _, nn_indices = kdtree.query(high_res_coords, k=1)
    high_res_feats = y_hat.F[nn_indices].cpu().numpy()
    return high_res_feats


@hydra.main(
    config_path="../../conf", config_name="config_base_instance_segmentation.yaml"
)
def main(cfg: DictConfig):
    # necessary for hydra object instantiation when script is not in project root
    sys.path.append(hydra.utils.get_original_cwd())
    from ncut.ncut_datasets import FeatureExtractionScannet
    from ncut.visualize import visualize_3d_feats

    device = torch.device("cuda:0")

    model = get_feature_extractor(cfg.ncut.feature_extraction_3d, device)
    loader = FeatureExtractionScannet.dataloader_from_hydra(
        cfg.ncut.feature_extraction_3d.data
    )

    for sample in loader:
        sample = sample[0]
        vox_coords = sample["mesh_voxel_coords"]
        colors = sample["mesh_colors"]
        x = FeatureExtractionScannet.to_sparse_tens(vox_coords, colors, device)

        # model forward pass
        y_hat = model(x).detach()

        # associate to original coordinate system
        original_coords = sample["mesh_original_coords"]
        csc_feats = associate_features_to_original_coords(
            y_hat, cfg.ncut.feature_extraction_3d.data, original_coords
        )

        # save as np arrays
        scene_name = sample["scene_name"]
        scan_dir = os.path.join(cfg.ncut.feature_extraction_3d.save_dir, scene_name)
        os.makedirs(scan_dir, exist_ok=True)
        coords_file = os.path.join(scan_dir, "coords.npy")
        feats_file = os.path.join(scan_dir, "csc_feats.npy")
        np.save(coords_file, original_coords)
        print("Saved: ", coords_file)
        np.save(feats_file, csc_feats)
        print("Saved: ", feats_file)

        # in_coords = x.C.cpu().numpy()[:, 1:]
        # visualize_3d_feats(original_coords, colors, save_path="./colors.html")
        # visualize_3d_feats(in_coords, x.F.cpu().numpy(), save_path="./in.html")
        # visualize_3d_feats(original_coords, csc_feats, save_path="./pred.html")


if __name__ == "__main__":
    main()
