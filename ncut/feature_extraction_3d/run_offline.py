import sys
from pathlib import Path
import os
import hydra
import torch
import torch.nn as nn
import open3d as o3d
from scipy.spatial import KDTree
from sklearn.decomposition import PCA
import plotly.graph_objects as go
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


def load_data(cfg_data: DictConfig, device, only_first=False):
    mesh_files = [
        str(i) for i in Path(cfg_data.dataset_dir).rglob(cfg_data.mesh_filename_pattern)
    ]
    if only_first:
        mesh_files = mesh_files[:1]
    for mesh_file in mesh_files:
        # load ply file and convert to minkowski engine sparse tensor
        scene_mesh = o3d.io.read_triangle_mesh(mesh_file)
        coords = np.array(scene_mesh.vertices).astype(np.single)
        original_coords = coords
        coords = coords / cfg_data.voxel_size
        colors = np.array(scene_mesh.vertex_colors).astype(np.single)
        coords = torch.IntTensor(coords).to(device)
        colors = torch.Tensor(colors).to(device)
        coords, colors = ME.utils.sparse_collate(coords=[coords], feats=[colors])
        x = ME.SparseTensor(coordinates=coords, features=colors)

        # get save path
        scan_name = os.path.basename(os.path.dirname(mesh_file))

        yield {
            "sparse_tensor": x,
            "scan_name": scan_name,
            "original_coords": original_coords,
        }


def visualize_feats(coords, feats, save_path=None):
    """
    Visualize features.
    If the feature vector has more than 3 dims, PCA is applied to map it down to 3.
    When using ssh + vscode, specify a save path to a html file,
    go into the hydra save dir and start a web server with:
    python -m http.server 8000
    VSCode will forward the port automatically, so you can view it
    in your local browser.
    """
    # if necessary map features to 3 dims using pca to visualize them as colors
    if feats.shape[1] != 3:
        pca = PCA(n_components=3)
        feats_reduced = pca.fit_transform(feats)
        minv = feats_reduced.min(axis=0)
        maxv = feats_reduced.max(axis=0)
        colors = (feats_reduced - minv) / (maxv - minv)
    else:
        colors = feats

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coords)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    scatter = go.Scatter3d(
        x=coords[:, 0],
        y=coords[:, 1],
        z=coords[:, 2],
        mode="markers",
        marker=dict(size=2, color=colors, opacity=0.8),
    )

    fig = go.Figure(data=[scatter])

    if save_path:
        fig.write_html(save_path)
    else:
        fig.show()

    return fig


def associate_features_to_original_coords(
    y_hat: ME.SparseTensor, cfg_data: DictConfig, original_coords
):
    """
    Transforming the ply file to a minkowski engine sparse tensor is lossy
    because several points map to the same voxel.
    This function assigns the nearest neighbor's features to each of the original points.
    """
    low_res_coords = y_hat.C[:, 1:].cpu().numpy()  # slice off batch dim
    high_res_coords = original_coords / cfg_data.voxel_size
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

    device = torch.device("cuda:0")

    model = get_feature_extractor(cfg.ncut.feature_extraction_3d, device)

    for data in load_data(cfg.ncut.feature_extraction_3d.data, device):
        # model forward pass
        x = data["sparse_tensor"]
        y_hat = model(x).detach()

        # associate to original coordinate system
        original_coords = data["original_coords"]
        csc_feats = associate_features_to_original_coords(
            y_hat, cfg.ncut.feature_extraction_3d.data, original_coords
        )

        # save as np arrays
        scan_name = data["scan_name"]
        scan_dir = os.path.join(cfg.ncut.feature_extraction_3d.save_dir, scan_name)
        os.makedirs(scan_dir, exist_ok=True)
        coords_file = os.path.join(scan_dir, "coords.npy")
        feats_file = os.path.join(scan_dir, "csc_feats.npy")
        np.save(coords_file, original_coords)
        print("Saved: ", coords_file)
        np.save(feats_file, csc_feats)
        print("Saved: ", feats_file)
        # visualize_feats(x.C[:, 1:].cpu().numpy(), x.F.cpu().numpy(), save_path="./in.html")
        # visualize_feats(original_coords, csc_feats, save_path="./pred.html")


if __name__ == "__main__":
    main()
