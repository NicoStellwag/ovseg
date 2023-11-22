import sys
from pathlib import Path
import hydra
import torch
import torch.nn as nn
import torchvision.transforms.functional as F
import open3d as o3d
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


def load_data(cfg_data: DictConfig, device):
    mesh_files = [
        str(i) for i in Path(cfg_data.dataset_dir).rglob(cfg_data.mesh_filename_pattern)
    ]
    for mesh_file in mesh_files[:1]:
        scene_mesh = o3d.io.read_triangle_mesh(mesh_file)
        coords = np.array(scene_mesh.vertices).astype(np.single)
        coords = coords / cfg_data.voxel_size
        colors = np.array(scene_mesh.vertex_colors).astype(np.single)
        coords = torch.IntTensor(coords).to(device)
        colors = torch.Tensor(colors).to(device)
        coords, colors = ME.utils.sparse_collate(coords=[coords], feats=[colors])
        x = ME.SparseTensor(coordinates=coords, features=colors)
        yield x


def visualize_sparse_tensor(tens, save_path=None):
    """
    Visualize a ME sparse tensor.
    When using ssh + vscode, specify a save path to a html file,
    go into the hydra save dir and start a web server with:
    python -m http.server 8000
    VSCode will forward the port automatically, so you can view it
    in your local browser.
    """
    coords = tens.C[:, 1:].cpu().numpy() # remove batch dim

    # if necessary map features to 3 dims using pca to visualize them as colors
    feats = tens.F.cpu().numpy()
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
        mode='markers',
        marker=dict(
            size=2,
            color=colors,
            opacity=0.8
        )
    )

    fig = go.Figure(data=[scatter])

    if save_path:
        fig.write_html(save_path)
    else:
        fig.show()

    return fig


@hydra.main(
    config_path="../../conf", config_name="config_base_instance_segmentation.yaml"
)
def main(cfg: DictConfig):
    # necessary for hydra object instantiation when script is not in project root
    sys.path.append(hydra.utils.get_original_cwd())

    device = torch.device("cuda:0")

    model = get_feature_extractor(cfg.ncut.feature_extraction_3d, device)

    for x in load_data(cfg.ncut.feature_extraction_3d.data, device):
        y_hat = model(x).detach()
        print("input sparse", x.F.shape)
        print("output sparse", y_hat.F.shape)
        visualize_sparse_tensor(x, save_path="./in.html")
        visualize_sparse_tensor(y_hat, save_path="./pred.html")

if __name__ == "__main__":
    main()
