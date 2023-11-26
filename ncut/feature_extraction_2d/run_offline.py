import sys
import hydra
import os
import torch
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor, AutoModel
from scipy.spatial import KDTree
from sklearn.decomposition import PCA
import numpy as np
from omegaconf import DictConfig

import f2dutil





def get_feature_extractor(cfg_fe2d: DictConfig, device) -> nn.Module:
    """
    Instantiates a minkowski resunet and loads pretrained weights.
    """
    if cfg_fe2d.model.type == "huggingface_transformers":
        processor = AutoImageProcessor.from_pretrained(cfg_fe2d.model.processor_name)
        model = AutoModel.from_pretrained(cfg_fe2d.model.model_name)
    return processor, model.eval().to(device)


def load_data(cfg_data: DictConfig, only_first=False):
    ds = f2dutil.SenseDS(cfg_data)
    loader = DataLoader(ds, batch_size=1, collate_fn=lambda x: x)#, num_workers=8, persistent_workers=True)
    if only_first:
        return next(iter(loader))
    return loader

# def visualize_feats(coords, feats, save_path=None):
#     """
#     Visualize features.
#     If the feature vector has more than 3 dims, PCA is applied to map it down to 3.
#     When using ssh + vscode, specify a save path to a html file,
#     go into the hydra save dir and start a web server with:
#     python -m http.server 8000
#     VSCode will forward the port automatically, so you can view it
#     in your local browser.
#     """
#     # if necessary map features to 3 dims using pca to visualize them as colors
#     if feats.shape[1] != 3:
#         pca = PCA(n_components=3)
#         feats_reduced = pca.fit_transform(feats)
#         minv = feats_reduced.min(axis=0)
#         maxv = feats_reduced.max(axis=0)
#         colors = (feats_reduced - minv) / (maxv - minv)
#     else:
#         colors = feats

#     if save_path:
#         fig.write_html(save_path)
#     else:
#         fig.show()

#     return fig


# def associate_features_to_original_coords(
#     y_hat: ME.SparseTensor, cfg_data: DictConfig, original_coords
# ):
#     """
#     Transforming the ply file to a minkowski engine sparse tensor is lossy
#     because several points map to the same voxel.
#     This function assigns the nearest neighbor's features to each of the original points.
#     """
#     low_res_coords = y_hat.C[:, 1:].cpu().numpy()  # slice off batch dim
#     high_res_coords = original_coords / cfg_data.voxel_size
#     kdtree = KDTree(low_res_coords)
#     _, nn_indices = kdtree.query(high_res_coords, k=1)
#     high_res_feats = y_hat.F[nn_indices].cpu().numpy()
#     return high_res_feats


@hydra.main(
    config_path="../../conf", config_name="config_base_instance_segmentation.yaml"
)
def main(cfg: DictConfig):
    device = torch.device("cuda:0")

    processor, model = get_feature_extractor(cfg.ncut.feature_extraction_2d, device)

    for subscene in load_data(cfg.ncut.feature_extraction_2d.data, only_first=True):
        # model forward pass
        for img in subscene["images"][:1]:
            print(img.shape)

            patches, pad_height, pad_width = f2dutil.split_to_patches(img, 224)
            print(patches.shape)

            recon = f2dutil.reconstruct_from_patches(patches, img.shape, pad_height, pad_width)
            print(recon.shape)

            # inputs = processor(images=img, return_tensors="pt")
            # inputs = inputs.to(device)
            # outputs = model(**inputs)
            # feats = outputs[0].detach().cpu().numpy() # last hidden states
            # print(img.shape)
            # print(feats.shape)
        # print(y_hat.shape)


        # # save as np arrays
        # scan_name = data["scan_name"]
        # scan_dir = os.path.join(cfg.ncut.feature_extraction_3d.save_dir, scan_name)
        # os.makedirs(scan_dir, exist_ok=True)
        # coords_file = os.path.join(scan_dir, "coords.npy")
        # feats_file = os.path.join(scan_dir, "csc_feats.npy")
        # np.save(coords_file, original_coords)
        # print("Saved: ", coords_file)
        # np.save(feats_file, csc_feats)
        # print("Saved: ", feats_file)
        # # visualize_feats(x.C[:, 1:].cpu().numpy(), x.F.cpu().numpy(), save_path="./in.html")
        # # visualize_feats(original_coords, csc_feats, save_path="./pred.html")


if __name__ == "__main__":
    main()
