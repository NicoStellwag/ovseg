import sys
import hydra
import torch
import torch.nn as nn
from omegaconf import DictConfig

import f2dutil


def get_feature_extractor(cfg_fe2d: DictConfig, device) -> nn.Module:
    """
    Instantiates a minkowski resunet and loads pretrained weights.
    """
    model = hydra.utils.instantiate(cfg_fe2d.model, _convert_="all")
    if cfg_fe2d.model.get("weights_path", None):
        if cfg_fe2d.model.weights_type == "state_dict":
            model.load_state_dict(torch.load(cfg_fe2d.model.weights_path))
        elif cfg_fe2d.model.weights_type == "checkpoint":
            model.load_state_dict(torch.load(cfg_fe2d.model.weights_path)["state_dict"])
    return model.eval().to(device)


def load_data(cfg_fe2d: DictConfig, only_first=False):
    ds = hydra.utils.instantiate(cfg_fe2d.data.dataset)
    if (
        cfg_fe2d.data.name == "imagedata"
    ):  # a bit hacky but i don't see a nice solution
        loader = hydra.utils.instantiate(cfg_fe2d.data.dataloader, dataset=ds, collate_fn=lambda x: x)
    else:
        loader = hydra.utils.instantiate(cfg_fe2d.data.dataloader, dataset=ds)
    if only_first:
        return [next(iter(loader))]
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
    sys.path.append(hydra.utils.get_original_cwd())

    device = torch.device("cuda:0")

    model = get_feature_extractor(cfg.ncut.feature_extraction_2d, device)

    # todo | think about memory management here, it won't even fit 2 sets of images
    # todo | + features in ram
    for subscene in load_data(cfg.ncut.feature_extraction_2d, only_first=True):
        subscene = subscene[0] # unwrap "batch"
        for img in subscene["images"][:1]:
            print("img", img.shape)
            patches, pad_h, pad_w, num_rows, num_cols = f2dutil.split_to_patches(
                img, 480
            )

            feat_patches = []
            for p in patches:
                p = p.unsqueeze(0)
                p = p.to(device)

                feat_p = model(p)
                feat_p = nn.functional.interpolate(
                    feat_p, scale_factor=2, mode="nearest"
                )  # model cuts size in half

                feat_p = feat_p.detach().cpu()
                feat_p = feat_p.squeeze(0)

                feat_patches.append(feat_p)

            feats = f2dutil.reconstruct_from_patches(
                feat_patches,
                img.shape[1],
                img.shape[2],
                pad_h,
                pad_w,
                num_rows,
                num_cols,
            )
            print("feats", feats.shape)

if __name__ == "__main__":
    main()
