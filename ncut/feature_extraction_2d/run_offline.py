import sys
import hydra
import torch
import torch.nn as nn
import torchvision.transforms.functional as F
import numpy as np
from sklearn.decomposition import PCA
import clip
from omegaconf import DictConfig
from PIL import Image

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


def visualize_feats(feature_image, save_path, cpm=None):
    """
    Visualize and save torch tensor image (C, H, W).
    If the C > 3, PCA is applied to map it down to 3.
    """
    # if necessary map features to 3 dims using pca to visualize them as colors
    feature_image = feature_image.numpy()
    channels, height, width = feature_image.shape
    if channels > 3: 
        reshaped = feature_image.reshape(channels, height * width).T
        pca = PCA(n_components=3)
        reshaped_reduced = pca.fit_transform(reshaped)
        feats_reduced = reshaped_reduced.T.reshape(3, height, width)
        minv = reshaped_reduced.min()
        maxv = reshaped_reduced.max()
        colors = (feats_reduced - minv) / (maxv - minv) * 255
    else:
        mi, ma = feature_image.min(), feature_image.max()
        colors = (feature_image - mi) / (ma - mi) * 255

    colors = colors.astype(np.uint8)
    if channels == 1:
        img = Image.fromarray(colors.squeeze(), "L")
    else:
        img = Image.fromarray(colors.transpose(1, 2, 0))
    img.save(save_path)


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

    # todo | think about memory management here
    # todo | a sens file is 3.5 gb (with compressed files i guess)
    # todo | a feature tensor should be around 2.6 gb
    for sample in load_data(cfg.ncut.feature_extraction_2d, only_first=True):
        sample = sample[0] # unwrap "batch"
        print(sample)
        for img in list(sample["color_images"])[:1]: # ! tmp
            img = F.to_tensor(img)
            patches, pad_h, pad_w, num_rows, num_cols = f2dutil.split_to_patches(
                img, cfg.ncut.feature_extraction_2d.model.crop_size
            )

            feat_patches = []
            for p in patches:
                p = p.unsqueeze(0)
                p = p.to(device)

                with torch.no_grad():
                    feat_p = model(p)
                    feat_p = nn.functional.normalize(feat_p, dim=1) # unit clip vector per pixel
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

            # tmp: verify clip space by querying with text prompt
            # textencoder = model.clip_pretrained.encode_text
            # prompt = clip.tokenize("table")
            # prompt = prompt.to(device)
            # with torch.no_grad():
            #     text_feat = textencoder(prompt)
            #     text_feat = nn.functional.normalize(text_feat, dim=1)
            # text_feat = text_feat.detach().cpu()
            # cosd = nn.CosineSimilarity(dim=1)
            # sim = cosd(feats, text_feat.unsqueeze(-1).unsqueeze(-1))
            # print(sim.shape)
            # visualize_feats(sim, "sim.png")

            visualize_feats(img, "img.png")
            visualize_feats(feats, "clip_feats.png")

if __name__ == "__main__":
    main()
