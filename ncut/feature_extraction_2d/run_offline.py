import sys
import hydra
from sklearn.neighbors import KDTree
import torch
import torch.nn as nn
import torchvision.transforms.functional as F
import numpy as np
from sklearn.decomposition import PCA
from omegaconf import DictConfig
from PIL import Image
import logging
import os

import f2dutil


log = logging.getLogger(__name__)


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


def associate_voxelized_features_to_high_res_coords(
    low_res_coords, low_res_feats, high_res_coords
):
    """
    Transforming the ply file to a minkowski engine sparse tensor is lossy
    because several points map to the same voxel.
    This function assigns the nearest neighbor's features to each of the original points.
    """
    kdtree = KDTree(low_res_coords)
    _, nn_indices = kdtree.query(high_res_coords, k=1)
    high_res_feats = low_res_feats[nn_indices].squeeze(1)
    return high_res_feats


def patch_wise_inference(cfg, device, model, img):
    patches, pad_h, pad_w, num_rows, num_cols = f2dutil.split_to_patches(
        img, cfg.ncut.feature_extraction_2d.model.crop_size
    )
    feat_patches = []
    for p in patches:
        p = p.unsqueeze(0)
        p = p.to(device)
        with torch.no_grad():
            feat_p = model(p)
            feat_p = nn.functional.normalize(
                feat_p, dim=1
            )  # unit clip vector per pixel
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
    return feats


@hydra.main(
    config_path="../../conf", config_name="config_base_instance_segmentation.yaml"
)
def main(cfg: DictConfig):
    sys.path.append(hydra.utils.get_original_cwd())
    from utils.cuda_utils.raycast_image import Project2DFeaturesCUDA
    from ncut.ncut_dataset import NcutScannetDataset
    from ncut.visualize import visualize_3d_feats

    device = torch.device("cuda:0")

    model = get_feature_extractor(cfg.ncut.feature_extraction_2d, device)
    loader = NcutScannetDataset.dataloader_from_hydra(
        cfg.ncut.feature_extraction_2d.data, only_first=True
    )

    for sample in loader:
        sample = sample[0]  # unwrap "batch"

        log.info(f"** {sample['scene_name']}")

        color_images = list(sample["color_images"])
        camera_poses = list(
            sample["camera_poses"]
        )  # note that these already are inverse extrinsics!

        # the feature extractor might return lower res features than the input image
        model_scale_fac = cfg.ncut.feature_extraction_2d.get("model_scale_fac", 1.0)

        # initialize projector
        img_width = color_images[0].shape[1]  # np images at this point
        img_height = color_images[0].shape[0]
        voxel_size = cfg.ncut.feature_extraction_2d.data.dataset.voxel_size
        projector = Project2DFeaturesCUDA(
            width=img_width * model_scale_fac,
            height=img_height * model_scale_fac,
            voxel_size=voxel_size,
            config={},
        )

        # initialize other stuff for the projection
        voxel_coords = sample["mesh_voxel_coords"]
        n_points_full = voxel_coords.shape[0]
        mock_feats = np.zeros(shape=(n_points_full, 1))
        sparse_tens = NcutScannetDataset.to_sparse_tens(
            voxel_coords, mock_feats, device
        )  # just used to transform coords for projector
        batched_sparse_voxel_coords = sparse_tens.C
        color_intrinsics = sample["color_intrinsics"]
        fx, fy, mx, my = (
            color_intrinsics[0, 0],
            color_intrinsics[1, 1],
            color_intrinsics[0, 2],
            color_intrinsics[1, 2],
        )
        color_intrinsics = torch.Tensor([fx, fy, mx, my])
        # apparently intrinsics are already normalized w.r.t. voxel size
        # but if the feature extractor scales down the image during inference we adapt
        # the intrinsics accordingly instead of upscaling again
        color_intrinsics = color_intrinsics * model_scale_fac
        color_intrinsics = color_intrinsics.unsqueeze(0).to(device)

        # initialize empty 3D features
        n_points_voxelized = batched_sparse_voxel_coords.shape[0]
        feature_dim = cfg.ncut.feature_extraction_2d.feature_dim
        projected_features = torch.zeros(size=(n_points_voxelized, feature_dim))
        total_hits = torch.zeros(size=(n_points_voxelized, 1))

        for i, (img, pose) in enumerate(zip(color_images, camera_poses)):
            # feature extraction
            img = F.to_tensor(img)
            img = img.unsqueeze(0).to(device)
            with torch.no_grad():
                feats = model(img)
            feats = nn.functional.normalize(feats, dim=1)
            feats = feats.detach().cpu().squeeze(0)

            # project to 3D and add to aggregated 3D features
            feats = feats.to(device).permute(1, 2, 0).unsqueeze(0).unsqueeze(0)
            pose[:3, 3] = pose[:3, 3] / voxel_size  # adapt shift to voxel coord system
            pose = torch.from_numpy(pose).to(device).unsqueeze(0).unsqueeze(0)
            curr_projected_features, hit_counts = projector(
                encoded_2d_features=feats,  # (batch_num, view_num, height, width, channels)
                coords=batched_sparse_voxel_coords,  # ME sparse tens coords
                view_matrix=pose,  # (batch_num, view_num, 4, 4)
                intrinsic_params=color_intrinsics,  # (batch_num, 4) [fx, fy, mx, my]
            )
            hit_counts = hit_counts.cpu()
            curr_projected_features = curr_projected_features.cpu()
            hit_points = hit_counts > 0
            projected_features[hit_points] += curr_projected_features[hit_points]
            total_hits[hit_points] += 1  # projector already takes mean

            log.info(f"\timage {i}")

        # take mean feature for every 3D point
        projected_features = projected_features / (total_hits + 1e-8)

        # associate back to original coord system
        original_coords = sample["mesh_original_coords"]
        sparse_voxel_coords = batched_sparse_voxel_coords[:, 1:].cpu().numpy()
        high_res_feats = associate_voxelized_features_to_high_res_coords(
            low_res_coords=sparse_voxel_coords,
            low_res_feats=projected_features,
            high_res_coords=voxel_coords,  # same order as original coords
        )

        # save as np arrays
        scene_name = sample["scene_name"]
        scan_dir = os.path.join(cfg.ncut.feature_extraction_2d.save_dir, scene_name)
        os.makedirs(scan_dir, exist_ok=True)
        coords_file = os.path.join(
            scan_dir, cfg.ncut.feature_extraction_2d.coords_filename
        )
        feats_file = os.path.join(
            scan_dir, cfg.ncut.feature_extraction_2d.feats_filename
        )
        np.save(coords_file, original_coords)
        log.info(f"Saved: {coords_file}")
        np.save(feats_file, high_res_feats)
        log.info(f"Saved: {feats_file}")

        # # visualize
        # visualize_3d_feats(
        #     original_coords,
        #     high_res_feats,
        #     "./high_res_feats3d.html"
        # )
        # visualize_3d_feats(
        #     sparse_voxel_coords,
        #     projected_features,
        #     "./feats3d.html",
        # )


if __name__ == "__main__":
    main()
