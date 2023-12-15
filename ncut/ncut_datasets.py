from torch.utils.data import Dataset
from pathlib import Path
import os
import cv2
import open3d as o3d
import numpy as np
import torch
import MinkowskiEngine as ME
from omegaconf import DictConfig
import hydra
import json

from ncut.SensorData_python3_port import SensorData


def color_images_from_sensor_data(sens: SensorData, image_size=None, frame_skip=1):
    """
    Returns a generator object for color images contained
    in the sensor data as torch tensors.
    """
    for f in range(0, len(sens.frames), frame_skip):
        color = sens.frames[f].decompress_color(sens.color_compression_type)
        if image_size is not None:
            color = cv2.resize(
                color, (image_size[1], image_size[0]), interpolation=cv2.INTER_NEAREST
            )
        yield color


def depth_images_from_sensor_data(sens: SensorData, image_size=None, frame_skip=1):
    """
    Returns a generator object for depth images contained
    in the sensor data as torch tensors.
    """
    for f in range(0, len(sens.frames), frame_skip):
        depth_data = sens.frames[f].decompress_depth(sens.depth_compression_type)
        depth = np.fromstring(depth_data, dtype=np.uint16).reshape(
            sens.depth_height, sens.depth_width
        )
        if image_size is not None:
            depth = cv2.resize(
                depth, (image_size[1], image_size[0]), interpolation=cv2.INTER_NEAREST
            )
        yield depth


def poses_from_sensor_data(sens: SensorData, frame_skip=1):
    """
    Returns a generator object for camera poses
    in the sensor data.
    """
    for f in range(0, len(sens.frames), frame_skip):
        yield sens.frames[f].camera_to_world


def color_intrinsics_from_sensor_data(sens: SensorData):
    return sens.intrinsic_color


class FeatureExtractionScannet(Dataset):
    """
    Used for 2D or 3D feature extraction of scannet scenes.
    """

    def __init__(
        self,
        dataset_dir,
        sensordata_glob_pattern,
        mesh_glob_pattern,
        voxel_size,
        scale_colors_to_depth_resolution,
        frame_skip=1,
        content=[
            "mesh_voxel_coords",
            "mesh_original_coords",
            "mesh_colors",
            "color_images",
            "depth_images",
            "camera_poses",
            "color_intrinsics",
        ],
    ):
        """
        ONLY TO BE USED WITH BATCH SIZE 1
        OVERRIDE DATALOADER collate_fn WITH lambda x: x

        dataset_dir: base dir of scannet dataset
        sensordata_filename_pattern: glob pattern for sens file
        mesh_filename_pattern: glob pattern for mesh ply file
        voxel_size: voxel size in meters
        scale_colors_to_depth_resolution: if true, color images are downscaled to depth
            image resolution and intrinsics are adapted accordingly
        frame_skip: every kth frame for color images, depth images, and poses
        content: dict entries of a sample
            scene_name (always contained): scene name as string of format sceneXXXX_XX
            mesh_voxel_coords: np array of shape (n_points, 3) of coords normalized w.r.t. voxel size
            mesh_original_coords: np array of shape (n_points, 3) of coords in meters
            mesh_colors: np array of shape (n_points, 3) of RGB colors of points
            color_images: generator object that loads images np arrays of shape (height, width, channels(3))
            depth_images: generator object that loads depth images as torch tensors of shape (depth_height, depth_width)
            color_intrinsics: intrinsic parameters of color camera - np array of shape (4, 4)
        """
        self.scans = [
            str(os.path.dirname(i))
            for i in Path(dataset_dir).rglob(sensordata_glob_pattern)
        ]
        self.sensordata_glob_pattern = sensordata_glob_pattern
        self.mesh_glob_pattern = mesh_glob_pattern
        self.voxel_size = voxel_size
        self.scale_colors_to_depth_resolution = scale_colors_to_depth_resolution
        self.frame_skip = frame_skip
        self.content = content

    def __len__(self):
        return len(self.scans)

    def __getitem__(self, idx):
        scene_name = os.path.basename(self.scans[idx])
        sample = {"scene_name": scene_name}

        # mesh stuff
        mesh_content = ["mesh_voxel_coords", "mesh_original_coords", "mesh_colors"]
        if any(file in self.content for file in mesh_content):
            mesh_path = str(
                next(iter(Path(self.scans[idx]).glob(self.mesh_glob_pattern)))
            )
            assert os.path.isfile(mesh_path), f"mesh file does not exist: {mesh_path}"
            mesh = o3d.io.read_triangle_mesh(mesh_path)
        if "mesh_voxel_coords" in self.content:
            sample["mesh_voxel_coords"] = (
                np.array(mesh.vertices).astype(np.single) / self.voxel_size
            )
        if "mesh_original_coords" in self.content:
            sample["mesh_original_coords"] = np.array(mesh.vertices).astype(np.single)
        if "mesh_colors" in self.content:
            sample["mesh_colors"] = np.array(mesh.vertex_colors).astype(np.single)

        # image stuff
        image_content = [
            "color_images",
            "depth_imgages",
            "camera_poses",
            "color_instrinsics",
        ]
        if any(item in self.content for item in image_content):
            sens_path = str(
                next(iter(Path(self.scans[idx]).glob(self.sensordata_glob_pattern)))
            )
            assert os.path.isfile(sens_path), f"sens file does not exist: {sens_path}"
            sensordata = SensorData(sens_path)
            if self.scale_colors_to_depth_resolution:
                depth_h, depth_w = next(
                    iter(depth_images_from_sensor_data(sensordata))
                ).shape
                img_h, img_w, _ = next(
                    iter(color_images_from_sensor_data(sensordata))
                ).shape
                scale_x = depth_w / img_w
                scale_y = depth_h / img_h

        if "color_images" in self.content:
            if self.scale_colors_to_depth_resolution:
                sample["color_images"] = color_images_from_sensor_data(
                    sensordata,
                    image_size=(depth_h, depth_w),
                    frame_skip=self.frame_skip,
                )
            else:
                sample["color_images"] = color_images_from_sensor_data(
                    sensordata, frame_skip=self.frame_skip
                )
        if "depth_images" in self.content:
            sample["depth_images"] = depth_images_from_sensor_data(
                sensordata, frame_skip=self.frame_skip
            )
        if "camera_poses" in self.content:
            sample["camera_poses"] = poses_from_sensor_data(
                sensordata, frame_skip=self.frame_skip
            )
        if "color_intrinsics" in self.content:
            intr = color_intrinsics_from_sensor_data(sensordata)
            if self.scale_colors_to_depth_resolution:
                intr[0, 0], intr[1, 1] = intr[0, 0] * scale_x, intr[1, 1] * scale_y
                intr[0, 2], intr[1, 2] = intr[0, 2] * scale_x, intr[1, 2] * scale_y
            sample["color_intrinsics"] = intr

        return sample

    @staticmethod
    def to_sparse_tens(coords, features, device):
        """
        Turns either mesh_voxel_coords + features
        or mesh_original_coords + features
        into a minkowski engine sparse tensor voxel representation.
        Note that this is a lossy conversion because several points might
        map to the same voxel.
        """
        coords = torch.IntTensor(coords).to(device)
        features = torch.Tensor(features).to(device)
        coords, features = ME.utils.sparse_collate(coords=[coords], feats=[features])
        return ME.SparseTensor(coordinates=coords, features=features)

    @staticmethod
    def dataloader_from_hydra(datacfg: DictConfig, only_first=False):
        """
        datacfg must be hydra config of the following form:
        If only_first is True, the dataloader's first batch will be returned.

        dataset:
          <config to instantiate this class>
        dataloader:
          <config to instantiate dataloader with batch size 1 (else will be overwritten!)>
        """
        ds = hydra.utils.instantiate(datacfg.dataset)
        loader = hydra.utils.instantiate(
            datacfg.dataloader, dataset=ds, collate_fn=lambda x: x
        )
        assert loader.batch_size == 1, "batch size must be 1!"
        if only_first:
            return [next(iter(loader))]
        return loader


class NormalizedCutDataset(Dataset):
    """
    Used for creation of initial pseudo masks from
    previously saved point-wise features.
    """

    def __init__(
        self,
        scannet_base_dir,
        mode,
        segments_base_dir,
        base_dir_3d,
        coords_filename_3d,
        features_filename_3d,
        base_dir_2d,
        coords_filename_2d,
        features_filename_2d,
    ):
        self.scannet_base_dir = scannet_base_dir
        self.segments_base_dir = segments_base_dir
        self.base_dir_3d = base_dir_3d
        self.coords_filename_3d = coords_filename_3d
        self.features_filename_3d = features_filename_3d
        self.base_dir_2d = base_dir_2d
        self.coords_filename_2d = coords_filename_2d
        self.features_filename_2d = features_filename_2d

        # read scenes from split file
        assert mode in ["train", "val", "test"], "mode must be train, val, or test"
        split_file = os.path.join(scannet_base_dir, "splits", f"scannetv2_{mode}.txt")
        with open(split_file, "r") as sf:
            self.scenes = [i.strip() for i in sf.readlines()]

    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, idx):
        # load coords and feats from np files
        coords = np.load(
            os.path.join(self.base_dir_3d, self.scenes[idx], self.coords_filename_3d)
        )
        feats_3d = np.load(
            os.path.join(self.base_dir_3d, self.scenes[idx], self.features_filename_3d)
        )
        feats_2d = np.load(
            os.path.join(self.base_dir_2d, self.scenes[idx], self.features_filename_2d)
        )

        # load segments from json and convert to np
        segments_filename = str(
            next(iter(Path(self.segments_base_dir).glob(f"{self.scenes[idx]}*.json")))
        )
        with open(segments_filename, "r") as sf:
            segments_file_dict = json.loads(sf.read())
        segment_ids = np.asarray(segments_file_dict["segIndices"], dtype=int)
        segment_connectivity = np.asarray(
            segments_file_dict["segConnectivity"], dtype=int
        )

        return {
            "coords": coords,  # (n_points, 3), float
            "feats_3d": feats_3d,  # (n_points, dim_feats_3d), float
            "feats_2d": feats_2d,  # (n_points, dim_feats_2d), float
            "segment_ids": segment_ids,  # (n_points,), int
            "segment_connectivity": segment_connectivity,  # (-1, 2), int (neighborhood edges of segments)
        }

    @staticmethod
    def dataloader_from_hydra(datacfg: DictConfig, only_first=False):
        """
        datacfg must be hydra config of the following form:
        If only_first is True, the dataloader's first batch will be returned.

        dataset:
          <config to instantiate this class>
        dataloader:
          <config to instantiate dataloader with batch size 1 (else will be overwritten!)>
        """
        ds = hydra.utils.instantiate(datacfg.dataset)
        loader = hydra.utils.instantiate(datacfg.dataloader, dataset=ds)
        if only_first:
            return [next(iter(loader))]
        return loader
