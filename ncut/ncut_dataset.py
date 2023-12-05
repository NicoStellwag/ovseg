from torch.utils.data import Dataset
from pathlib import Path
import os
import cv2
import open3d as o3d
import numpy as np
import torch
import MinkowskiEngine as ME

from SensorData_python3_port import SensorData


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


class NcutScannetDataset(Dataset):
    """
    Dataset helper for easy parallel loading.
    """

    def __init__(
        self,
        dataset_dir,
        sensordata_glob_pattern,
        mesh_glob_pattern,
        voxel_size,
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
        voxel_size: voxel size in cm
        content: dict entries of a sample
            mesh_voxel_coords: np array of shape (n_points, 3) of coords normalized w.r.t. voxel size
            mesh_original_coords: np array of shape (n_points, 3) of coords in meters
            mesh_colors: np array of shape (n_points, 3) of RGB colors of points
            color_images: generator object that loads images np arrays of shape (height, width, channels/3)
            depth_images: generator object that loads depth images as torch tensors of shape (depth_height, depth_width)
            color_intrinsics: intinsic paramters of color camera - np array of shape (4, 4)
        """
        self.scans = [
            str(os.path.dirname(i))
            for i in Path(dataset_dir).rglob(sensordata_glob_pattern)
        ]
        self.sensordata_glob_pattern = sensordata_glob_pattern
        self.mesh_glob_pattern = mesh_glob_pattern
        self.voxel_size = voxel_size
        self.content = content

    def __len__(self):
        return len(self.scans)

    def __getitem__(self, idx):
        scan_name = os.path.basename(self.scans[idx])
        sample = {"scan_name": scan_name}

        # mesh stuff
        mesh_content = ["mesh_voxel_coords", "mesh_original_coords", "mesh_colors"]
        if any(file in self.content for file in mesh_content):
            mesh_path = next(iter(Path(self.scans[idx]).glob(self.mesh_glob_pattern)))
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
            sens_path = next(
                iter(Path(self.scans[idx]).glob(self.sensordata_glob_pattern))
            )
            assert os.path.isfile(sens_path), f"sens file does not exist: {sens_path}"
            sensordata = SensorData(sens_path)
        if "color_images" in self.content:
            sample["color_images"] = color_images_from_sensor_data(sensordata)
        if "depth_images" in self.content:
            sample["depth_images"] = depth_images_from_sensor_data(sensordata)
        if "camera_poses" in self.content:
            sample["camera_poses"] = poses_from_sensor_data(sensordata)
        if "color_intrinsics" in self.content:
            sample["color_intrinsics"] = color_intrinsics_from_sensor_data(sensordata)

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
        features, coords = ME.utils.sparse_collate(coords=[coords], feats=[features])
        return ME.SparseTensor(coordinates=coords, features=features)
