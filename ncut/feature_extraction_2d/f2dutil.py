import cv2
import torchvision.transforms.functional as F
from torch.utils.data import Dataset
import torch
import numpy as np
from pathlib import Path
import os

from SensorData_python3_port import SensorData


class SensDS(Dataset):
    """
    Dataset helper for easy parallel loading.
    """

    def __init__(self, dataset_dir, sensordata_filename_pattern):
        self.sens_files = [
            str(i) for i in Path(dataset_dir).rglob(sensordata_filename_pattern)
        ]

    def __len__(self):
        return len(self.sens_files)

    def __getitem__(self, idx):
        images = list(color_images_from_sens_file(self.sens_files[idx]))
        scan_name = os.path.basename(os.path.dirname(self.sens_files[idx]))
        return {"images": images, "scan_name": scan_name}


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
        yield F.to_tensor(color)


def color_images_from_sens_file(path, image_size=None, frame_skip=1):
    """
    Returns a generator object for color images contained
    in a sens file as torch tensors.
    """
    data = SensorData(path)
    yield from color_images_from_sensor_data(data, image_size, frame_skip)


def split_to_patches(img, patch_size):
    """
    Split torch image tensor of shape (C, H, W)
    to a list of square patches [(C, patch_size, patch_size), ...].
    Pad if necessary.
    Returns: patches, pad_height, pad_width, num_rows, num_cols
    """
    img = img.numpy()
    pad_h = (patch_size - (img.shape[1] % patch_size)) % patch_size
    pad_w = (patch_size - (img.shape[2] % patch_size)) % patch_size
    padded_img = np.pad(img, ((0, 0), (0, pad_h), (0, pad_w)))

    num_rows = padded_img.shape[1] // patch_size
    num_cols = padded_img.shape[2] // patch_size

    patches = []
    for row in range(num_rows):
        for col in range(num_cols):
            h_from = row * patch_size
            h_to = (row + 1) * patch_size
            w_from = col * patch_size
            w_to = (col + 1) * patch_size
            patches.append(padded_img[:, h_from:h_to, w_from:w_to])

    tensor_patches = [torch.from_numpy(i) for i in patches]
    return tensor_patches, pad_h, pad_w, num_rows, num_cols


def reconstruct_from_patches(
    patches, original_h, original_w, pad_h, pad_w, num_rows, num_cols
):
    """
    Reconstruct torch image tensor with original dimensions
    from a batch of patches and remove padding.
    """
    feature_channels, patch_size, _ = patches[0].shape
    reconstructed = torch.empty(
        (feature_channels, original_h + pad_h, original_w + pad_w)
    )
    for row in range(num_rows):
        for col in range(num_cols):
            h_from = row * patch_size
            h_to = (row + 1) * patch_size
            w_from = col * patch_size
            w_to = (col + 1) * patch_size
            reconstructed[:, h_from:h_to, w_from:w_to] = patches[row * num_cols + col]
    original_reconstructed = reconstructed[:, :original_h, :original_w]
    return original_reconstructed
