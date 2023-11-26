import cv2
import torchvision.transforms.functional as F
from torch.utils.data import Dataset
import torch.nn.functional as nnF
import torch
from pathlib import Path
import os

from SensorData_python3_port import SensorData


class SenseDS(Dataset):
    """
    Dataset helper for easy parallel loading.
    """

    def __init__(self, cfg_data):
        self.sens_files = [
            str(i)
            for i in Path(cfg_data.dataset_dir).rglob(
                cfg_data.sensordata_filename_pattern
            )
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


def split_to_patches(image_tensor, patch_size):
    """
    Split torch image tensor of shape (C, H, W)
    to a batch of square patches (patches, C, patch_size, patch_size).
    Pad if necessary.
    Returns (patches, pad_height, pad_width) - padding dims needed for reconstruction.
    """
    C, H, W = image_tensor.size()
    pad_height = (patch_size - H % patch_size) % patch_size
    pad_width = (patch_size - W % patch_size) % patch_size

    # Padding the image if necessary
    image_tensor = F.pad(image_tensor, (0, pad_width, 0, pad_height), padding_mode='constant', fill=0)

    # Using unfold to split the image into patches
    patches = image_tensor.unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size)
    patches = patches.contiguous().view(-1, C, patch_size, patch_size)
    return patches, pad_height, pad_width



# todo fix this
def reconstruct_from_patches(patches, original_size, pad_height, pad_width):
    """
    Reconstruct torch image tensor with original dimensions
    from a batch of patches and remove padding.
    """
    _, H, W = original_size
    n_patches, channels, patch_size, _ = patches.size()

    # Calculate the total number of patch positions
    num_patches_h = (H + pad_height) // patch_size
    num_patches_w = (W + pad_width) // patch_size
    total_patches = num_patches_h * num_patches_w

    # Ensure that the number of patches matches the expected number
    assert n_patches == total_patches, "Number of patches does not match expected total."

    # Reshape patches for the fold operation
    patches = patches.permute(0, 2, 3, 1).reshape(1, channels * patch_size * patch_size, -1)

    # Reconstruct the image using fold
    output = nnF.fold(patches, output_size=(H + pad_height, W + pad_width), kernel_size=patch_size, stride=patch_size)

    # Remove the padding
    if pad_height > 0 or pad_width > 0:
        output = output[:, :, :H, :W]

    return output
