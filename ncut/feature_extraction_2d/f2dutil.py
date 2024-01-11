import torch
import numpy as np


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
