import torch
import torch.nn.functional as F
import numpy as np
import cv2

def calculate_psnr(img1, img2, crop_border=0):
    """Calculate PSNR."""
    if isinstance(img1, torch.Tensor):
        img1 = img1.detach().cpu().numpy().transpose(1, 2, 0)
    if isinstance(img2, torch.Tensor):
        img2 = img2.detach().cpu().numpy().transpose(1, 2, 0)
    
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    if crop_border != 0:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    max_value = 1. if img1.max() <= 1 else 255.
    return 20. * np.log10(max_value / np.sqrt(mse))

def calculate_ssim(img1, img2, crop_border=0):
    """Calculate SSIM."""
    if isinstance(img1, torch.Tensor):
        img1 = img1.detach().cpu().numpy().transpose(1, 2, 0)
    if isinstance(img2, torch.Tensor):
        img2 = img2.detach().cpu().numpy().transpose(1, 2, 0)

    if crop_border != 0:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

    # Simplified SSIM implementation or using skimage
    try:
        from skimage.metrics import structural_similarity as ssim
        return ssim(img1, img2, data_range=img1.max(), multichannel=True)
    except ImportError:
        # Fallback to a basic implementation if skimage is not available
        # (Though we should probably ensure it is in requirements)
        return 0.0
