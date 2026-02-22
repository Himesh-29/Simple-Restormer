import torch
import torch.nn.functional as F
import numpy as np
import cv2

# Lazily initialized LPIPS model
_lpips_model = None

def rgb2ycbcr(img, only_y=True):
    """Convert RGB image to YCbCr.
    The implementation is the same as MATLAB's rgb2ycbcr.
    """
    img = img.astype(np.float32)
    if only_y:
        # Standard RGB coefficients: R: 65.481, G: 128.553, B: 24.966
        rlt = np.dot(img, [65.481, 128.553, 24.966]) / 255.0 + 16.0 / 255.0
    else:
        rlt = np.matmul(img, [[65.481, -37.797, 112], [128.553, -74.203, -93.786], [24.966, 112, -18.214]]) / 255.0 + [16, 128, 128] / 255.0
    return rlt.astype(np.float32)

def calculate_psnr(img1, img2, crop_border=0, test_y_channel=False):
    """Calculate PSNR."""
    if isinstance(img1, torch.Tensor):
        img1 = img1.detach().cpu().numpy().transpose(1, 2, 0)
    if isinstance(img2, torch.Tensor):
        img2 = img2.detach().cpu().numpy().transpose(1, 2, 0)
    
    # Restormer datasets/models usually output in [0, 1] range.
    # Basicsr metrics often expect [0, 255] or handle scaling explicitly.
    # To be consistent with basicsr, we'll work in [0, 255] for PSNR calculation.
    if img1.max() <= 1.0:
        img1 = img1 * 255.0
    if img2.max() <= 1.0:
        img2 = img2 * 255.0

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    if crop_border != 0:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

    if test_y_channel:
        img1 = rgb2ycbcr(img1, only_y=True)
        img2 = rgb2ycbcr(img2, only_y=True)

    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    
    return 20. * np.log10(255.0 / np.sqrt(mse))

def calculate_ssim(img1, img2, crop_border=0, test_y_channel=False):
    """Calculate SSIM."""
    if isinstance(img1, torch.Tensor):
        img1 = img1.detach().cpu().numpy().transpose(1, 2, 0)
    if isinstance(img2, torch.Tensor):
        img2 = img2.detach().cpu().numpy().transpose(1, 2, 0)

    if img1.max() <= 1.0:
        img1 = img1 * 255.0
    if img2.max() <= 1.0:
        img2 = img2 * 255.0

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    if crop_border != 0:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

    if test_y_channel:
        img1 = rgb2ycbcr(img1, only_y=True)
        img2 = rgb2ycbcr(img2, only_y=True)

    try:
        from skimage.metrics import structural_similarity as ssim
        return ssim(img1, img2, data_range=255.0, multichannel=(not test_y_channel))
    except ImportError:
        return 0.0

def calculate_lpips(img1_tensor, img2_tensor, device=None):
    """
    Calculate LPIPS perceptual distance. Lower is better.
    Expects input tensors shaped (B, C, H, W) in range [0, 1].
    """
    global _lpips_model
    try:
        import lpips
    except ImportError:
        return 0.0
        
    if _lpips_model is None:
        if device is None:
            device = img1_tensor.device
        _lpips_model = lpips.LPIPS(net='alex').to(device)
        _lpips_model.eval()

    # LPIPS expects images in range [-1, 1]
    # Restormer generally outputs [0, 1]
    if img1_tensor.max() <= 1.0:
        img1_val = img1_tensor * 2.0 - 1.0
    else:
        img1_val = (img1_tensor / 255.0) * 2.0 - 1.0
        
    if img2_tensor.max() <= 1.0:
        img2_val = img2_tensor * 2.0 - 1.0
    else:
        img2_val = (img2_tensor / 255.0) * 2.0 - 1.0
        
    with torch.no_grad():
        dist = _lpips_model(img1_val, img2_val)
        
    return dist.mean().item()
