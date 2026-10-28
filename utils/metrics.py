import math
import torch
import torch.nn.functional as F
import numpy as np

try:
    from skimage.metrics import peak_signal_noise_ratio as compute_psnr_np
    from skimage.metrics import structural_similarity as compute_ssim_np
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False

def tensor_to_numpy_image(tensor):
    """
    Converts a PyTorch tensor [-1, 1] of shape (C, H, W) or (1, C, H, W)
    to a uint8 RGB numpy array [0, 255] of shape (H, W, C).
    """
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    img = tensor.detach().cpu().clamp(-1.0, 1.0)
    img = (img + 1.0) / 2.0  # [0, 1]
    img = img.permute(1, 2, 0).numpy()
    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    return img

def calculate_psnr(img1_np, img2_np):
    """
    Calculates Peak Signal-to-Noise Ratio (PSNR) in dB.
    """
    if HAS_SKIMAGE:
        return float(compute_psnr_np(img2_np, img1_np, data_range=255))
    mse = np.mean((img1_np.astype(float) - img2_np.astype(float)) ** 2)
    if mse == 0:
        return 100.0
    return float(10.0 * np.log10((255.0 ** 2) / mse))

def calculate_ssim(img1_np, img2_np):
    """
    Calculates Structural Similarity Index Measure (SSIM) in range [0, 1].
    """
    if HAS_SKIMAGE:
        return float(compute_ssim_np(img1_np, img2_np, channel_axis=2, data_range=255))
    
    # Pure NumPy SSIM computation
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    
    img1 = img1_np.astype(float)
    img2 = img2_np.astype(float)
    
    mu1 = img1.mean(axis=(0, 1))
    mu2 = img2.mean(axis=(0, 1))
    
    sigma1_sq = ((img1 - mu1) ** 2).mean(axis=(0, 1))
    sigma2_sq = ((img2 - mu2) ** 2).mean(axis=(0, 1))
    sigma12 = ((img1 - mu1) * (img2 - mu2)).mean(axis=(0, 1))
    
    ssim_channels = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / \
                    ((mu1 ** 2 + mu2 ** 2 + c1) * (sigma1_sq + sigma2_sq + c2))
    return float(np.mean(ssim_channels))

def calculate_mae(img1_np, img2_np):
    """Mean Absolute Error (L1) normalized to [0, 1]."""
    return float(np.mean(np.abs(img1_np.astype(float) - img2_np.astype(float))) / 255.0)
