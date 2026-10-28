import os
import numpy as np
from PIL import Image
from .metrics import tensor_to_numpy_image, calculate_psnr, calculate_ssim

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

def save_comparison_grid(degraded_t, restored_t, clean_t, save_path, title=None):
    """
    Saves a 4-panel visual comparison:
    1. Degraded (Rain + Haze)
    2. Model Restored Output
    3. Ground Truth Clean Road
    4. Difference Map (|Restored - Degraded|)
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    deg_np = tensor_to_numpy_image(degraded_t)
    res_np = tensor_to_numpy_image(restored_t)
    gt_np = tensor_to_numpy_image(clean_t)

    psnr_val = calculate_psnr(res_np, gt_np)
    ssim_val = calculate_ssim(res_np, gt_np)
    deg_psnr = calculate_psnr(deg_np, gt_np)
    deg_ssim = calculate_ssim(deg_np, gt_np)

    diff_map = np.abs(res_np.astype(np.float32) - deg_np.astype(np.float32)).mean(axis=2)
    # Normalize diff to [0, 255]
    diff_norm = np.clip(diff_map * 3.0, 0, 255).astype(np.uint8)
    diff_rgb = np.stack([diff_norm, (diff_norm * 0.4).astype(np.uint8), 255 - diff_norm], axis=2)

    if HAS_MPL:
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        axes[0].imshow(deg_np)
        axes[0].set_title(f"Degraded Input (Rain+Haze)\nPSNR: {deg_psnr:.2f}dB | SSIM: {deg_ssim:.3f}", fontsize=11)
        axes[0].axis("off")

        axes[1].imshow(res_np)
        axes[1].set_title(f"FGAN Restored Road\nPSNR: {psnr_val:.2f}dB | SSIM: {ssim_val:.3f}", fontsize=11, color="green", fontweight="bold")
        axes[1].axis("off")

        axes[2].imshow(gt_np)
        axes[2].set_title("Ground Truth (Target)", fontsize=11)
        axes[2].axis("off")

        axes[3].imshow(diff_rgb)
        axes[3].set_title("Removed Rain/Haze Heatmap", fontsize=11)
        axes[3].axis("off")

        if title:
            plt.suptitle(title, fontsize=14, y=1.02)

        plt.tight_layout()
        plt.savefig(save_path, dpi=180, bbox_inches="tight")
        plt.close()
    else:
        # Concatenate horizontally and save with PIL
        h, w, _ = deg_np.shape
        combined = np.concatenate([deg_np, res_np, gt_np, diff_rgb], axis=1)
        Image.fromarray(combined).save(save_path)

def plot_training_curves(log_csv_path, save_path):
    """
    Plots Generator Loss, Discriminator Loss, and Validation PSNR/SSIM progression.
    """
    if not (HAS_MPL and os.path.exists(log_csv_path)):
        return
    import pandas as pd
    try:
        df = pd.read_csv(log_csv_path)
    except Exception:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    if "gen_loss" in df.columns:
        ax1.plot(df["epoch"], df["gen_loss"], label="Generator Loss", color="royalblue", lw=2)
    if "disc_loss" in df.columns:
        ax1.plot(df["epoch"], df["disc_loss"], label="Discriminator Loss", color="darkorange", lw=2)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training Loss Progression (FGAN)")
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.6)

    if "val_psnr" in df.columns:
        ax2.plot(df["epoch"], df["val_psnr"], label="Val PSNR (dB)", color="forestgreen", lw=2)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("PSNR (dB)")
    ax2.set_title("Validation PSNR Improvement")
    ax2.legend(loc="lower right")
    ax2.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close()
