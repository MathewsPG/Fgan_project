import os
import sys
import argparse
import time
import csv
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.generator import DeRainDeHazeGenerator
from models.discriminator import PatchGANDiscriminator
from models.losses import VGGPerceptualLoss, SSIMLoss, EdgeLoss
from data.dataset import PairedRoadDataset
from data.download_dataset import create_sample_road_dataset
from utils.metrics import calculate_psnr, calculate_ssim, calculate_mae, tensor_to_numpy_image
from utils.visualizer import save_comparison_grid, plot_training_curves

def train_fgan(args):
    # Set device
    if torch.cuda.is_available() and not args.force_cpu:
        device = torch.device("cuda")
        print(f"[+] CUDA Activated: Using {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device("cpu")
        print("[!] Using CPU for training.")

    # Prepare directories
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)
    vis_dir = os.path.join(args.results_dir, "visuals")
    os.makedirs(vis_dir, exist_ok=True)

    # Ensure dataset exists
    train_dir = os.path.join(args.dataset_dir, "train")
    test_dir = os.path.join(args.dataset_dir, "test")
    if not (os.path.exists(train_dir) and len(os.listdir(os.path.join(train_dir, "clean"))) > 0):
        print("[*] Dataset not found. Auto-generating paired road dataset...")
        create_sample_road_dataset(target_dir=args.dataset_dir, num_samples=args.num_samples)

    train_dataset = PairedRoadDataset(train_dir, image_size=args.img_size, is_train=True)
    val_dataset = PairedRoadDataset(test_dir, image_size=args.img_size, is_train=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True if device.type == "cuda" else False,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0
    )

    print(f"[+] Dataset Loaded: {len(train_dataset)} training pairs, {len(val_dataset)} validation pairs.")

    # Initialize Models
    generator = DeRainDeHazeGenerator(in_channels=3, out_channels=3, features=args.gen_features).to(device)
    discriminator = PatchGANDiscriminator(in_channels=6, features=(64, 128, 256, 512)).to(device)

    # Loss Functions
    criterion_gan = nn.BCEWithLogitsLoss()
    criterion_l1 = nn.L1Loss()
    criterion_ssim = SSIMLoss().to(device)
    criterion_edge = EdgeLoss().to(device)
    
    use_vgg = not args.no_vgg
    if use_vgg:
        try:
            criterion_vgg = VGGPerceptualLoss().to(device)
            print("[+] VGG-19 Perceptual Loss Loaded.")
        except Exception as e:
            print(f"[!] Warning: VGG perceptual loss unavailable ({e}). Using L1 + SSIM + Edge loss.")
            use_vgg = False

    # Optimizers
    opt_g = torch.optim.Adam(generator.parameters(), lr=args.lr_g, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=args.lr_d, betas=(0.5, 0.999))

    # Schedulers
    scheduler_g = torch.optim.lr_scheduler.CosineAnnealingLR(opt_g, T_max=args.epochs, eta_min=1e-6)
    scheduler_d = torch.optim.lr_scheduler.CosineAnnealingLR(opt_d, T_max=args.epochs, eta_min=1e-6)

    # Mixed Precision Scaler for GPU speedup
    use_amp = (device.type == "cuda")
    scaler_g = torch.amp.GradScaler('cuda', enabled=use_amp)
    scaler_d = torch.amp.GradScaler('cuda', enabled=use_amp)

    # Training state
    best_psnr = -1.0
    log_file = os.path.join(args.results_dir, "training_log.csv")
    csv_header = ["epoch", "gen_loss", "disc_loss", "val_psnr", "val_ssim", "val_mae", "lr_g"]
    with open(log_file, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(csv_header)

    print(f"\n{'='*70}\nStarting FGAN Road Rain & Haze Removal Training ({args.epochs} Epochs)\n{'='*70}")
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        generator.train()
        discriminator.train()
        
        running_g_loss = 0.0
        running_d_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch [{epoch:02d}/{args.epochs:02d}]", unit="batch")

        for step, batch in enumerate(pbar):
            degraded = batch["degraded"].to(device, non_blocking=True)
            clean = batch["clean"].to(device, non_blocking=True)

            # ---------------------
            # Train Discriminator
            # ---------------------
            opt_d.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type='cuda', enabled=use_amp):
                # Real pair: (degraded, clean)
                d_real = discriminator(degraded, clean)
                loss_d_real = criterion_gan(d_real, torch.ones_like(d_real))

                # Fake pair: (degraded, restored)
                with torch.no_grad():
                    restored_fake = generator(degraded)
                d_fake = discriminator(degraded, restored_fake.detach())
                loss_d_fake = criterion_gan(d_fake, torch.zeros_like(d_fake))

                loss_d = 0.5 * (loss_d_real + loss_d_fake)

            scaler_d.scale(loss_d).backward()
            scaler_d.step(opt_d)
            scaler_d.update()

            # ---------------------
            # Train Generator
            # ---------------------
            opt_g.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type='cuda', enabled=use_amp):
                restored = generator(degraded)
                d_fake_for_g = discriminator(degraded, restored)
                loss_g_adv = criterion_gan(d_fake_for_g, torch.ones_like(d_fake_for_g))

                loss_l1 = criterion_l1(restored, clean)
                loss_ssim = criterion_ssim(restored, clean)
                loss_edge = criterion_edge(restored, clean)

                loss_vgg = criterion_vgg(restored, clean) if use_vgg else 0.0

                loss_g = (
                    args.lambda_adv * loss_g_adv +
                    args.lambda_l1 * loss_l1 +
                    args.lambda_ssim * loss_ssim +
                    args.lambda_edge * loss_edge +
                    (args.lambda_vgg * loss_vgg if use_vgg else 0.0)
                )

            scaler_g.scale(loss_g).backward()
            scaler_g.step(opt_g)
            scaler_g.update()

            running_g_loss += loss_g.item()
            running_d_loss += loss_d.item()
            pbar.set_postfix({
                "G_Loss": f"{loss_g.item():.4f}",
                "D_Loss": f"{loss_d.item():.4f}",
                "L1": f"{loss_l1.item():.4f}"
            })

        scheduler_g.step()
        scheduler_d.step()

        epoch_g_loss = running_g_loss / len(train_loader)
        epoch_d_loss = running_d_loss / len(train_loader)

        # ---------------------
        # Validation Evaluation
        # ---------------------
        generator.eval()
        val_psnr_list, val_ssim_list, val_mae_list = [], [], []

        with torch.no_grad():
            for v_idx, v_batch in enumerate(val_loader):
                v_deg = v_batch["degraded"].to(device)
                v_clean = v_batch["clean"].to(device)
                with torch.amp.autocast(device_type='cuda', enabled=use_amp):
                    v_res = generator(v_deg)

                deg_np = tensor_to_numpy_image(v_deg[0])
                res_np = tensor_to_numpy_image(v_res[0])
                gt_np = tensor_to_numpy_image(v_clean[0])

                val_psnr_list.append(calculate_psnr(res_np, gt_np))
                val_ssim_list.append(calculate_ssim(res_np, gt_np))
                val_mae_list.append(calculate_mae(res_np, gt_np))

                if v_idx == 0:
                    vis_path = os.path.join(vis_dir, f"epoch_{epoch:03d}_val.png")
                    save_comparison_grid(
                        v_deg[0], v_res[0], v_clean[0],
                        vis_path,
                        title=f"FGAN Epoch {epoch} Road Restoration"
                    )

        mean_psnr = float(sum(val_psnr_list) / len(val_psnr_list))
        mean_ssim = float(sum(val_ssim_list) / len(val_ssim_list))
        mean_mae = float(sum(val_mae_list) / len(val_mae_list))

        print(f"[*] Epoch {epoch:02d} Summary -> G Loss: {epoch_g_loss:.4f} | D Loss: {epoch_d_loss:.4f} | "
              f"Val PSNR: {mean_psnr:.2f} dB | Val SSIM: {mean_ssim:.4f} | Val MAE: {mean_mae:.4f}")

        # Append to CSV
        with open(log_file, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                round(epoch_g_loss, 4),
                round(epoch_d_loss, 4),
                round(mean_psnr, 2),
                round(mean_ssim, 4),
                round(mean_mae, 4),
                scheduler_g.get_last_lr()[0]
            ])

        # Save latest checkpoint
        torch.save({
            "epoch": epoch,
            "generator": generator.state_dict(),
            "discriminator": discriminator.state_dict(),
            "opt_g": opt_g.state_dict(),
            "opt_d": opt_d.state_dict(),
            "best_psnr": best_psnr
        }, os.path.join(args.checkpoint_dir, "latest_checkpoint.pth"))

        # Save Best Model Checkpoint
        if mean_psnr > best_psnr:
            best_psnr = mean_psnr
            torch.save(generator.state_dict(), os.path.join(args.checkpoint_dir, "best_generator.pth"))
            torch.save(discriminator.state_dict(), os.path.join(args.checkpoint_dir, "best_discriminator.pth"))
            print(f" [*] New Best Model Saved with PSNR: {best_psnr:.2f} dB")

    # Plot training metrics curve
    plot_path = os.path.join(args.results_dir, "training_curves.png")
    plot_training_curves(log_file, plot_path)

    elapsed = time.time() - start_time
    print(f"\n{'='*70}\n[+] Training Complete in {elapsed/60:.2f} minutes.")
    print(f"[+] Best Validation PSNR: {best_psnr:.2f} dB")
    print(f"[+] Model Checkpoints: {args.checkpoint_dir}")
    print(f"[+] Visuals & Curves:  {args.results_dir}\n{'='*70}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Feature-Attention GAN for Road Rain & Haze Removal")
    parser.add_argument("--dataset_dir", type=str, default="dataset", help="Dataset directory")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Save directory for model weights")
    parser.add_argument("--results_dir", type=str, default="results", help="Directory for logs and visuals")
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size (default 4 fits 4GB RTX 3050)")
    parser.add_argument("--img_size", type=int, default=256, help="Image resolution for training")
    parser.add_argument("--lr_g", type=float, default=2e-4, help="Generator learning rate")
    parser.add_argument("--lr_d", type=float, default=1e-4, help="Discriminator learning rate")
    parser.add_argument("--gen_features", type=int, default=64, help="Base feature dimension")
    parser.add_argument("--lambda_l1", type=float, default=100.0, help="L1 loss weight")
    parser.add_argument("--lambda_adv", type=float, default=1.0, help="Adversarial loss weight")
    parser.add_argument("--lambda_ssim", type=float, default=10.0, help="SSIM loss weight")
    parser.add_argument("--lambda_edge", type=float, default=5.0, help="Sobel edge loss weight")
    parser.add_argument("--lambda_vgg", type=float, default=10.0, help="VGG perceptual loss weight")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader workers")
    parser.add_argument("--num_samples", type=int, default=120, help="Paired dataset count if auto-generated")
    parser.add_argument("--no_vgg", action="store_true", help="Disable VGG perceptual loss")
    parser.add_argument("--force_cpu", action="store_true", help="Force CPU mode")
    
    args = parser.parse_args()
    train_fgan(args)
