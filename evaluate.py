import os
import argparse
import csv
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.generator import DeRainDeHazeGenerator
from data.dataset import PairedRoadDataset
from utils.metrics import calculate_psnr, calculate_ssim, calculate_mae, tensor_to_numpy_image
from utils.visualizer import save_comparison_grid
from utils.driving_eval import evaluate_autonomous_driving_readiness

def evaluate_model(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.force_cpu else "cpu")
    print(f"[*] Running Evaluation on: {device}")

    # Check model path
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found at '{args.checkpoint}'. Please run train.py first!")

    # Load generator
    generator = DeRainDeHazeGenerator(in_channels=3, out_channels=3, features=args.gen_features).to(device)
    state_dict = torch.load(args.checkpoint, map_location=device)
    generator.load_state_dict(state_dict)
    generator.eval()
    print(f"[+] Loaded weights from {args.checkpoint}")

    # Load test dataset
    test_dir = os.path.join(args.dataset_dir, "test")
    test_dataset = PairedRoadDataset(test_dir, image_size=args.img_size, is_train=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    os.makedirs(args.output_dir, exist_ok=True)
    comp_dir = os.path.join(args.output_dir, "comparisons")
    os.makedirs(comp_dir, exist_ok=True)

    records = []
    csv_path = os.path.join(args.output_dir, "evaluation_metrics.csv")
    csv_header = [
        "filename", "input_psnr", "restored_psnr", "psnr_gain",
        "input_ssim", "restored_ssim", "ssim_gain", "mae",
        "lane_edge_gain_pct", "contrast_gain_pct", "perception_score"
    ]

    print(f"[*] Benchmarking {len(test_dataset)} test road scenes...")
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(test_loader, desc="Evaluating")):
            fname = batch["filename"][0]
            degraded_t = batch["degraded"].to(device)
            clean_t = batch["clean"].to(device)

            restored_t = generator(degraded_t)

            deg_np = tensor_to_numpy_image(degraded_t[0])
            res_np = tensor_to_numpy_image(restored_t[0])
            gt_np = tensor_to_numpy_image(clean_t[0])

            # Standard Image Quality Metrics
            psnr_deg = calculate_psnr(deg_np, gt_np)
            psnr_res = calculate_psnr(res_np, gt_np)
            ssim_deg = calculate_ssim(deg_np, gt_np)
            ssim_res = calculate_ssim(res_np, gt_np)
            mae_res = calculate_mae(res_np, gt_np)

            # Autonomous driving perception metrics
            driving_metrics = evaluate_autonomous_driving_readiness(deg_np, res_np)

            rec = [
                fname,
                round(psnr_deg, 3),
                round(psnr_res, 3),
                round(psnr_res - psnr_deg, 3),
                round(ssim_deg, 4),
                round(ssim_res, 4),
                round(ssim_res - ssim_deg, 4),
                round(mae_res, 4),
                round(driving_metrics["edge_feature_gain_pct"], 2),
                round(driving_metrics["contrast_gain_pct"], 2),
                round(driving_metrics["perception_readiness_score"], 2)
            ]
            records.append(rec)

            # Save visual comparison for samples
            if idx < args.max_vis:
                save_path = os.path.join(comp_dir, f"eval_{idx:03d}_{fname}")
                save_comparison_grid(
                    degraded_t[0], restored_t[0], clean_t[0],
                    save_path,
                    title=f"Test Sample: {fname} (PSNR: {psnr_res:.2f}dB | SSIM: {ssim_res:.3f})"
                )

    with open(csv_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(csv_header)
        writer.writerows(records)

    # Compute averages
    avg_in_psnr = sum(r[1] for r in records) / len(records)
    avg_out_psnr = sum(r[2] for r in records) / len(records)
    avg_gain_psnr = sum(r[3] for r in records) / len(records)
    avg_in_ssim = sum(r[4] for r in records) / len(records)
    avg_out_ssim = sum(r[5] for r in records) / len(records)
    avg_gain_ssim = sum(r[6] for r in records) / len(records)
    avg_mae = sum(r[7] for r in records) / len(records)
    avg_edge_gain = sum(r[8] for r in records) / len(records)
    avg_contrast_gain = sum(r[9] for r in records) / len(records)
    avg_score = sum(r[10] for r in records) / len(records)

    print("\n" + "="*80)
    print("                     EVALUATION BENCHMARK RESULTS                     ")
    print("="*80)
    print(f"Total Test Images Evaluated: {len(records)}")
    print(f"Average Input PSNR:          {avg_in_psnr:.2f} dB")
    print(f"Average Restored PSNR:       {avg_out_psnr:.2f} dB  (Gain: +{avg_gain_psnr:.2f} dB)")
    print(f"Average Input SSIM:          {avg_in_ssim:.4f}")
    print(f"Average Restored SSIM:       {avg_out_ssim:.4f}  (Gain: +{avg_gain_ssim:.4f})")
    print(f"Average Restored MAE:        {avg_mae:.4f}")
    print(f"Avg Road Edge Feature Gain:  +{avg_edge_gain:.2f}%")
    print(f"Avg Road Contrast Gain:      +{avg_contrast_gain:.2f}%")
    print(f"Perception Readiness Score:  {avg_score:.2f} / 100")
    print("="*80)
    print(f"[+] Full Metrics saved to:   {csv_path}")
    print(f"[+] Visual Grids saved to:   {comp_dir}")
    print("="*80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate FGAN Road Rain & Haze Removal Model")
    parser.add_argument("--dataset_dir", type=str, default="dataset", help="Dataset directory")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_generator.pth", help="Path to weights")
    parser.add_argument("--output_dir", type=str, default="results/evaluation", help="Directory to save evaluation results")
    parser.add_argument("--img_size", type=int, default=256, help="Image resolution for inference")
    parser.add_argument("--gen_features", type=int, default=64, help="Generator feature dimension")
    parser.add_argument("--max_vis", type=int, default=15, help="Maximum number of comparison grids to save")
    parser.add_argument("--force_cpu", action="store_true", help="Force CPU inference")
    args = parser.parse_args()
    evaluate_model(args)
