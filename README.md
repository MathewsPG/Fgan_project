# Rain and Haze Removal from Road Images for Autonomous Driving (FGAN)

[![PyTorch](https://img.shields.io/badge/PyTorch-2.5%2Bcu121-EE4C2C.svg?logo=pytorch)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Autonomous Driving](https://img.shields.io/badge/Application-Autonomous_Driving-success.svg)](https://github.com)

An end-to-end deep learning framework based on a **Feature-Attention Conditional Generative Adversarial Network (FGAN)** for simultaneous rain streak de-raining and atmospheric haze de-hazing on road and dashcam imagery.

> **Academic Integrity Notice:**  
> This project implements, trains, and evaluates a standalone PyTorch neural network directly from source code on paired road driving datasets. It relies on **zero third-party cloud inference APIs** (no OpenAI, no Gemini, no Hugging Face Inference endpoints). All weights, losses, and inference pipelines are 100% locally trained and reproducible.

---

## 1. Problem Formulation & Autonomous Driving Motivation

Adverse meteorological conditions—specifically **rain streaks** and **atmospheric haze/fog**—severely degrade visual perception for autonomous vehicles:
1. **Atmospheric Haze** causes contrast degradation and loss of distant road visibility according to Koschmieder's Law:
   $$I(x) = J(x)t(x) + A(1 - t(x))$$
   where $J(x)$ is clean scene radiance, $A$ is global atmospheric light, and $t(x) = e^{-\beta d(x)}$ is depth-dependent transmission.
2. **Rain Streaks** introduce high-frequency directional noise that occludes lane markings, traffic signs, and pedestrian boundaries:
   $$O(x) = I(x) + \sum_{i} S_i(x)$$
3. **Compound Degradation**: During storms, both phenomena occur simultaneously, catastrophic for downstream perception systems (YOLO, Mask R-CNN, lane segmentation networks).

---

## 2. Architecture: Feature-Attention GAN (FGAN)

The system consists of two synergistic neural networks:

### A. Generator: Dual-Attention ResU-Net
- **Encoder**: Progressive multi-scale convolutions capturing spatial road hierarchy.
- **Dual-Attention Bottleneck**:
  - **Channel Attention Module (CAM)**: Dynamically weights feature channels to separate rain noise from semantic objects (vehicles, road surface).
  - **Spatial Attention Module (SAM)**: Identifies spatial clusters of dense fog and streak accumulation across the horizon and road plane.
- **Decoder & Global Residual Connection**: U-Net skip connections preserve fine road boundaries, with a global residual formulation:
  $$J_{pred} = \text{clamp}(I_{degraded} + \mathcal{R}_\theta(I_{degraded}), -1, 1)$$

### B. Discriminator: $70 \times 70$ PatchGAN
- Conditions on both degraded input and restored/clean target: $[I_{degraded}, J_{restored}]$.
- Evaluates local $70 \times 70$ receptive field patches to enforce sharp edge transitions on lane markers and obstacle contours rather than blurry averages.

---

## 3. Multi-Objective Loss Formulation

Training is driven by a balanced combination of 5 complementary loss objectives:

$$\mathcal{L}_{total} = \lambda_{adv}\mathcal{L}_{GAN} + \lambda_{L1}\mathcal{L}_{L1} + \lambda_{SSIM}\mathcal{L}_{SSIM} + \lambda_{VGG}\mathcal{L}_{VGG} + \lambda_{edge}\mathcal{L}_{edge}$$

1. **Adversarial Loss ($\mathcal{L}_{GAN}$)**: Encourages natural photographic realism.
2. **L1 Reconstruction ($\mathcal{L}_{L1}$)**: Enforces pixel-wise color constancy.
3. **SSIM Loss ($\mathcal{L}_{SSIM}$)**: Maximizes structural similarity: $1 - \text{SSIM}(J_{pred}, J_{gt})$.
4. **VGG-19 Perceptual Loss ($\mathcal{L}_{VGG}$)**: Matches deep feature representations at `relu1_2`, `relu2_2`, `relu3_4`, and `relu4_4`.
5. **Sobel Edge Loss ($\mathcal{L}_{edge}$)**: Explicitly penalizes blurred lane lines and vehicle edges using spatial gradient filters.

---

## 4. Project Structure

```
fgan/
├── data/
│   ├── download_dataset.py       # Automated dataset generator & benchmark fetcher
│   ├── dataset.py                # PyTorch Paired Dataset loader with augmentations
│   └── synthetic_generator.py    # Physics-based atmospheric scattering & rain streak simulator
├── models/
│   ├── generator.py              # Feature-Attention U-Net Generator (CAM + SAM)
│   ├── discriminator.py          # Conditional PatchGAN Discriminator
│   └── losses.py                 # Multi-task loss (L1, Perceptual, SSIM, Edge, GAN)
├── utils/
│   ├── metrics.py                # PSNR, SSIM, MAE, and sharpness calculation
│   ├── visualizer.py             # Side-by-side comparison grids & training curve plotting
│   └── driving_eval.py           # Downstream autonomous driving readiness evaluator
├── checkpoints/                  # Saved model weights (.pth)
├── results/                      # Evaluation metrics CSV, visual grids, loss curves
├── train.py                      # Standalone GPU training script with AMP & checkpoints
├── evaluate.py                   # Quantitative benchmarking script
├── demo_app.py                   # Local interactive GUI/Dashboard (Streamlit)
└── requirements.txt              # Project dependencies
```

---

## 5. Quickstart: Step-by-Step Execution

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Prepare / Generate Dataset
To create or download the paired road training and testing dataset:
```bash
python data/download_dataset.py
```
This prepares paired samples in `dataset/train/` and `dataset/test/` containing both clean road scenes and corresponding rain+haze degraded scenes.

### Step 3: Train the FGAN Model
Run model training with automatic GPU acceleration (NVIDIA RTX 3050):
```bash
python train.py --epochs 25 --batch_size 4 --lr_g 2e-4 --lr_d 1e-4
```
- Real-time progress bars show Generator Loss, Discriminator Loss, and L1 Loss.
- Validation PSNR and SSIM are evaluated every epoch.
- Best model weights are automatically saved to `checkpoints/best_generator.pth`.
- Visual comparisons are saved to `results/visuals/`.

### Step 4: Evaluate Test Metrics
Compute quantitative benchmark metrics across the test dataset:
```bash
python evaluate.py --checkpoint checkpoints/best_generator.pth
```
Outputs:
- **Peak Signal-to-Noise Ratio (PSNR)** in dB
- **Structural Similarity Index Measure (SSIM)**
- **Mean Absolute Error (MAE)**
- **Road Edge & Lane Clarity Gain (%)**
- **Autonomous Driving Perception Readiness Score**
- Full CSV report saved to `results/evaluation/evaluation_metrics.csv`

### Step 5: Launch Local Interactive Demo
Run the local web dashboard for interactive real-time inference:
```bash
streamlit run demo_app.py
```
Upload road driving dashcam frames or synthesize rain/haze on the fly with dynamic sliders and inspect real-time restoration and lane edge detection!
