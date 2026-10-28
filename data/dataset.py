import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

class PairedRoadDataset(Dataset):
    """
    PyTorch Dataset for Paired Rain/Haze Road Scenes.
    Expects directory structure:
        root_dir/
            degraded/
            clean/
    Both folders contain identically named image files.
    """
    def __init__(self, root_dir, image_size=256, is_train=True):
        self.root_dir = root_dir
        self.degraded_dir = os.path.join(root_dir, "degraded")
        self.clean_dir = os.path.join(root_dir, "clean")
        
        valid_extensions = {".png", ".jpg", ".jpeg", ".bmp"}
        self.filenames = [
            f for f in os.listdir(self.degraded_dir)
            if os.path.splitext(f.lower())[1] in valid_extensions
        ]
        self.filenames.sort()
        self.image_size = (image_size, image_size)
        self.is_train = is_train

    def __len__(self):
        return len(self.filenames)

    def _to_tensor(self, img_pil):
        # Resize using PIL Resampling
        img_resized = img_pil.resize(self.image_size, Image.Resampling.BILINEAR)
        img_np = np.array(img_resized, dtype=np.float32)
        # Normalize from [0, 255] to [-1.0, 1.0]
        img_norm = (img_np / 127.5) - 1.0
        # Convert (H, W, C) -> (C, H, W)
        tensor = torch.from_numpy(img_norm).permute(2, 0, 1).float()
        return tensor

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        degraded_path = os.path.join(self.degraded_dir, fname)
        clean_path = os.path.join(self.clean_dir, fname)

        degraded_img = Image.open(degraded_path).convert("RGB")
        clean_img = Image.open(clean_path).convert("RGB")

        # Synchronized Random Horizontal Flip during training
        if self.is_train and np.random.rand() > 0.5:
            degraded_img = degraded_img.transpose(Image.FLIP_LEFT_RIGHT)
            clean_img = clean_img.transpose(Image.FLIP_LEFT_RIGHT)

        degraded_tensor = self._to_tensor(degraded_img)
        clean_tensor = self._to_tensor(clean_img)

        return {
            "degraded": degraded_tensor,
            "clean": clean_tensor,
            "filename": fname
        }
