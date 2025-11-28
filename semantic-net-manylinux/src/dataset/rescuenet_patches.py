from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class RescueNetPatches(Dataset):
    def __init__(self, root_dir, split="train"):
        self.root_dir = Path(root_dir)
        self.split = split
        self.img_dir = self.root_dir /f"RescueNet_patches"/ split / "images"
        self.mask_dir = self.root_dir /f"RescueNet_patches"/ split / "masks"

        self.img_paths = sorted(self.img_dir.glob("*.png"))
        assert len(self.img_paths) > 0, f"no patch images found in {self.img_dir}"

        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        mask_path = self.mask_dir / img_path.name

        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)

        img = np.array(img).astype(np.float32) / 255.0
        mask = np.array(mask).astype(np.int64)

        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img)

        for c in range(3):
            img[c] = (img[c] - self.mean[c]) / self.std[c]

        mask = torch.from_numpy(mask)

        return img, mask
