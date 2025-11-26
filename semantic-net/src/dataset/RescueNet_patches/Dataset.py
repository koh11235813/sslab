from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class RescueNetPatches(Dataset):
    def __init__(self, root_dir, split="train", transform=None):
        print(root_dir)
        self.root_dir = Path(root_dir)
        print(self.root_dir)
        self.split = split
        self.img_dir = self.root_dir / split / "images"
        self.mask_dir = self.root_dir / split / "masks"
        print(self.img_dir)
        self.transform = transform  # 画像側への変換（ToTensor + Normalizeなど）

        self.img_paths = sorted(self.img_dir.glob("*.png"))
        assert len(self.img_paths) > 0, "no patch images found"

        # ImageNet の mean/std （SegFormer 事前学習に合わせる）
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
        mask = np.array(mask).astype(np.int64)  # ラベルID 0〜10

        # HWC -> CHW
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img)
        # 正規化
        for c in range(3):
            img[c] = (img[c] - self.mean[c]) / self.std[c]

        mask = torch.from_numpy(mask)  # (H, W), long

        # 必要ならここでランダムフリップなどの data augmentation を入れる
        # transform を torchvision.transforms.v2 に寄せるなら、img/mask 一緒に扱うように設計し直してもいい

        return img, mask
