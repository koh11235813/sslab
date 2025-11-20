#!/usr/bin/env python3
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path("/home/kinoko/development/github/sslab/dataset/RescueNet")
OUT_ROOT = Path("/home/kinoko/development/github/sslab/dataset/RescueNet_patches")

PATCH = 512
RESIZE_W, RESIZE_H = 2048, 1536
BG_ID = 0
FG_RATIO_MIN = 0.01  # foreground <=1% なら捨てる

SPLITS = ["train", "val", "test"]


def is_mostly_background(mask_patch: np.ndarray, bg_id: int = BG_ID) -> bool:
    """背景割合が高すぎるパッチかどうか"""
    if mask_patch.ndim == 3:
        mask_patch = mask_patch[..., 0]
    total = mask_patch.size
    bg = np.sum(mask_patch == bg_id)
    return (total == 0) or (bg / total >= 1.0 - FG_RATIO_MIN)


def process_split(split: str):
    img_dir = ROOT / split / f"{split}-org-img"
    mask_dir = ROOT / split / f"{split}-label-img"

    out_img_dir = OUT_ROOT / split / "images"
    out_mask_dir = OUT_ROOT / split / "masks"
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_mask_dir.mkdir(parents=True, exist_ok=True)

    img_paths = sorted(img_dir.glob("*.jpg"))
    print(f"[{split}] #images = {len(img_paths)}")

    total_patches = 0
    kept_patches = 0

    for img_path in img_paths:
        img_id = img_path.stem  # "10778"
        mask_path = mask_dir / f"{img_id}_lab.png"
        if not mask_path.exists():
            raise FileNotFoundError(f"mask not found: {mask_path}")

        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)

        # 4:3 のまま縮小
        img = img.resize((RESIZE_W, RESIZE_H), resample=Image.BILINEAR)
        mask = mask.resize((RESIZE_W, RESIZE_H), resample=Image.NEAREST)

        img_np = np.array(img)
        mask_np = np.array(mask)

        H, W = mask_np.shape[:2]
        if img_np.shape[0] != H or img_np.shape[1] != W:
            raise ValueError(f"size mismatch: {img_path} vs {mask_path}")

        # 念のためユニーク値を確認（最初の数枚だけ print してもいい）
        # print(img_id, np.unique(mask_np))

        for y in (0, 512, 1024):  # 1536 = 3 * 512
            for x in (0, 512, 1024, 1536):  # 2048 = 4 * 512
                img_patch = img_np[y : y + PATCH, x : x + PATCH]
                mask_patch = mask_np[y : y + PATCH, x : x + PATCH]

                if img_patch.shape[0] != PATCH or img_patch.shape[1] != PATCH:
                    continue  # 念のため端でずれないように保険

                total_patches += 1

                if is_mostly_background(mask_patch):
                    continue

                patch_name = f"{img_id}_y{y}_x{x}.png"
                Image.fromarray(img_patch).save(out_img_dir / patch_name)
                Image.fromarray(mask_patch).save(out_mask_dir / patch_name)
                kept_patches += 1

    print(f"[{split}] total patches = {total_patches}, kept = {kept_patches}")


def main():
    for split in SPLITS:
        process_split(split)


if __name__ == "__main__":
    main()
