#!/usr/bin/env python3
"""
Precompute per-patch semantic values for SegFormer-B0 / B1.

- 入力:
    - RescueNet_patches の train / val / test
    - 学習済み SegFormer-B0 / B1 の checkpoint (.pt)
- 出力:
    - CSV: 1 行 = 1 パッチ
      columns:
        split, index, patch, value_b0, value_b1, miou_b0, miou_b1

Value の定義:
    V = sum_c w_c * IoU_c

w_c は「建物損傷度・道路・水」を重くするようにここで決めている。
必要なら VALUE_WEIGHTS を書き換えろ。
"""

import argparse
import csv
from contextlib import nullcontext
from pathlib import Path
from typing import List

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import SegformerForSemanticSegmentation

from dataset.rescuenet_patches import RescueNetPatches

NUM_CLASSES = 11

ID2LABEL = {
    0: "background",
    1: "water",
    2: "building_no_damage",
    3: "building_minor_damage",
    4: "building_major_damage",
    5: "building_total_destruction",
    6: "vehicle",
    7: "road_clear",
    8: "road_blocked",
    9: "tree",
    10: "pool",
}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}

# -----------------------------
# Value 用のクラス重み w_c
# -----------------------------
# 方針:
#   - background, tree は軽め
#   - water, building*, road* は重め
#   - vehicle / pool は中くらい
VALUE_WEIGHTS = torch.tensor(
    [
        0.1,  # 0: background
        1.0,  # 1: water
        1.0,  # 2: building_no_damage
        2.0,  # 3: building_minor_damage
        2.0,  # 4: building_major_damage
        2.0,  # 5: building_total_destruction
        1.0,  # 6: vehicle
        2.0,  # 7: road_clear
        2.0,  # 8: road_blocked
        0.5,  # 9: tree
        1.0,  # 10: pool
    ],
    dtype=torch.float32,
)


def build_model(variant: str, device: torch.device) -> SegformerForSemanticSegmentation:
    if variant == "b0":
        ckpt_name = "nvidia/segformer-b0-finetuned-ade-512-512"
    elif variant == "b1":
        ckpt_name = "nvidia/segformer-b1-finetuned-ade-512-512"
    else:
        raise ValueError(f"unknown variant: {variant}")

    model = SegformerForSemanticSegmentation.from_pretrained(
        ckpt_name,
        num_labels=NUM_CLASSES,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
        use_safetensors=True,
    )
    model.to(device)
    model.eval()
    return model


def load_checkpoint(
    model: torch.nn.Module, ckpt_path: Path, device: torch.device
) -> None:
    """train_segformer_b0/b1.py と同じ形式の checkpoint を読む."""
    ckpt = torch.load(ckpt_path, map_location=device)
    if "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    else:
        # もし state_dict そのものを保存していた場合の fallback
        state_dict = ckpt
    model.load_state_dict(state_dict)
    print(f"loaded checkpoint from {ckpt_path}")


@torch.no_grad()
def compute_per_class_iou(
    pred: torch.Tensor, target: torch.Tensor, num_classes: int
) -> torch.Tensor:
    """
    pred, target: (H, W), int64
    return: (num_classes,) torch.float32, 各クラスの IoU
    """
    pred = pred.view(-1).to(torch.int64).cpu()
    target = target.view(-1).to(torch.int64).cpu()

    valid = (target >= 0) & (target < num_classes)
    pred = pred[valid]
    target = target[valid]

    cm = (
        torch.bincount(
            target * num_classes + pred,
            minlength=num_classes * num_classes,
        )
        .reshape(num_classes, num_classes)
        .float()
    )

    intersection = torch.diag(cm)
    gt_count = cm.sum(dim=1)
    pred_count = cm.sum(dim=0)
    union = gt_count + pred_count - intersection

    iou = intersection / union.clamp(min=1.0)
    return iou  # (num_classes,)


def compute_value_from_iou(iou: torch.Tensor) -> float:
    """
    iou: (num_classes,)
    V = sum_c w_c * IoU_c を返す
    """
    v = (iou * VALUE_WEIGHTS.to(iou.device)).sum()
    return float(v.item())


@torch.no_grad()
def process_split(
    split: str,
    data_root: Path,
    model_b0: SegformerForSemanticSegmentation,
    model_b1: SegformerForSemanticSegmentation,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    use_autocast: bool,
) -> List[dict]:
    """
    1 split 分の Value_B0 / Value_B1 を計算して、行 (dict) のリストを返す。
    """
    ds = RescueNetPatches(str(data_root), split=split)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    rows = []
    global_idx = 0
    if use_autocast and device.type == "cuda":
        amp_ctx = torch.amp.autocast("cuda", dtype=torch.float16)
    else:
        amp_ctx = nullcontext()

    pbar = tqdm(loader, desc=f"split={split}", total=len(loader))
    for imgs, masks in pbar:
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        with amp_ctx:
            out_b0 = model_b0(pixel_values=imgs)
            logits_b0 = out_b0.logits  # (B, C, H, W)
            logits_b0 = F.interpolate(
                logits_b0,
                size=masks.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

            out_b1 = model_b1(pixel_values=imgs)
            logits_b1 = out_b1.logits  # (B, C, H, W)
            logits_b1 = F.interpolate(
                logits_b1,
                size=masks.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        preds_b0 = logits_b0.argmax(dim=1)  # (B, H, W)
        preds_b1 = logits_b1.argmax(dim=1)  # (B, H, W)

        bsz = imgs.size(0)
        for bi in range(bsz):
            idx = global_idx + bi
            patch_path = ds.img_paths[idx]

            iou_b0 = compute_per_class_iou(preds_b0[bi], masks[bi], NUM_CLASSES)
            iou_b1 = compute_per_class_iou(preds_b1[bi], masks[bi], NUM_CLASSES)

            value_b0 = compute_value_from_iou(iou_b0)
            value_b1 = compute_value_from_iou(iou_b1)

            miou_b0 = float(iou_b0.mean().item())
            miou_b1 = float(iou_b1.mean().item())

            rows.append(
                {
                    "split": split,
                    "index": idx,
                    "patch": patch_path.name,
                    "value_b0": value_b0,
                    "value_b1": value_b1,
                    "miou_b0": miou_b0,
                    "miou_b1": miou_b1,
                }
            )

        global_idx += bsz

    return rows


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_root",
        type=str,
        default="src/dataset",
    )
    parser.add_argument(
        "--splits",
        type=str,
        default="train,val,test",
        help="comma-separated list, e.g. 'train,val' or 'test'",
    )
    parser.add_argument(
        "--ckpt_b0",
        type=str,
        default="checkpoints_segformer_b0/best_segformer_b0.pt",
    )
    parser.add_argument(
        "--ckpt_b1",
        type=str,
        default="checkpoints_segformer_b1/best_segformer_b1.pt",
    )
    parser.add_argument(
        "--out_csv",
        type=str,
        default="precomputed_values_segformer_b0_b1.csv",
    )
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="use torch.amp.autocast('cuda', dtype=torch.float16) for model forward",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    data_root = Path(args.data_root)
    out_csv = Path(args.out_csv)
    device = torch.device(args.device)

    print(f"device = {device}")
    print(f"data_root = {data_root}")
    print(f"out_csv = {out_csv}")

    print("building models...")
    model_b0 = build_model("b0", device)
    model_b1 = build_model("b1", device)

    print("loading checkpoints...")
    load_checkpoint(model_b0, Path(args.ckpt_b0), device)
    load_checkpoint(model_b1, Path(args.ckpt_b1), device)

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    all_rows: List[dict] = []

    for split in splits:
        print(f"=== processing split={split} ===")
        rows = process_split(
            split=split,
            data_root=data_root,
            model_b0=model_b0,
            model_b1=model_b1,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=device,
            use_autocast=args.fp16,
        )
        all_rows.extend(rows)

    fieldnames = [
        "split",
        "index",
        "patch",
        "value_b0",
        "value_b1",
        "miou_b0",
        "miou_b1",
    ]

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"done. wrote {len(all_rows)} rows to {out_csv}")


if __name__ == "__main__":
    main()
