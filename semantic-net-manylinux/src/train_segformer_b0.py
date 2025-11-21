#!/usr/bin/env python3
import argparse
from pathlib import Path

import torch
import torch.nn as nn
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


def get_class_weights(device: torch.device) -> torch.Tensor:
    # 背景, 木: 0.25 / 車, プール: 2.0 / その他: 1.0
    w = torch.ones(NUM_CLASSES, dtype=torch.float32)
    w[0] = 0.25  # background
    w[9] = 0.25  # tree
    w[6] = 2.0  # vehicle
    w[10] = 2.0  # pool
    return w.to(device)


def get_dataloaders(root: Path, batch_size: int, num_workers: int):
    train_ds = RescueNetPatches(root, split="train")
    val_ds = RescueNetPatches(root, split="val")

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader


def build_model(device: torch.device):
    ckpt = "nvidia/segformer-b0-finetuned-ade-512-512"
    model = SegformerForSemanticSegmentation.from_pretrained(
        ckpt,
        num_labels=NUM_CLASSES,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,  # 150クラス→11クラスでheadを差し替え
    )
    return model.to(device)


@torch.no_grad()
def evaluate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    n_batches = 0

    # IoU用の分子・分母集計
    inter = torch.zeros(NUM_CLASSES, dtype=torch.float64, device=device)
    union = torch.zeros(NUM_CLASSES, dtype=torch.float64, device=device)

    for images, masks in tqdm(val_loader, desc="val", leave=False):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        outputs = model(pixel_values=images)
        logits = outputs.logits  # (B, C, h, w)
        # ラベルと同じ 512x512 にリサイズ
        logits = F.interpolate(
            logits,
            size=masks.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        loss = criterion(logits, masks)
        total_loss += loss.item()
        n_batches += 1

        preds = logits.argmax(dim=1)  # (B, H, W)

        for c in range(NUM_CLASSES):
            pred_c = preds == c
            mask_c = masks == c
            inter[c] += (pred_c & mask_c).sum()
            union[c] += (pred_c | mask_c).sum()

    avg_loss = total_loss / max(n_batches, 1)

    iou = inter / union.clamp(min=1)
    # union==0 のクラスは mIoU 計算から除外
    valid = union > 0
    miou = iou[valid].mean().item() if valid.any() else 0.0

    iou_cpu = iou.detach().cpu().tolist()

    return avg_loss, miou, iou_cpu


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = Path(args.data_root)

    train_loader, val_loader = get_dataloaders(
        root=root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    model = build_model(device)
    class_weights = get_class_weights(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-2,
    )
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
    )

    best_miou = 0.0
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        n_batches = 0

        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}")
        for images, masks in pbar:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(pixel_values=images)
            logits = outputs.logits
            logits = F.interpolate(
                logits,
                size=masks.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

            loss = criterion(logits, masks)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()
            n_batches += 1
            pbar.set_postfix(loss=running_loss / n_batches)

        lr_scheduler.step()

        # validation
        val_loss, val_miou, per_class_iou = evaluate(
            model, val_loader, criterion, device
        )

        print(
            f"[epoch {epoch}] "
            f"train_loss={running_loss / n_batches:.4f} "
            f"val_loss={val_loss:.4f} "
            f"val_mIoU={val_miou:.4f}"
        )

        # 必要ならクラス別 IoU も軽く表示
        if epoch % args.print_iou_every == 0:
            for c, iou_c in enumerate(per_class_iou):
                print(f"  class {c:2d} ({ID2LABEL[c]:>24s}): IoU={iou_c:.4f}")

        # ベストモデルを保存
        if val_miou > best_miou:
            best_miou = val_miou
            ckpt_path = out_dir / "best_segformer_b0.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_miou": best_miou,
                    "args": vars(args),
                },
                ckpt_path,
            )
            print(f"  -> best model updated: mIoU={best_miou:.4f}")

    # 最終モデルも保存しておく
    torch.save(model.state_dict(), out_dir / "last_segformer_b0.pt")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_root",
        type=str,
        default="/home/kinoko/development/github/sslab/dataset/RescueNet_patches",
    )
    parser.add_argument("--out_dir", type=str, default="./checkpoints_segformer_b0")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--print_iou_every", type=int, default=5)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
