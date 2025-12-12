# src/eval_segformer_fp16.py
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import SegformerForSemanticSegmentation

from dataset.rescuenet_patches import RescueNetPatches  # さっきのクラス

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


def build_model(model_name: str, device: torch.device):
    if model_name == "b0":
        ckpt = "nvidia/segformer-b0-finetuned-ade-512-512"
    elif model_name == "b1":
        ckpt = "nvidia/segformer-b1-finetuned-ade-512-512"
    elif model_name == "b2":
        ckpt = "nvidia/segformer-b2-finetuned-ade-512-512"
    else:
        raise ValueError(f"unknown model_name: {model_name}")

    model = SegformerForSemanticSegmentation.from_pretrained(
        ckpt,
        num_labels=NUM_CLASSES,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    ).to(device)
    return model


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()

    inter = torch.zeros(NUM_CLASSES, dtype=torch.float16, device=device)
    union = torch.zeros(NUM_CLASSES, dtype=torch.float16, device=device)

    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        outputs = model(pixel_values=images)
        logits = outputs.logits
        logits = F.interpolate(
            logits,
            size=masks.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        preds = logits.argmax(dim=1)

        for c in range(NUM_CLASSES):
            pred_c = preds == c
            mask_c = masks == c
            inter[c] += (pred_c & mask_c).sum()
            union[c] += (pred_c | mask_c).sum()

    iou = inter / union.clamp(min=1)
    valid = union > 0
    miou = iou[valid].mean().item() if valid.any() else 0.0
    return miou, iou.detach().cpu().tolist()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, default="src/dataset/RescueNet_patches")
    p.add_argument(
        "--split", type=str, default="test", choices=["train", "val", "test"]
    )
    p.add_argument("--model", type=str, required=True, choices=["b0", "b1", "b2"])
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=4)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = RescueNetPatches(args.data_root, split=args.split)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = build_model(args.model, device)

    # checkpoint 読み込み
    ckpt_path = Path(args.checkpoint)
    ckpt = torch.load(ckpt_path, map_location=device)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)

    miou, per_class_iou = evaluate(model, loader, device)

    print(f"split={args.split}, model={args.model}, mIoU={miou:.4f}")
    for c, iou_c in enumerate(per_class_iou):
        print(f"  class {c:2d} ({ID2LABEL[c]:>24s}): IoU={iou_c:.4f}")


if __name__ == "__main__":
    main()
