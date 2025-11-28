#!/usr/bin/env python3
import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
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


def build_model(variant: str, device: torch.device):
    if variant == "b0":
        ckpt_name = "nvidia/segformer-b0-finetuned-ade-512-512"
    elif variant == "b1":
        ckpt_name = "nvidia/segformer-b1-finetuned-ade-512-512"
    elif variant == "b2":
        ckpt_name = "nvidia/segformer-b2-finetuned-ade-512-512"
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


def load_checkpoint(model, ckpt_path: Path, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)


def prepare_samples(
    data_root: Path, split: str, num_samples: int, device: torch.device
):
    ds = RescueNetPatches(data_root, split=split)
    n = len(ds)
    indices = np.random.choice(n, size=min(num_samples, n), replace=False)
    imgs = []

    for idx in indices:
        img, _ = ds[idx]  # mask は不要
        imgs.append(img)

    imgs = torch.stack(imgs, dim=0)  # (N, 3, 512, 512)
    return imgs.to(device)


def measure_latency(model, images: torch.Tensor, warmup: int, iters: int):
    device = images.device
    torch.backends.cudnn.benchmark = True

    times = []

    # 1枚ずつ測る（バッチ1）
    def run_once(img: torch.Tensor):
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            out = model(pixel_values=img)
            # logits -> upsample (実際の推論と同じ処理を入れておくなら）
            _ = F.interpolate(
                out.logits,
                size=(512, 512),
                mode="bilinear",
                align_corners=False,
            )
        torch.cuda.synchronize()
        end = time.perf_counter()
        return (end - start) * 1000.0  # ms

    # warmup
    num_imgs = images.shape[0]
    for i in range(warmup):
        img = images[i % num_imgs : i % num_imgs + 1]
        _ = run_once(img)

    # 実測
    for i in range(iters):
        img = images[i % num_imgs : i % num_imgs + 1]
        t_ms = run_once(img)
        times.append(t_ms)

    times = np.array(times)
    return float(times.mean()), float(times.std()), times


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, default="src/dataset/RescueNet_patches")
    p.add_argument("--split", type=str, default="test")
    p.add_argument("--model", type=str, required=True, choices=["b0", "b1", "b2"])
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--num_samples", type=int, default=32)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--iters", type=int, default=100)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("WARNING: CUDA not available; Jetson で実行しているか確認しろ")

    data_root = Path(args.data_root)
    ckpt_path = Path(args.checkpoint)

    model = build_model(args.model, device)
    load_checkpoint(model, ckpt_path, device)

    images = prepare_samples(data_root, args.split, args.num_samples, device)

    mean_ms, std_ms, _ = measure_latency(
        model,
        images,
        warmup=args.warmup,
        iters=args.iters,
    )

    print(
        f"model={args.model}, "
        f"mean_latency={mean_ms:.3f} ms, std={std_ms:.3f} ms "
        f"(batch=1, H=W=512, iters={args.iters})"
    )


if __name__ == "__main__":
    main()
