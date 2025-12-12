#!/usr/bin/env python3
"""
measure_latency_selector.py

Jetson 上で「セレクタ付き推論パイプライン」のレイテンシを計測するスクリプト。

パイプライン:
  1. 入力パッチを SegFormer-B0 に通す
  2. B0 の出力ロジットから 12 次元特徴量を計算
     - 11 クラスの出現割合
     - max softmax の平均
  3. Selector MLP に特徴量を入力して B0/B1 を選択 (0=B0, 1=B1)
  4. アクションが 1 のときだけ SegFormer-B1 を追加で推論

計測対象:
  - B0 + selector + (必要なら) B1 を含めた 1パッチあたりの処理時間 [ms]
  - B1 が選ばれた割合 (frac_B1)
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import SegformerConfig, SegformerForSemanticSegmentation

# ============================================================
# Dataset: RescueNetPatches (画像のみ, mask は不要)
# ============================================================


class RescueNetPatches(Dataset):
    def __init__(self, root_dir: str, split: str = "test"):
        self.root_dir = Path(root_dir)
        self.split = split
        self.img_dir = self.root_dir / split / "images"
        self.img_paths = sorted(self.img_dir.glob("*.png"))
        if len(self.img_paths) == 0:
            raise RuntimeError(f"No .png files found in {self.img_dir}")

        # ImageNet mean/std (SegFormer 学習時と合わせる)
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        path = self.img_paths[idx]
        img = Image.open(path).convert("RGB")
        arr = np.array(img).astype(np.float32) / 255.0  # HWC, [0,1]
        arr = arr.transpose(2, 0, 1)  # CHW
        tensor = torch.from_numpy(arr)  # (3,H,W)

        # normalize in-place
        for c in range(3):
            tensor[c] = (tensor[c] - self.mean[c]) / self.std[c]

        return tensor, path.name


# ============================================================
# SegFormer モデル構築 & checkpoint ロード
# ============================================================

NUM_CLASSES = 11


def build_segformer(
    variant: str, device: torch.device
) -> SegformerForSemanticSegmentation:
    """
    variant: 'b0' or 'b1'

    ここでは HuggingFace の「重み」は一切使わない。
    SegformerConfig だけを取得して、アーキテクチャを組み立てる。
    （重みは後で fine-tuned ckpt からロードする）
    """
    if variant not in ("b0", "b1"):
        raise ValueError(f"unknown variant: {variant}")

    hf_name = f"nvidia/segformer-{variant}-finetuned-ade-512-512"

    id2label = {
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
    label2id = {v: k for k, v in id2label.items()}

    # ★ 重要: from_pretrained(…) で .bin を読ませない。
    #   -> Config だけ取得して num_labels/id2label/label2id を上書き
    cfg = SegformerConfig.from_pretrained(
        hf_name,
        num_labels=NUM_CLASSES,
        id2label=id2label,
        label2id=label2id,
    )
    model = SegformerForSemanticSegmentation(cfg)
    model.to(device)
    model.eval()
    return model


def load_segformer_checkpoint(model: nn.Module, ckpt_path: Path, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict) and "model" in ckpt:
        state_dict = ckpt["model"]
    else:
        state_dict = ckpt
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[warn] missing keys in state_dict: {missing}")
    if unexpected:
        print(f"[warn] unexpected keys in state_dict: {unexpected}")


# ============================================================
# Selector MLP (学習と同じ構造にすること)
# ============================================================


class SelectorMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 2),  # 出力: [P(B0), P(B1)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ============================================================
# B0 ロジットから特徴量を作る
# ============================================================


def extract_features_from_logits(
    logits: torch.Tensor, num_classes: int = NUM_CLASSES
) -> torch.Tensor:
    """
    logits: (B, C, H, W) on device
    returns: (B, num_classes + 1) float32
      [0:C-1]  : 各クラスの出現割合
      [C]      : max softmax の平均
    """
    B, C, H, W = logits.shape
    probs = torch.softmax(logits, dim=1)  # (B,C,H,W)
    max_probs, _ = probs.max(dim=1)  # (B,H,W)
    preds = logits.argmax(dim=1)  # (B,H,W)
    total = H * W

    feats = torch.empty(B, num_classes + 1, device=logits.device, dtype=torch.float32)

    for i in range(B):
        hist = torch.bincount(
            preds[i].view(-1),
            minlength=num_classes,
        ).float() / float(total)
        mean_conf = max_probs[i].mean()
        feats[i, :num_classes] = hist
        feats[i, num_classes] = mean_conf

    return feats


# ============================================================
# レイテンシ計測ループ
# ============================================================


def measure_latency_selector(
    data_root: Path,
    split: str,
    ckpt_b0: Path,
    ckpt_b1: Path,
    selector_ckpt: Path,
    iters: int,
    warmup: int,
    batch_size: int,
    hidden_dim: int,
    device: torch.device,
    fp16: bool,
):
    if device.type != "cuda":
        raise RuntimeError("This script is intended to run on CUDA (Jetson)")

    torch.backends.cudnn.benchmark = True

    # dataset / loader
    ds = RescueNetPatches(str(data_root), split=split)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )
    print(f"dataset size (split={split}) = {len(ds)}")

    # models
    print("[build] SegFormer-B0/B1 ...")
    model_b0 = build_segformer("b0", device)
    model_b1 = build_segformer("b1", device)

    print(f"[load] B0 ckpt: {ckpt_b0}")
    load_segformer_checkpoint(model_b0, ckpt_b0, device)

    print(f"[load] B1 ckpt: {ckpt_b1}")
    load_segformer_checkpoint(model_b1, ckpt_b1, device)

    # selector
    in_dim = NUM_CLASSES + 1  # 11 + 1
    selector = SelectorMLP(in_dim=in_dim, hidden_dim=hidden_dim).to(device)
    print(f"[load] selector ckpt: {selector_ckpt}")
    selector.load_state_dict(torch.load(selector_ckpt, map_location=device))
    selector.eval()

    # timing
    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)

    times_ms = []
    n_measured = 0
    n_b1 = 0

    use_autocast = fp16 and (device.type == "cuda")

    with torch.no_grad():
        for imgs, _names in loader:
            imgs = imgs.to(device, non_blocking=True)

            if n_measured >= warmup + iters:
                break

            starter.record()

            # 1) B0 forward
            with torch.cuda.amp.autocast(enabled=use_autocast):
                out_b0 = model_b0(pixel_values=imgs)
            logits_b0 = out_b0.logits  # (B,C,H,W)

            # 2) 特徴量抽出
            feats = extract_features_from_logits(
                logits_b0, num_classes=NUM_CLASSES
            )  # (B,12)

            # 3) selector による B0/B1 選択
            logits_sel = selector(feats)  # (B,2)
            actions = logits_sel.argmax(dim=1)  # 0=B0, 1=B1

            # 4) action == 1 の場合のみ B1 も推論
            if (actions == 1).any():
                with torch.cuda.amp.autocast(enabled=use_autocast):
                    _ = model_b1(pixel_values=imgs)
                n_b1 += int((actions == 1).sum().item())

            ender.record()
            torch.cuda.synchronize()

            if n_measured >= warmup:
                elapsed = starter.elapsed_time(ender)  # ms
                times_ms.append(elapsed)

            n_measured += 1

    if len(times_ms) == 0:
        raise RuntimeError(
            "No measurements collected (check iters/warmup/dataset size)"
        )

    times_ms = np.array(times_ms, dtype=np.float64)
    avg_ms = float(times_ms.mean())
    std_ms = float(times_ms.std())
    frac_b1 = n_b1 / float(len(times_ms) * batch_size)

    print("\n=== selector pipeline latency (Jetson) ===")
    print(f"  split         : {split}")
    print(f"  samples       : {len(times_ms)} (iters={iters}, warmup={warmup})")
    print(f"  batch_size    : {batch_size}")
    print(f"  avg_latency   : {avg_ms:.3f} ms / patch")
    print(f"  std_latency   : {std_ms:.3f} ms")
    print(f"  frac_B1       : {frac_b1 * 100:.2f} %  (B1 を選択した割合)")


# ============================================================
# CLI
# ============================================================


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="RescueNet_patches の root ディレクトリ (例: src/dataset/RescueNet_patches)",
    )
    p.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="使用する split",
    )
    p.add_argument(
        "--ckpt_b0",
        type=str,
        required=True,
        help="SegFormer-B0 の fine-tuned checkpoint (.pt)",
    )
    p.add_argument(
        "--ckpt_b1",
        type=str,
        required=True,
        help="SegFormer-B1 の fine-tuned checkpoint (.pt)",
    )
    p.add_argument(
        "--selector_ckpt",
        type=str,
        required=True,
        help="SelectorMLP の checkpoint (.pt, λ ごとに別ファイル)",
    )
    p.add_argument(
        "--iters",
        type=int,
        default=100,
        help="計測に使うサンプル数 (warmup を除く)",
    )
    p.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="ウォームアップ用に捨てるサンプル数",
    )
    p.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="DataLoader の batch size (レイテンシを見るなら基本 1 を推奨)",
    )
    p.add_argument(
        "--hidden_dim",
        type=int,
        default=64,
        help="SelectorMLP の hidden_dim (学習時と合わせる)",
    )
    p.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="使用デバイス (Jetson なら cuda 固定でよい)",
    )
    p.add_argument(
        "--fp16",
        action="store_true",
        help="半精度 (AMP) を有効化して計測する場合に指定",
    )
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    data_root = Path(args.data_root)
    ckpt_b0 = Path(args.ckpt_b0)
    ckpt_b1 = Path(args.ckpt_b1)
    selector_ckpt = Path(args.selector_ckpt)

    print(f"device        = {device}")
    print(f"data_root     = {data_root}")
    print(f"split         = {args.split}")
    print(f"ckpt_b0       = {ckpt_b0}")
    print(f"ckpt_b1       = {ckpt_b1}")
    print(f"selector_ckpt = {selector_ckpt}")
    print(f"iters         = {args.iters}")
    print(f"warmup        = {args.warmup}")
    print(f"batch_size    = {args.batch_size}")
    print(f"hidden_dim    = {args.hidden_dim}")
    print(f"fp16          = {args.fp16}")

    if not data_root.is_dir():
        raise FileNotFoundError(data_root)
    if not ckpt_b0.is_file():
        raise FileNotFoundError(ckpt_b0)
    if not ckpt_b1.is_file():
        raise FileNotFoundError(ckpt_b1)
    if not selector_ckpt.is_file():
        raise FileNotFoundError(selector_ckpt)

    t0 = time.time()
    measure_latency_selector(
        data_root=data_root,
        split=args.split,
        ckpt_b0=ckpt_b0,
        ckpt_b1=ckpt_b1,
        selector_ckpt=selector_ckpt,
        iters=args.iters,
        warmup=args.warmup,
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim,
        device=device,
        fp16=args.fp16,
    )
    t1 = time.time()
    print(f"\n[done] total elapsed wall time = {t1 - t0:.1f} sec")


if __name__ == "__main__":
    main()
