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
import csv
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import SegformerConfig, SegformerForSemanticSegmentation

from dataset.rescuenet_patches import RescueNetPatches

# ============================================================
# Dataset: RescueNetPatches (画像のみ, mask は不要)
# ============================================================


# class RescueNetPatches(Dataset):
#     def __init__(self, root_dir: str, split: str = "test"):
#         self.root_dir = Path(root_dir)
#         self.split = split
#         self.img_dir = self.root_dir / split / "images"
#         self.img_paths = sorted(self.img_dir.glob("*.png"))
#         if len(self.img_paths) == 0:
#             raise RuntimeError(f"No .png files found in {self.img_dir}")

#         # ImageNet mean/std (SegFormer 学習時と合わせる)
#         self.mean = [0.485, 0.456, 0.406]
#         self.std = [0.229, 0.224, 0.225]

#     def __len__(self):
#         return len(self.img_paths)

#     def __getitem__(self, idx):
#         path = self.img_paths[idx]
#         img = Image.open(path).convert("RGB")
#         arr = np.array(img).astype(np.float32) / 255.0  # HWC, [0,1]
#         arr = arr.transpose(2, 0, 1)  # CHW
#         tensor = torch.from_numpy(arr)  # (3,H,W)

#         # normalize in-place
#         for c in range(3):
#             tensor[c] = (tensor[c] - self.mean[c]) / self.std[c]

#         return tensor, path.name


# ============================================================
# SegFormer モデル構築 & checkpoint ロード
# ============================================================

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


def init_iou_stats(device: torch.device):
    # 半精度でもカウントは float64 にしておいたほうが安全
    inter = torch.zeros(NUM_CLASSES, dtype=torch.float64, device=device)
    union = torch.zeros(NUM_CLASSES, dtype=torch.float64, device=device)
    return inter, union


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

    label2id = {v: k for k, v in ID2LABEL.items()}

    # ★ 重要: from_pretrained(…) で .bin を読ませない。
    #   -> Config だけ取得して num_labels/id2label/label2id を上書き
    cfg = SegformerConfig.from_pretrained(
        hf_name,
        num_labels=NUM_CLASSES,
        id2label=ID2LABEL,
        label2id=label2id,
    )
    model = SegformerForSemanticSegmentation(cfg)
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def update_iou_stats(
    inter: torch.Tensor, union: torch.Tensor, preds: torch.Tensor, masks: torch.Tensor
):
    """
    preds, masks: (B, H, W)
    inter/union を in-place で更新
    """
    # バッチを 1 枚ずつ処理（batch_size=1 前提なら for はほぼタダ）
    for b in range(preds.size(0)):
        pred = preds[b]
        target = masks[b]

        for c in range(NUM_CLASSES):
            pred_c = pred == c
            target_c = target == c

            # & / | の結果は bool → sum() でカウントになる
            inter[c] += (pred_c & target_c).sum()
            union[c] += (pred_c | target_c).sum()


def finalize_iou(inter: torch.Tensor, union: torch.Tensor):
    # ゼロ割回避
    iou = inter / union.clamp(min=1)
    valid = union > 0
    if valid.any():
        miou = iou[valid].mean().item()
    else:
        miou = 0.0
    return miou, iou.detach().cpu().tolist()


# CSV を読み込む関数
def load_precomputed_metrics(csv_path: Path, split: str) -> Dict[str, Dict[str, float]]:
    """
    事前計算済みの CSV から mIoU / value を読み込んで、
    patch ファイル名 -> 指標 の dict にして返す。

    期待する CSV カラム:
      - split
      - patch
      - value_b0, value_b1
      - miou_b0, miou_b1
    """
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"precomputed CSV not found: {csv_path}")

    table: Dict[str, Dict[str, float]] = {}
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        required = {"split", "patch", "miou_b0", "miou_b1"}
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {csv_path}")
        missing = required.difference(reader.fieldnames)
        if missing:
            raise ValueError(
                f"CSV {csv_path} is missing required columns: {sorted(missing)} "
                f"(found={reader.fieldnames})"
            )

        for row in reader:
            if row["split"] != split:
                continue
            patch = row["patch"]
            table[patch] = {
                "miou_b0": float(row["miou_b0"]),
                "miou_b1": float(row["miou_b1"]),
                # おまけで value も握っておく（今は使わないが将来用）
                "value_b0": float(row.get("value_b0", 0.0)),
                "value_b1": float(row.get("value_b1", 0.0)),
            }

    if not table:
        raise RuntimeError(
            f"no rows for split={split!r} in precomputed CSV: {csv_path}"
        )

    return table


def load_segformer_checkpoint(model: nn.Module, ckpt_path: Path, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device)
    # 学習スクリプト側の保存形式に合わせる
    # torch.save({
    #   'epoch': ...,
    #   'model_state_dict': model.state_dict(),
    #   'optimizer_state_dict': ...,
    #   'best_miou': ...,
    #   'args': ...,
    # }, path)
    if isinstance(ckpt, dict):
        if "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        elif "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        else:
            # そのまま state_dict だとみなす
            state_dict = ckpt
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
        # ★ 学習時と同じ 2 層 MLP:
        #   Linear(in_dim, hidden_dim) -> ReLU -> Linear(hidden_dim, 2)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 2),
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


@torch.inference_mode()
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
    precomputed_csv: Optional[Path] = None,
) -> None:
    """
    RL セレクタ付きパイプラインのレイテンシを測定し、
    precomputed CSV から mIoU（B0-only, B1-only, RL selector）も算出して出力する。
    ここでは Jetson 側で GT マスクを読むのではなく、
    事前に CPU/GPU 環境で計算しておいた mIoU を利用する。
    """
    if device.type != "cuda":
        print(
            "[WARN] device is not CUDA; latency numbers will not be comparable to Jetson."
        )

    if not ckpt_b0.is_file():
        raise FileNotFoundError(f"B0 checkpoint not found: {ckpt_b0}")
    if not ckpt_b1.is_file():
        raise FileNotFoundError(f"B1 checkpoint not found: {ckpt_b1}")
    if not selector_ckpt.is_file():
        raise FileNotFoundError(f"selector checkpoint not found: {selector_ckpt}")

    # cuDNN 最適化
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    # SegFormer B0/B1 を構築して ckpt をロード
    print("[build] SegFormer-B0 / B1")
    model_b0 = build_segformer(variant="b0", device=device)
    model_b1 = build_segformer(variant="b1", device=device)
    load_segformer_checkpoint(model_b0, ckpt_b0, device=device)
    load_segformer_checkpoint(model_b1, ckpt_b1, device=device)
    model_b0.eval()
    model_b1.eval()

    # セレクタ MLP を構築・ロード
    print("[build] Selector MLP")
    in_dim = NUM_CLASSES + 1  # per-class ratio (11) + mean confidence (1)
    selector = SelectorMLP(in_dim=in_dim, hidden_dim=hidden_dim).to(device)
    sel_state = torch.load(selector_ckpt, map_location=device)
    if isinstance(sel_state, dict) and "model_state_dict" in sel_state:
        selector.load_state_dict(sel_state["model_state_dict"])
    else:
        selector.load_state_dict(sel_state)
    selector.eval()

    # Dataset / DataLoader
    dataset = RescueNetPatches(data_root, split=split)
    if len(dataset) == 0:
        raise RuntimeError(f"dataset is empty: root={data_root}, split={split}")
    print(f"[dataset] split={split}, size={len(dataset)}")

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    # autocast context
    if fp16 and device.type == "cuda":
        amp_dtype = torch.float16
        amp_ctx = torch.amp.autocast(device_type="cuda", dtype=amp_dtype)
        print("[amp] fp16 enabled for selector pipeline")
    else:
        from contextlib import nullcontext

        amp_ctx = nullcontext()
        print("[amp] fp32 (no autocast) for selector pipeline")

    # CUDA Event for timing
    if device.type == "cuda":
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
    else:
        starter = ender = None

    # 事前計算済み mIoU をロード
    metrics = None
    if precomputed_csv is not None:
        print(f"[metrics] load precomputed metrics from: {precomputed_csv}")
        metrics = load_precomputed_metrics(precomputed_csv, split=split)
    else:
        print("[metrics] precomputed_csv is None -> IoU 集計はスキップされる")

    times_ms: List[float] = []
    n_b1 = 0
    n_measured = 0

    # mIoU 集計用
    sum_miou_b0 = 0.0
    sum_miou_b1 = 0.0
    sum_miou_sel = 0.0
    n_iou_samples = 0
    missing_in_csv = 0

    for imgs, names in loader:
        imgs = imgs.to(device, non_blocking=True)

        # 測定サンプルが規定数を超えたら終了
        if n_measured >= warmup + iters:
            break

        if device.type == "cuda":
            torch.cuda.synchronize()
            starter.record()

        # B0 推論 + セレクタ決定 + 必要に応じて B1
        with amp_ctx:
            # B0 推論
            out_b0 = model_b0(pixel_values=imgs)
            logits_b0 = out_b0.logits  # (B, C, H, W)

            # B0 ログits から特徴量を抽出 → セレクタ
            feats = extract_features_from_logits(
                logits_b0, num_classes=NUM_CLASSES, device=device
            )
            logits_sel = selector(feats)  # (B, 2)
            actions = torch.argmax(logits_sel, dim=1)  # 0: use B0, 1: use B1
            use_b1_flags = actions == 1  # (B,)

            # B1 が必要なサンプルに対してのみ B1 を実行
            if use_b1_flags.any():
                out_b1 = model_b1(pixel_values=imgs[use_b1_flags])
                _ = out_b1.logits  # 出力は IoU 計算には使わない（CSV から拾う）

        if device.type == "cuda":
            ender.record()
            torch.cuda.synchronize()
            dt_ms = starter.elapsed_time(ender)
        else:
            dt_ms = 0.0  # CPU ならおまけ

        # warmup 期間を除いて統計に入れる
        if n_measured >= warmup:
            times_ms.append(float(dt_ms))
            n_b1 += int(use_b1_flags.sum().item())

            # mIoU を precomputed CSV から集計
            if metrics is not None:
                for j, name in enumerate(names):
                    patch = str(name)
                    info = metrics.get(patch)
                    if info is None:
                        missing_in_csv += 1
                        continue

                    miou_b0 = info["miou_b0"]
                    miou_b1 = info["miou_b1"]

                    sum_miou_b0 += miou_b0
                    sum_miou_b1 += miou_b1

                    if bool(use_b1_flags[j].item()):
                        sum_miou_sel += miou_b1
                    else:
                        sum_miou_sel += miou_b0

                    n_iou_samples += 1

        n_measured += 1

    if len(times_ms) == 0:
        raise RuntimeError(
            f"no latency samples collected (dataset too small? warmup={warmup}, iters={iters})"
        )

    times_arr = np.array(times_ms, dtype=np.float32)
    mean_ms = float(times_arr.mean())
    p50_ms = float(np.percentile(times_arr, 50))
    p90_ms = float(np.percentile(times_arr, 90))

    print("\n=== selector pipeline latency (Jetson) ===")
    print(f"  device        : {device}")
    print(f"  split         : {split}")
    print(f"  batch_size    : {batch_size}")
    print(f"  fp16          : {fp16}")
    print(f"  warmup        : {warmup}")
    print(f"  iters         : {iters}")
    print(f"  #samples used : {len(times_ms)}")
    print(f"  #B1 calls     : {n_b1}  ({n_b1 / len(times_ms):.3f} per sample)")
    print(f"  mean latency  : {mean_ms:.3f} ms")
    print(f"  p50 latency   : {p50_ms:.3f} ms")
    print(f"  p90 latency   : {p90_ms:.3f} ms")

    if metrics is not None:
        if n_iou_samples == 0:
            print("\n[IoU] no samples matched in precomputed CSV -> IoU stats skipped")
        else:
            mean_miou_b0 = sum_miou_b0 / n_iou_samples
            mean_miou_b1 = sum_miou_b1 / n_iou_samples
            mean_miou_sel = sum_miou_sel / n_iou_samples
            print(
                f"\n=== selector mIoU (from precomputed CSV, split={split}, samples={n_iou_samples}) ==="
            )
            print(f"  always B0     : mIoU = {mean_miou_b0:.4f}")
            print(f"  always B1     : mIoU = {mean_miou_b1:.4f}")
            print(f"  RL selector   : mIoU = {mean_miou_sel:.4f}")
            if missing_in_csv > 0:
                print(f"  (ignored patches missing in CSV: {missing_in_csv})")


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
    p.add_argument(
        "--precomputed_csv",
        type=str,
        default=None,
        help=(
            "事前計算済みの mIoU / value CSV のパス "
            "(例: precomputed_values_rescuenet_b0_b1_fp16.csv)。"
            "指定した場合、レイテンシとあわせて mIoU も報告する。"
        ),
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
    print(f"csv           = {args.precomputed_csv}")

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
        precomputed_csv=Path(args.precomputed_csv) if args.precomputed_csv else None,
    )
    t1 = time.time()
    print(f"\n[done] total elapsed wall time = {t1 - t0:.1f} sec")


if __name__ == "__main__":
    main()
