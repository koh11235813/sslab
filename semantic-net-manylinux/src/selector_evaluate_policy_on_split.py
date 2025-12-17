#!/usr/bin/env python3
"""
selector_evaluate_policy_on_split.py

学習済みセレクタ (selector_lambda_xxxx.pt) を用いて、
train / val / test の各 split で

  - selector policy
  - baseline: always B0
  - baseline: always B1

の Value / Cost / B1 使用率 をまとめて評価するスクリプト。

前提:
  - precomputed_values_rescuenet_b0_b1.csv は
      (split, patch) ごとの value_b0 / value_b1 を持つ
  - train_model_selecter.py で
      - features の計算
      - evaluate_policy_on_split
      - SelectorMLP
      が定義されている
"""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

import train_model_selector as tms


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--csv",
        type=str,
        required=True,
        help="precompute_values.py で生成した CSV パス",
    )
    p.add_argument(
        "--data_root",
        type=str,
        default="src/dataset",
        help="RescueNetPatches の root_dir",
    )
    p.add_argument(
        "--ckpt_b0",
        type=str,
        default="checkpoints_segformer_b0/best_segformer_b0.pt",
        help="SegFormer-B0 の学習済み checkpoint",
    )
    p.add_argument(
        "--selector_ckpt",
        type=str,
        default="",
        help="セレクタ MLP の checkpoint パス "
        "(未指定なら checkpoints_selector/selector_lambda_{lambda:.4f}.pt を使う)",
    )
    p.add_argument(
        "--lambda",
        "--lambda_",
        dest="lambda_",
        type=float,
        default=0.01,
        help="トレードオフ係数 λ",
    )
    p.add_argument(
        "--cost_b0",
        type=float,
        default=37.45,
        help="B0 のレイテンシ [ms/枚]",
    )
    p.add_argument(
        "--cost_b1",
        type=float,
        default=60.07,
        help="B1 のレイテンシ [ms/枚]",
    )
    p.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="セレクタ推論時の batch size",
    )
    p.add_argument(
        "--feat_batch_size",
        type=int,
        default=32,
        help="特徴量抽出 (B0 forward) の batch size",
    )
    p.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="特徴量抽出時 DataLoader の num_workers",
    )
    p.add_argument(
        "--hidden_dim",
        type=int,
        default=64,
        help="SelectorMLP の hidden dim (学習時と合わせる)",
    )
    p.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="cuda / cpu",
    )
    return p.parse_args()


def main():
    args = parse_args()

    device = torch.device(args.device)
    data_root = Path(args.data_root)
    csv_path = Path(args.csv)
    ckpt_b0_path = Path(args.ckpt_b0)

    # selector ckpt
    if args.selector_ckpt:
        selector_ckpt = Path(args.selector_ckpt)
    else:
        selector_ckpt = (
            Path("checkpoints_selector") / f"selector_lambda_{args.lambda_:.4f}.pt"
        )

    print(f"device        = {device}")
    print(f"data_root     = {data_root}")
    print(f"csv           = {csv_path}")
    print(f"lambda        = {args.lambda_}")
    print(f"cost_b0       = {args.cost_b0:.3f} ms")
    print(f"cost_b1       = {args.cost_b1:.3f} ms")
    print(f"B0 ckpt       = {ckpt_b0_path}")
    print(f"selector ckpt = {selector_ckpt}")

    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)
    if not ckpt_b0_path.is_file():
        raise FileNotFoundError(ckpt_b0_path)
    if not selector_ckpt.is_file():
        raise FileNotFoundError(selector_ckpt)

    # value テーブル読み込み
    value_table = tms.load_value_table(csv_path)

    # SegFormer-B0 を構築 & checkpoint 読み込み
    print("\n[eval] building SegFormer-B0...")
    model_b0 = tms.build_b0_model(device)
    print("[eval] loading B0 checkpoint...")
    tms.load_checkpoint(model_b0, ckpt_b0_path, device)
    model_b0.eval()

    # 各 split の特徴量を計算
    feats = {}
    for split in ("train", "val", "test"):
        print(f"\n[eval] computing features for split={split} ...")
        feats_split, _labels_dummy = tms.compute_features_and_labels_for_split(
            split=split,
            data_root=data_root,
            value_table=value_table,
            model_b0=model_b0,
            lambda_=args.lambda_,
            cost_b0=args.cost_b0,
            cost_b1=args.cost_b1,
            device=device,
            batch_size=args.feat_batch_size,
            num_workers=args.num_workers,
        )
        feats[split] = feats_split  # (N, F)

    # セレクタ MLP を構築 & checkpoint 読み込み
    in_dim = feats["train"].shape[1]
    selector = tms.SelectorMLP(in_dim=in_dim, hidden_dim=args.hidden_dim)
    selector.load_state_dict(torch.load(selector_ckpt, map_location=device))
    selector.to(device)
    selector.eval()

    # 各 split で selector の Value/Cost を評価
    for split in ("train", "val", "test"):
        print(f"\n========== EVALUATE split={split} ==========")
        feat_split = feats[split]

        # selector で予測
        all_logits = []
        with torch.no_grad():
            for i in range(0, feat_split.size(0), args.batch_size):
                xb = feat_split[i : i + args.batch_size].to(device)
                lb = selector(xb)
                all_logits.append(lb.cpu())
        all_logits = torch.cat(all_logits, dim=0)
        pred_actions = all_logits.argmax(dim=1)  # (N,)

        # Value / Cost / frac_B1 を表示
        tms.evaluate_policy_on_split(
            split=split,
            feats=feat_split,
            labels_pred=pred_actions,
            value_table=value_table,
            data_root=data_root,
            lambda_=args.lambda_,
            cost_b0=args.cost_b0,
            cost_b1=args.cost_b1,
        )


if __name__ == "__main__":
    main()
