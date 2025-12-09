#!/usr/bin/env python3
"""
B0/B1 の Value と Jetson レイテンシをもとに、
「どのパッチで B1 を使うべきか」を学習するセレクタを訓練するスクリプト。

- 入力:
    - precomputed_values_rescuenet_b0_b1.csv
        (precompute_values.py で作った CSV)
    - RescueNet_patches データセット
    - SegFormer-B0 の学習済み checkpoint
- 出力:
    - checkpoints_selector/selector_lambda_{lambda}.pt

特徴量:
    - B0 の予測ラベルマップからのクラス頻度 (11次元)
    - B0 の max softmax 確率の平均値 (1次元)
    → 合計 12 次元

ラベル:
    - r0 = V_b0 - λ C0
    - r1 = V_b1 - λ C1
    - r1 > r0 なら 1 (B1 を使うべき), それ以外 0
"""

import argparse
import csv
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset
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


def build_b0_model(device: torch.device) -> SegformerForSemanticSegmentation:
    ckpt_name = "nvidia/segformer-b0-finetuned-ade-512-512"
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


def load_checkpoint(model: nn.Module, ckpt_path: Path, device: torch.device) -> None:
    ckpt = torch.load(ckpt_path, map_location=device)
    if "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    else:
        state_dict = ckpt
    model.load_state_dict(state_dict)
    print(f"[selector] loaded B0 checkpoint from {ckpt_path}")


def load_value_table(
    csv_path: Path,
) -> Dict[Tuple[str, str], Tuple[float, float]]:
    """
    (split, patch_name) -> (V_b0, V_b1)
    """
    table: Dict[Tuple[str, str], Tuple[float, float]] = {}
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            split = row["split"]
            patch = row["patch"]
            v0 = float(row["value_b0"])
            v1 = float(row["value_b1"])
            table[(split, patch)] = (v0, v1)
    return table


class SelectorRawDataset(Dataset):
    """
    SegFormer-B0 用の入力画像と、
    (split, patch_name) に対応する V_b0, V_b1 を返す Dataset。
    """

    def __init__(
        self,
        data_root: Path,
        split: str,
        value_table: Dict[Tuple[str, str], Tuple[float, float]],
    ):
        self.base = RescueNetPatches(str(data_root), split=split)
        self.split = split
        self.value_table = value_table

        # 一応整合性チェック
        for p in self.base.img_paths:
            key = (self.split, p.name)
            if key not in self.value_table:
                raise KeyError(f"value_table に {key} が見つからない")

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, _ = self.base[idx]  # mask は不要
        patch_name = self.base.img_paths[idx].name
        v0, v1 = self.value_table[(self.split, patch_name)]
        return img, patch_name, v0, v1


@torch.no_grad()
def compute_features_and_labels_for_split(
    split: str,
    data_root: Path,
    value_table: Dict[Tuple[str, str], Tuple[float, float]],
    model_b0: SegformerForSemanticSegmentation,
    lambda_: float,
    cost_b0: float,
    cost_b1: float,
    device: torch.device,
    batch_size: int = 32,
    num_workers: int = 4,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    指定した split (train/val) について、
    - 特徴量: (N, 12)
    - ラベル:  (N,)  (0 or 1)
    を返す。
    """
    ds = SelectorRawDataset(data_root, split, value_table)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    all_feats = []
    all_labels = []

    C0 = cost_b0
    C1 = cost_b1

    pbar = tqdm(loader, desc=f"feat_extract[{split}]", total=len(loader))
    for imgs, patch_names, v0_batch, v1_batch in pbar:
        imgs = imgs.to(device, non_blocking=True)  # (B, 3, H, W)

        # SegFormer-B0 forward
        out = model_b0(pixel_values=imgs)
        logits = out.logits  # (B, C, h, w)
        probs = logits.softmax(dim=1)  # (B, C, h, w)
        preds = probs.argmax(dim=1)  # (B, h, w)

        B, _, H, W = logits.shape

        # 各クラスの頻度（preds ベース）
        # (B, C)
        class_freqs = []
        for c in range(NUM_CLASSES):
            freq_c = (preds == c).float().mean(dim=(1, 2))  # (B,)
            class_freqs.append(freq_c)
        class_freqs = torch.stack(class_freqs, dim=1)  # (B, C)

        # max softmax 確率の平均
        max_prob = probs.max(dim=1).values  # (B, h, w)
        mean_max_prob = max_prob.mean(dim=(1, 2))  # (B,)

        # 特徴量: [class_freqs, mean_max_prob]
        feats = torch.cat([class_freqs, mean_max_prob.unsqueeze(1)], dim=1)  # (B, 12)

        v0 = torch.tensor(v0_batch, dtype=torch.float32)
        v1 = torch.tensor(v1_batch, dtype=torch.float32)

        # 報酬
        r0 = v0 - lambda_ * C0
        r1 = v1 - lambda_ * C1

        labels = (r1 > r0).long()  # (B,)

        all_feats.append(feats.cpu())
        all_labels.append(labels.cpu())

    all_feats_tensor = torch.cat(all_feats, dim=0)
    all_labels_tensor = torch.cat(all_labels, dim=0)

    print(
        f"[{split}] features shape = {all_feats_tensor.shape}, "
        f"labels mean = {all_labels_tensor.float().mean().item():.4f}"
    )
    return all_feats_tensor, all_labels_tensor


class SelectorMLP(nn.Module):
    def __init__(self, in_dim: int = 12, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),  # logits for [B0, B1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def evaluate_policy_on_split(
    split: str,
    feats: torch.Tensor,
    labels_pred: torch.Tensor,
    value_table: Dict[Tuple[str, str], Tuple[float, float]],
    data_root: Path,
    lambda_: float,
    cost_b0: float,
    cost_b1: float,
) -> None:
    """
    学習済みポリシー (labels_pred: 0/1) がどの程度 Value/Cost を出せているかを簡単に評価する。
    split 内のサンプル順は RescueNetPatches と同じである前提。
    """
    ds = RescueNetPatches(str(data_root), split=split)
    assert len(ds) == feats.shape[0]

    C0 = cost_b0
    C1 = cost_b1

    sum_v_sel = 0.0
    sum_c_sel = 0.0

    sum_v_b0 = 0.0
    sum_c_b0 = 0.0

    sum_v_b1 = 0.0
    sum_c_b1 = 0.0

    for idx in range(len(ds)):
        patch_name = ds.img_paths[idx].name
        v0, v1 = value_table[(split, patch_name)]

        a = int(labels_pred[idx].item())  # 0 or 1

        if a == 1:
            sum_v_sel += v1
            sum_c_sel += C1
        else:
            sum_v_sel += v0
            sum_c_sel += C0

        sum_v_b0 += v0
        sum_c_b0 += C0

        sum_v_b1 += v1
        sum_c_b1 += C1

    n = len(ds)
    avg_v_sel = sum_v_sel / n
    avg_c_sel = sum_c_sel / n
    avg_v_b0 = sum_v_b0 / n
    avg_c_b0 = sum_c_b0 / n
    avg_v_b1 = sum_v_b1 / n
    avg_c_b1 = sum_c_b1 / n

    print(f"\n[{split}] === selector policy (learned) ===")
    print(f"  avg_value = {avg_v_sel:.4f}")
    print(f"  avg_cost  = {avg_c_sel:.3f} ms")

    print(f"\n[{split}] === baseline: always B0 ===")
    print(f"  avg_value = {avg_v_b0:.4f}")
    print(f"  avg_cost  = {avg_c_b0:.3f} ms")

    print(f"\n[{split}] === baseline: always B1 ===")
    print(f"  avg_value = {avg_v_b1:.4f}")
    print(f"  avg_cost  = {avg_c_b1:.3f} ms")

    frac_b1 = labels_pred.float().mean().item()
    print(f"\n[{split}] selector: frac_B1 = {frac_b1 * 100:.2f} %")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--csv",
        type=str,
        required=True,
        default="precomputed_values_rescuenet_b0_b1.csv",
        help="precompute_values.py で生成した CSV",
    )
    p.add_argument(
        "--data_root",
        type=str,
        default="src/dataset",
    )
    p.add_argument(
        "--ckpt_b0",
        type=str,
        default="checkpoints_segformer_b0/best_segformer_b0.pt",
    )
    p.add_argument(
        "--lambda_",
        type=float,
        default=0.01,
        help="トレードオフ係数 λ (default: 0.01)",
    )
    p.add_argument(
        "--cost_b0",
        type=float,
        default=37.45,
        help="B0 のレイテンシ [ms/枚] (default: 37.45)",
    )
    p.add_argument(
        "--cost_b1",
        type=float,
        default=60.07,
        help="B1 のレイテンシ [ms/枚] (default: 60.07)",
    )
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--feat_batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden_dim", type=int, default=64)
    p.add_argument(
        "--out_dir",
        type=str,
        default="checkpoints_selector",
    )
    p.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return p.parse_args()


def main():
    args = parse_args()

    device = torch.device(args.device)
    data_root = Path(args.data_root)
    csv_path = Path(args.csv)
    ckpt_b0_path = Path(args.ckpt_b0)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"device   = {device}")
    print(f"data_root= {data_root}")
    print(f"csv      = {csv_path}")
    print(f"lambda   = {args.lambda_}")
    print(f"cost_b0  = {args.cost_b0:.3f} ms")
    print(f"cost_b1  = {args.cost_b1:.3f} ms")

    value_table = load_value_table(csv_path)

    print("[selector] building SegFormer-B0...")
    model_b0 = build_b0_model(device)
    print("[selector] loading B0 checkpoint...")
    load_checkpoint(model_b0, ckpt_b0_path, device)

    # 特徴量 & ラベル前計算
    feats_train, labels_train = compute_features_and_labels_for_split(
        "train",
        data_root,
        value_table,
        model_b0,
        args.lambda_,
        args.cost_b0,
        args.cost_b1,
        device,
        batch_size=args.feat_batch_size,
        num_workers=args.num_workers,
    )
    feats_val, labels_val = compute_features_and_labels_for_split(
        "val",
        data_root,
        value_table,
        model_b0,
        args.lambda_,
        args.cost_b0,
        args.cost_b1,
        device,
        batch_size=args.feat_batch_size,
        num_workers=args.num_workers,
    )

    train_ds = TensorDataset(feats_train, labels_train)
    val_ds = TensorDataset(feats_val, labels_val)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    # モデル定義
    model = SelectorMLP(in_dim=feats_train.shape[1], hidden_dim=args.hidden_dim)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_reward = -1e9
    best_path = out_dir / f"selector_lambda_{args.lambda_:.4f}.pt"

    C0 = args.cost_b0
    C1 = args.cost_b1
    lam = args.lambda_

    for epoch in range(1, args.epochs + 1):
        model.train()
        sum_loss = 0.0
        sum_correct = 0
        n_train = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            sum_loss += loss.item() * x.size(0)
            preds = logits.argmax(dim=1)
            sum_correct += (preds == y).sum().item()
            n_train += x.size(0)

        train_loss = sum_loss / n_train
        train_acc = sum_correct / n_train

        # validation
        model.eval()
        sum_correct_val = 0
        n_val = 0

        # ついでに V - λC の平均も見る
        sum_reward_val = 0.0

        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)
                logits = model(x)
                preds = logits.argmax(dim=1)

                sum_correct_val += (preds == y).sum().item()
                n_val += x.size(0)

        val_acc = sum_correct_val / n_val

        # バリデーションの reward を CSV ベースで計算
        # （ラベル y ではなく、実際の V_b0, V_b1 と C0, C1 を用いる）
        with torch.no_grad():
            model.eval()
            # feats_val: (N_val, F)
            logits_all = []
            for i in range(0, feats_val.size(0), args.batch_size):
                xb = feats_val[i : i + args.batch_size].to(device)
                lb = model(xb)
                logits_all.append(lb.cpu())
            logits_all = torch.cat(logits_all, dim=0)
            pred_actions = logits_all.argmax(dim=1)  # (N_val,)

        # reward 計算
        ds_val = RescueNetPatches(str(data_root), split="val")
        assert len(ds_val) == pred_actions.size(0)
        for idx in range(len(ds_val)):
            patch_name = ds_val.img_paths[idx].name
            v0, v1 = value_table[("val", patch_name)]
            a = int(pred_actions[idx].item())
            if a == 1:
                r = v1 - lam * C1
            else:
                r = v0 - lam * C0
            sum_reward_val += r

        avg_reward_val = sum_reward_val / len(ds_val)

        print(
            f"[epoch {epoch:02d}] "
            f"train_loss={train_loss:.4f} "
            f"train_acc={train_acc:.4f} "
            f"val_acc={val_acc:.4f} "
            f"val_avg_reward={avg_reward_val:.4f}"
        )

        if avg_reward_val > best_val_reward:
            best_val_reward = avg_reward_val
            torch.save(model.state_dict(), best_path)
            print(f"  -> best updated, saved to {best_path}")

    print(f"\n[done] best_val_reward = {best_val_reward:.4f}")
    print(f"saved selector to {best_path}")

    # 最終モデルで train/val の Value/Cost をざっくり確認
    model.load_state_dict(torch.load(best_path, map_location=device))
    model.eval()

    with torch.no_grad():
        logits_train = []
        for i in range(0, feats_train.size(0), args.batch_size):
            xb = feats_train[i : i + args.batch_size].to(device)
            lb = model(xb)
            logits_train.append(lb.cpu())
        logits_train = torch.cat(logits_train, dim=0)
        pred_train = logits_train.argmax(dim=1)

        logits_val = []
        for i in range(0, feats_val.size(0), args.batch_size):
            xb = feats_val[i : i + args.batch_size].to(device)
            lb = model(xb)
            logits_val.append(lb.cpu())
        logits_val = torch.cat(logits_val, dim=0)
        pred_val = logits_val.argmax(dim=1)

    evaluate_policy_on_split(
        "train",
        feats_train,
        pred_train,
        value_table,
        data_root,
        lam,
        C0,
        C1,
    )
    evaluate_policy_on_split(
        "val",
        feats_val,
        pred_val,
        value_table,
        data_root,
        lam,
        C0,
        C1,
    )


if __name__ == "__main__":
    main()
