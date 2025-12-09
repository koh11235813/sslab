#!/usr/bin/env python3
"""
CSV から B0/B1 の Value と Jetson レイテンシを使って、
lambda に応じた最適ポリシー（各パッチごとに B0/B1 どちらを選ぶか）を解析するスクリプト。

- 入力:
    --csv         : precompute_values.py で生成した CSV
    --lambda      : トレードオフ係数 λ
    --cost_b0/ms  : Jetson 上の B0 レイテンシ [ms/枚]
    --cost_b1/ms  : Jetson 上の B1 レイテンシ [ms/枚]

- 出力:
    - 全サンプルに対して
        r0 = value_b0 - λ * cost_b0
        r1 = value_b1 - λ * cost_b1
      を比較し、r1 > r0 なら B1 を選択、それ以外は B0 を選択とする「オラクル」ポリシーを仮定。
    - そのときの平均 Value / 平均 Cost / B1 を選んだ割合
    - baseline:
        - 常に B0 の場合
        - 常に B1 の場合
      の平均 Value / Cost も併せて表示する。

これで λ を変えながら Value–Cost のトレードオフを数値で確認できる。
"""

import argparse
import csv
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--csv",
        type=str,
        required=True,
        default="precomputed_values_rescuenet_b0_b1.csv",
        help="precompute_values.py で生成した CSV パス",
    )
    p.add_argument(
        "--lambda_",
        type=float,
        default=0.1,
        help="トレードオフ係数 λ (default: 0.1)",
    )
    p.add_argument(
        "--cost_b0",
        type=float,
        default=37.45,  # Jetson の測定結果に合わせて変えてよい
        help="B0 のレイテンシ [ms/枚] (default: 37.45)",
    )
    p.add_argument(
        "--cost_b1",
        type=float,
        default=60.07,
        help="B1 のレイテンシ [ms/枚] (default: 60.07)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    csv_path = Path(args.csv)

    print(f"csv       = {csv_path}")
    print(f"lambda    = {args.lambda_}")
    print(f"cost_b0   = {args.cost_b0:.3f} ms")
    print(f"cost_b1   = {args.cost_b1:.3f} ms")

    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)

    # 集計用
    n = 0

    # baseline: always B0, always B1
    sum_value_b0 = 0.0
    sum_value_b1 = 0.0

    # cost はモデルごとに一定なので平均すると元と同じになるが、
    # 一応サンプル数を掛けて持っておく
    sum_cost_b0 = 0.0
    sum_cost_b1 = 0.0

    # oracle policy (per-sample best for given λ)
    sum_value_oracle = 0.0
    sum_cost_oracle = 0.0
    count_b1_oracle = 0

    lam = args.lambda_
    C0 = args.cost_b0
    C1 = args.cost_b1

    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            n += 1
            v0 = float(row["value_b0"])
            v1 = float(row["value_b1"])

            # baseline
            sum_value_b0 += v0
            sum_value_b1 += v1
            sum_cost_b0 += C0
            sum_cost_b1 += C1

            # oracle policy: choose argmax of (V - λ C)
            r0 = v0 - lam * C0
            r1 = v1 - lam * C1

            if r1 > r0:
                # choose B1
                sum_value_oracle += v1
                sum_cost_oracle += C1
                count_b1_oracle += 1
            else:
                # choose B0
                sum_value_oracle += v0
                sum_cost_oracle += C0

    if n == 0:
        print("no rows in CSV")
        return

    # 平均値
    avg_value_b0 = sum_value_b0 / n
    avg_value_b1 = sum_value_b1 / n
    avg_cost_b0 = sum_cost_b0 / n
    avg_cost_b1 = sum_cost_b1 / n

    avg_value_oracle = sum_value_oracle / n
    avg_cost_oracle = sum_cost_oracle / n
    frac_b1_oracle = count_b1_oracle / n

    print("\n=== baseline: always B0 ===")
    print(f"  avg_value = {avg_value_b0:.4f}")
    print(f"  avg_cost  = {avg_cost_b0:.3f} ms")

    print("\n=== baseline: always B1 ===")
    print(f"  avg_value = {avg_value_b1:.4f}")
    print(f"  avg_cost  = {avg_cost_b1:.3f} ms")

    print("\n=== oracle policy (per-sample argmax of V - λC) ===")
    print(f"  avg_value = {avg_value_oracle:.4f}")
    print(f"  avg_cost  = {avg_cost_oracle:.3f} ms")
    print(f"  frac_B1   = {frac_b1_oracle * 100:.2f} %  (B1 を選択した割合)")
    print(f"  improvement over B0 (value) = {avg_value_oracle - avg_value_b0:+.4f}")
    print(f"  cost increase over B0       = {avg_cost_oracle - avg_cost_b0:+.3f} ms")


if __name__ == "__main__":
    main()
