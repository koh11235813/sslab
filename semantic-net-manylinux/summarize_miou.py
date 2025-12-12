#!/usr/bin/env python3
import argparse
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True)
    args = parser.parse_args()

    csv_path = Path(args.csv)
    df = pd.read_csv(csv_path)

    print(f"csv = {csv_path}")
    for split in ["train", "val", "test"]:
        sub = df[df["split"] == split]
        if len(sub) == 0:
            continue
        miou_b0 = sub["miou_b0"].mean()
        miou_b1 = sub["miou_b1"].mean()
        print(f"\n[{split}]")
        print(f"  mIoU_B0(fp16) = {miou_b0:.4f}")
        print(f"  mIoU_B1(fp16) = {miou_b1:.4f}")


if __name__ == "__main__":
    main()
