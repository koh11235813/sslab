from collections import Counter

import numpy as np
import torch
from Dataset import RescueNetPatches

ds = RescueNetPatches(
    "/home/kinoko/development/github/sslab/dataset/RescueNet_patches",
    split="val",
)
cnt = Counter()

for _, mask in ds:
    vals, freqs = torch.unique(mask, return_counts=True)
    for v, f in zip(vals.tolist(), freqs.tolist()):
        cnt[v] += f

print(cnt)
