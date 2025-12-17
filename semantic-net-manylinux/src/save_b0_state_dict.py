# save_b0_state_dict.py（manylinux側で実行）
import torch
from transformers import SegformerForSemanticSegmentation

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

ckpt_name = "nvidia/segformer-b0-finetuned-ade-512-512"

model = SegformerForSemanticSegmentation.from_pretrained(
    ckpt_name,
    num_labels=NUM_CLASSES,
    id2label=ID2LABEL,
    label2id=LABEL2ID,
    ignore_mismatched_sizes=True,
)

ckpt = torch.load("checkpoints_segformer_b0/best_segformer_b0.pt", map_location="cpu")
if "model_state_dict" in ckpt:
    model.load_state_dict(ckpt["model_state_dict"])
else:
    model.load_state_dict(ckpt)

# 純粋な state_dict として書き出す
torch.save(model.state_dict(), "checkpoints_segformer_b0/b0_state_dict.pt")
print("saved checkpoints_segformer_b0/b0_state_dict.pt")
