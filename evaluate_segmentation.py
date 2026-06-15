

import json
from pathlib import Path
from PIL import Image
import numpy as np
import torch

from utils.unet_loader import load_unet, predict_mask_from_pil


# Directories

ROOT = Path(__file__).parent
DATA_IMAGES = ROOT / "data/seg/val/images"
DATA_MASKS = ROOT / "data/seg/val/masks"

MODELS = ROOT / "models"
RESULTS = ROOT / "results"
METRICS = RESULTS / "metrics"
METRICS.mkdir(exist_ok=True)

UNET_PATH = MODELS / "unet_best.pth"


# IoU + Dice calculator

def compute_iou_dice(pred, gt, cls_ids=(1,2,3)):
    out = {}
    for c in cls_ids:
        p = (pred == c).astype(np.uint8)
        t = (gt == c).astype(np.uint8)

        inter = (p & t).sum()
        union = (p | t).sum()
        iou = float(inter) / union if union > 0 else (1.0 if p.sum() == 0 and t.sum() == 0 else 0.0)

        denom = p.sum() + t.sum()
        dice = float(2 * inter) / denom if denom > 0 else (1.0 if p.sum() == 0 and t.sum() == 0 else 0.0)

        out[c] = {"iou": iou, "dice": dice}
    return out


# Load model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_unet(str(UNET_PATH), device=device, n_classes=4)
model.eval()


# Evaluate

results_per_class = {1: [], 2: [], 3: []}

image_files = sorted(DATA_IMAGES.glob("*.jpg")) + sorted(DATA_IMAGES.glob("*.png"))

print(f"[INFO] Found {len(image_files)} validation images")

for img_path in image_files:
    name = img_path.stem
    mask_path = DATA_MASKS / f"{name}.png"

    if not mask_path.exists():
        print(f"[WARN] GT mask missing for {name}, skipping")
        continue

    img = Image.open(img_path).convert("RGB")
    gt = np.array(Image.open(mask_path))

    pred = predict_mask_from_pil(model, img, device=device)

    metrics = compute_iou_dice(pred, gt)

    for cid, vals in metrics.items():
        results_per_class[cid].append(vals)


# Average metrics

final = {"per_class": {}, "mean_iou": 0.0, "mean_dice": 0.0}
ious = []
dices = []

for cid in (1,2,3):
    cls_vals = results_per_class[cid]
    if len(cls_vals) == 0:
        continue

    cls_iou = sum(v["iou"] for v in cls_vals) / len(cls_vals)
    cls_dice = sum(v["dice"] for v in cls_vals) / len(cls_vals)

    final["per_class"][str(cid)] = {
        "iou": cls_iou,
        "dice": cls_dice,
        "samples": len(cls_vals),
    }

    ious.append(cls_iou)
    dices.append(cls_dice)

final["mean_iou"] = sum(ious) / len(ious)
final["mean_dice"] = sum(dices) / len(dices)


# Save JSON

output_path = METRICS / "segmentation_metrics.json"
output_path.write_text(json.dumps(final, indent=2))

print(f"[OK] Saved segmentation metrics → {output_path}")
