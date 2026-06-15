

import argparse
import os
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader
import pandas as pd

from train_segmentation import PalmSegDataset, UNetSmall, compute_iou_and_dice

def evaluate(model_path, data_dir, img_size=256, batch_size=8, device=None, out_csv="results/eval_metrics.csv"):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model_ckpt = torch.load(model_path, map_location=device)
    model = UNetSmall(n_classes=4).to(device)
    model.load_state_dict(model_ckpt["model_state"])
    model.eval()

    ds = PalmSegDataset(str(Path(data_dir)/"images"), str(Path(data_dir)/"masks"), img_size=(img_size,img_size), augment=False)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2)

    all_rows = []
    with torch.no_grad():
        for imgs, masks in loader:
            imgs = imgs.to(device)
            masks = masks.to(device)
            preds = model(imgs)
            pred_labels = preds.argmax(dim=1)

            # compute per-sample metrics
            for b in range(pred_labels.shape[0]):
                pred_b = pred_labels[b:b+1]
                mask_b = masks[b:b+1]
                ious, dices = compute_iou_and_dice(pred_b, mask_b, num_classes=4)
                row = {
                    "iou_bg": ious[0], "iou_life": ious[1], "iou_head": ious[2], "iou_heart": ious[3],
                    "dice_bg": dices[0], "dice_life": dices[1], "dice_head": dices[2], "dice_heart": dices[3],
                }
                all_rows.append(row)

    df = pd.DataFrame(all_rows)
    # compute means
    summary = df.mean().to_dict()
    os.makedirs(Path(out_csv).parent, exist_ok=True)
    df.to_csv(out_csv.replace(".csv", "_per_sample.csv"), index=False)
    # save summary
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(out_csv, index=False)
    print(f"Saved evaluation CSV -> {out_csv}")
    print("Mean metrics:")
    print(summary)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/unet_best.pth")
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--out", default="results/eval_metrics.csv")
    args = parser.parse_args()
    evaluate(args.model, args.data_dir, img_size=args.img_size, batch_size=args.batch_size, out_csv=args.out)
