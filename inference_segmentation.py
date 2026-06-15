

import argparse
import os
from pathlib import Path
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as T

from train_segmentation import UNetSmall

COLOR_MAP = {
    0: (0,0,0,0),       # background transparent / black
    1: (255,0,0,160),   # life - red-ish
    2: (0,255,0,160),   # head - green
    3: (0,0,255,160),   # heart - blue
}

def load_model(path, device):
    ckpt = torch.load(path, map_location=device)
    model = UNetSmall(n_classes=4).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model

def predict_mask(model, pil_img, img_size=256, device="cpu"):
    tf = T.Compose([T.Resize((img_size, img_size)), T.ToTensor()])
    inp = tf(pil_img).unsqueeze(0).to(device)  # [1,3,H,W]
    with torch.no_grad():
        logits = model(inp)
        pred = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)  # HxW
    # resize back to original size
    pred_pil = Image.fromarray(pred)
    pred_pil = pred_pil.resize(pil_img.size, resample=Image.NEAREST)
    return np.array(pred_pil)

def colorize_mask(mask):
    h,w = mask.shape
    canvas = Image.new("RGBA", (w,h))
    arr = np.zeros((h,w,4), dtype=np.uint8)
    for cls, col in COLOR_MAP.items():
        mask_c = (mask == cls)
        arr[mask_c] = col
    return Image.fromarray(arr, mode="RGBA")

def overlay_mask_on_img(img_pil, mask_pil, alpha=0.65):
    colored = mask_pil.convert("RGBA")
    base = img_pil.convert("RGBA")
    out = Image.alpha_composite(base, colored)
    return out

def run_inference(model_path, input_path, out_dir="results/inference", img_size=256, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_path, device)
    input_path = Path(input_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = []
    if input_path.is_dir():
        for f in sorted(input_path.glob("*.*")):
            if f.suffix.lower() in (".jpg",".jpeg",".png"):
                files.append(f)
    else:
        files = [input_path]

    for f in files:
        pil = Image.open(f).convert("RGB")
        mask = predict_mask(model, pil, img_size=img_size, device=device)
        # save mask as PNG (values 0..3)
        mask_pil = Image.fromarray(mask.astype(np.uint8), mode="L")
        mask_out = out_dir / (f.stem + "_mask.png")
        mask_pil.save(mask_out)

        # colored overlay and combined image
        colored = colorize_mask(mask)
        overlay = Image.alpha_composite(pil.convert("RGBA"), colored)
        overlay_out = out_dir / (f.stem + "_overlay.png")
        overlay.save(overlay_out)

        print("Saved:", mask_out, overlay_out)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/unet_best.pth")
    parser.add_argument("--input", required=True, help="Image file or directory")
    parser.add_argument("--out", default="results/inference")
    parser.add_argument("--img_size", type=int, default=256)
    args = parser.parse_args()
    run_inference(args.model, args.input, out_dir=args.out, img_size=args.img_size)
