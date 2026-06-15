import os, random, numpy as np
from pathlib import Path
from tqdm import tqdm
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from PIL import Image

from src.segmentation.models import build_unet

class PalmMaskDataset(Dataset):
    def __init__(self, images_dir, masks_dir, size=(256,256)):
        self.images = sorted(Path(images_dir).glob("*.*"))
        self.masks = sorted(Path(masks_dir).glob("*.*"))
        assert len(self.images) == len(self.masks), "images/masks mismatch"
        self.size = size

    def __len__(self): return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert("RGB").resize(self.size)
        mask = Image.open(self.masks[idx]).convert("L").resize(self.size)
        img = np.array(img).astype('float32')/255.0
        img = np.transpose(img,(2,0,1))
        mask = np.array(mask).astype('int64')
        import torch
        return torch.tensor(img, dtype=torch.float32), torch.tensor(mask, dtype=torch.long)

def iou_score(pred, targ, n_classes=4):
    pred = np.array(pred)
    targ = np.array(targ)
    ious=[]
    for c in range(n_classes):
        inter = ((pred==c) & (targ==c)).sum()
        union = ((pred==c)|(targ==c)).sum()
        ious.append(inter/(union+1e-9) if union>0 else np.nan)
    return np.nanmean(ious)

def train_segmentation(data_dir="data", epochs=5, batch_size=4, lr=1e-4):
    images_dir = os.path.join(data_dir,"images")
    masks_dir = os.path.join(data_dir,"masks")
    ds = PalmMaskDataset(images_dir, masks_dir)
    n = len(ds)
    if n == 0:
        print("No training samples found in data/images & data/masks")
        return
    idx = list(range(n)); random.shuffle(idx)
    split = int(n*0.8)
    train_idx = idx[:split]; val_idx = idx[split:]
    train_loader = DataLoader(Subset(ds, train_idx), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(Subset(ds, val_idx), batch_size=batch_size, shuffle=False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_unet().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    ce = nn.CrossEntropyLoss()
    best = 0.0
    for epoch in range(1, epochs+1):
        model.train()
        for imgs, masks in tqdm(train_loader, desc=f"Epoch {epoch}"):
            imgs = imgs.to(device)
            masks = masks.to(device)
            preds = model(imgs)
            loss = ce(preds, masks)
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        miou_vals=[]
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs=imgs.to(device); masks=masks.to(device)
                preds = model(imgs).argmax(dim=1).cpu().numpy()
                miou_vals.append(iou_score(preds, masks.cpu().numpy()))
        m = float(np.nanmean(miou_vals))
        print(f"Epoch {epoch} val_mIoU={m:.4f}")
        if m>best:
            best=m
            os.makedirs("models", exist_ok=True)
            torch.save(model.state_dict(), "models/unet_best.pth")
            print("Saved best:", "models/unet_best.pth")
    print("Done. Best mIoU:", best)

if __name__=="__main__":
    train_segmentation()
