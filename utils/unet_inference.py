


import torch
from PIL import Image
import numpy as np
from torchvision import transforms

# try to import UNet
try:
    from models.unet import UNetSmall
except Exception:
    UNetSmall = None

def load_unet(checkpoint_path: str, device=None, n_classes: int = 4):
    device = device or torch.device("cpu")
    if UNetSmall is None:
        return None
    model = UNetSmall(n_classes=n_classes)
    ck = torch.load(checkpoint_path, map_location=device)
    state = None
    if isinstance(ck, dict):
        # some checkpoints contain epoch/model_state/mean_iou
        if "model_state" in ck:
            state = ck["model_state"]
        elif "state_dict" in ck:
            state = ck["state_dict"]
        elif "model" in ck:
            state = ck["model"]
    if state is None:
        state = ck
    # try to load with strict=False
    try:
        model.load_state_dict(state, strict=False)
    except Exception:
        # last attempt: try keys with "enc"/"dec" vs "down"/"conv"
        try:
            new_state = {}
            for k, v in state.items():
                nk = k.replace("enc", "down").replace("dec", "conv").replace("middle", "middle")
                new_state[nk] = v
            model.load_state_dict(new_state, strict=False)
        except Exception:
            pass
    model.to(device)
    model.eval()
    return model

def predict_mask_from_pil(model, pil_img: Image.Image, device=None, img_size=(256,256)):
    device = device or torch.device("cpu")
    tf = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
    ])
    x = tf(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(x)
        probs = torch.softmax(out, dim=1)
        mask = probs.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
    # upsample to original
    mask_img = Image.fromarray(mask)
    mask_img = mask_img.resize(pil_img.size, resample=Image.NEAREST)
    return np.array(mask_img, dtype=np.uint8)
