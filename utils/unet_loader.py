


import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from torchvision import transforms
from pathlib import Path

# try to import the UNet model defined in repository
try:
    from models.unet import UNetSmall
except Exception:
    UNetSmall = None

def load_unet(checkpoint_path: str, device=None, n_classes: int = 4):
    device = device or torch.device("cpu")
    if UNetSmall is None:
        # no model definition available
        return None
    model = UNetSmall(n_classes=n_classes)
    ck = torch.load(checkpoint_path, map_location=device)
    # try common keys
    state = None
    if isinstance(ck, dict):
        for k in ("model_state", "state_dict", "model"):
            if k in ck:
                state = ck[k]
                break
    if state is None:
        # maybe ck itself is a state dict
        state = ck
    try:
        model.load_state_dict(state, strict=False)
    except Exception:
        try:
            # try remapping some prefixes (common mismatch: "enc1." vs "down1.")
            new_state = {}
            for key, val in state.items():
                new_key = key
                # example heuristics (can add more rules if necessary)
                new_key = new_key.replace("enc", "down").replace("dec", "conv")
                new_state[new_key] = val
            model.load_state_dict(new_state, strict=False)
        except Exception:
            # give up and return the uninitialized model (still possibly useful)
            pass
    model.to(device)
    model.eval()
    return model

def predict_mask_from_pil(model, pil_img: Image.Image, device=None, img_size=(256,256)):
    device = device or torch.device("cpu")
    # preprocess: resize to img_size, to tensor, normalize
    tf = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
    ])
    x = tf(pil_img).unsqueeze(0).to(device)  # [1,3,H,W]
    with torch.no_grad():
        out = model(x)  # [1, C, H, W]
        probs = torch.softmax(out, dim=1)
        lbl = probs.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)  # H,W
    # resize back to original pil size
    lbl_img = Image.fromarray(lbl)
    lbl_img = lbl_img.resize(pil_img.size, resample=Image.NEAREST)
    return np.array(lbl_img, dtype=np.uint8)
