

import json
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from PIL import Image

from utils.feature_extraction import extract_features_from_mask
from utils.segmentation_fallback import fallback_segment_palm
from utils.unet_loader import load_unet, predict_mask_from_pil

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
)

# ---------------------------------------------------------
ROOT = Path(__file__).parent
DATA_CSV = ROOT / "data/classifier/val.csv"
MODELS = ROOT / "models"
RESULTS = ROOT / "results"
METRICS = RESULTS / "metrics"
METRICS.mkdir(exist_ok=True)

UNET_PATH = MODELS / "unet_best.pth"
RF_PATH = MODELS / "rf_model.joblib"
FEAT_KEYS_PATH = MODELS / "feature_keys.joblib"


# Load model(s)

rf = joblib.load(RF_PATH)
feature_keys = joblib.load(FEAT_KEYS_PATH)

device = "cuda" if torch.cuda.is_available() else "cpu"
unet = load_unet(str(UNET_PATH), device=device, n_classes=4)


# Load CSV

df = pd.read_csv(DATA_CSV)

y_true = []
y_pred = []

print(f"[INFO] Loaded {len(df)} evaluation samples")

for idx, row in df.iterrows():
    img_path = Path(row["image"])
    label = row["label"]

    if not img_path.exists():
        print(f"[WARN] Missing image → {img_path}")
        continue

    img = Image.open(img_path).convert("RGB")

    # segmentation
    try:
        mask = predict_mask_from_pil(unet, img, device=device)
    except:
        mask = fallback_segment_palm(img)

    # features
    feats = extract_features_from_mask(mask)

    vec = np.array([feats.get(k,0) for k in feature_keys]).reshape(1,-1)

    pred = rf.predict(vec)[0]

    y_true.append(label)
    y_pred.append(pred)


# Compute metrics

report = classification_report(y_true, y_pred, output_dict=True)
cm = confusion_matrix(y_true, y_pred).tolist()

final = {
    "classification_report": report,
    "confusion_matrix": cm,
    "feature_importance": list(zip(feature_keys, rf.feature_importances_.tolist())),
}


# Save JSON

output_path = METRICS / "classification_report.json"
output_path.write_text(json.dumps(final, indent=2))

print(f"[OK] Saved classifier metrics → {output_path}")
