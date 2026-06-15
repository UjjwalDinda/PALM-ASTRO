import os, joblib, numpy as np
from glob import glob
from PIL import Image
from src.features.feature_extraction import extract_features_from_mask
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import classification_report

DATA_DIR = "data"
IMG_DIR = os.path.join(DATA_DIR,"images")
MASK_DIR = os.path.join(DATA_DIR,"masks")

def load_pairs():
    imgs = sorted(glob(os.path.join(IMG_DIR,"*.*")))
    pairs=[]
    for img in imgs:
        base = os.path.splitext(os.path.basename(img))[0]
        mask = os.path.join(MASK_DIR, base + ".png")
        if os.path.exists(mask):
            pairs.append((img, mask))
    return pairs

def main():
    pairs = load_pairs()
    if len(pairs)==0:
        print("No pairs found in data/")
        return
    X=[]; y=[]
    last_feats = None
    for i, (img_path, mask_path) in enumerate(pairs):
        mask = np.array(Image.open(mask_path).convert("L").resize((256,256)))
        feats = extract_features_from_mask(mask)
        last_feats = feats
        # synthetic label using dominant_line
        label = feats.get("dominant_line","balanced")
        X.append([feats[k] for k in sorted(feats.keys())])
        y.append(label)
    X = np.array(X); y = np.array(y)
    strat = y if len(np.unique(y))>1 else None
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=strat, random_state=42)
    clf = xgb.XGBClassifier(n_estimators=200, use_label_encoder=False, eval_metric='mlogloss')
    clf.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=10, verbose=False)
    preds = clf.predict(X_test)
    print(classification_report(y_test, preds))
    os.makedirs("models", exist_ok=True)
    joblib.dump(clf, "models/xgb_clf.joblib")
    joblib.dump(sorted(last_feats.keys()), "models/feature_keys.joblib")
    print("Saved classifier and keys to models/")

if __name__=="__main__":
    main()
