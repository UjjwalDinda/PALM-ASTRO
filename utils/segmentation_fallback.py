
import numpy as np
import cv2
from skimage.filters import frangi
from skimage.morphology import skeletonize, remove_small_objects
from scipy.ndimage import gaussian_filter
import numpy as np
import cv2

def visualize_mask(mask):
    """
    Convert mask (0=bg, 1=life, 2=head, 3=heart) to color overlay.
    """
    h, w = mask.shape
    color = np.zeros((h, w, 3), dtype=np.uint8)

    # Assign bright colors
    color[mask == 1] = (0, 255, 255)   # Life line – Yellow
    color[mask == 2] = (255, 0, 255)   # Head line – Pink
    color[mask == 3] = (0, 255, 0)     # Heart line – Green

    return color




# 1. Improved Frangi Vessel Filter for Palm Lines

def _frangi_enhance(gray):
    """
    Enhance palm lines using Frangi filter (new scikit-image API).
    """
    img = gray.astype(np.float32)
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)

    # New parameter names for scikit-image 0.20+
    fr = frangi(
        img,
        scale_range=(1, 4),
        scale_step=1,
        beta=0.5,      # replaces beta1
        gamma=15       # replaces beta2
    )

    fr = (fr - fr.min()) / (fr.max() - fr.min() + 1e-8)
    return (fr * 255).astype(np.uint8)



# 2. Adaptive Threshold + Morphology

def _binarize_lines(enhanced):
    """
    Convert enhanced image to clean binary mask of lines.
    """
    # Smooth noise
    blur = cv2.GaussianBlur(enhanced, (5, 5), 0)

    # Adaptive threshold (robust)
    th = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        7
    )

    # Remove small noise
    kernel = np.ones((3, 3), np.uint8)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=1)

    return th



# 3. Skeletonize to Extract Line Structure

def _skeletonize_mask(mask):
    """
    Turn thick lines into 1-pixel skeletons for better feature extraction.
    """
    sk = skeletonize(mask > 0)
    sk = sk.astype(np.uint8) * 255
    return sk


# 4. Split into life / head / heart by geometry

def _split_lines(mask):
    """
    Heuristic separation:
    - Life line: large curve on thumb-side (left)
    - Head line: horizontal central line
    - Heart line: upper horizontal line
    """

    H, W = mask.shape
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return np.zeros_like(mask)

    # Final categorical mask (0=bg, 1=life, 2=head, 3=heart)
    out = np.zeros_like(mask)

    # Heuristic zones (works well for fallback mode)
    mid_y = H // 2
    upper_y = H // 3
    left_x = W // 3

    for y, x in zip(ys, xs):
        if x < left_x:
            out[y, x] = 1  # life line (curves on thumb side)
        elif y < upper_y:
            out[y, x] = 3  # heart line (top)
        else:
            out[y, x] = 2  # head line (middle)

    return out


# 5. MAIN API: Fallback Palm Segmentation

def fallback_segment_palm(img_pil):
    """
    Input:
        PIL.Image (RGB)
    Output:
        mask (H,W) with values:
            0 = background
            1 = life line
            2 = head line
            3 = heart line
    """

    # Convert to grayscale
    img = np.array(img_pil)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Step 1: Frangi enhancement
    enhanced = _frangi_enhance(gray)

    # Step 2: threshold lines
    bin_mask = _binarize_lines(enhanced)

    # Step 3: skeletonize for thin lines
    skel = _skeletonize_mask(bin_mask)

    # Step 4: split into three line categories
    final_mask = _split_lines(skel)

    return final_mask
