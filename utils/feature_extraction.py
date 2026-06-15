

from typing import Dict
import numpy as np
from skimage.morphology import skeletonize
from scipy.ndimage import convolve
from skimage.measure import label, regionprops
import math


def _skeletonize_bin(bin_mask: np.ndarray) -> np.ndarray:
    """Return boolean skeleton of binary mask."""
    if bin_mask.sum() == 0:
        return np.zeros_like(bin_mask, dtype=bool)
    sk = skeletonize(bin_mask > 0).astype(bool)
    return sk


def _skeleton_length(skel: np.ndarray) -> int:
    """Return number of True pixels in skeleton (approx length in px)."""
    return int(skel.sum())


def _endpoints_and_junctions(skel: np.ndarray) -> (int, int):
    """
    Count endpoints (degree ==1) and junctions (degree >=3)
    using 3x3 neighbor sum convolution.
    """
    if skel.sum() == 0:
        return 0, 0
    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]], dtype=int)
    neigh = convolve(skel.astype(int), kernel, mode='constant', cval=0)
    endpoints = int(((skel == 1) & (neigh == 1)).sum())
    junctions = int(((skel == 1) & (neigh >= 3)).sum())
    return endpoints, junctions


def _compute_curvature(skel: np.ndarray) -> float:
    """
    Estimate curvature for a skeleton component.
    Approach:
      - get ordered point sequences by following connected pixels (simple greedy)
      - compute angles between successive segments, sum absolute angle change normalized by length
    Returns small floats; 0 for straight line.
    """
    coords = np.argwhere(skel)
    if len(coords) < 3:
        return 0.0

    # attempt to get an ordered path: greedily walk from one endpoint to next
    endpoints, junctions = _endpoints_and_junctions(skel)
    # find any endpoint to start, else pick a pixel
    start = None
    if endpoints > 0:
        kernel = np.array([[1, 1, 1],
                           [1, 0, 1],
                           [1, 1, 1]], dtype=int)
        neigh = convolve(skel.astype(int), kernel, mode='constant', cval=0)
        ep_coords = np.argwhere((skel == 1) & (neigh == 1))
        if len(ep_coords) > 0:
            start = tuple(ep_coords[0])
    if start is None:
        start = tuple(coords[0])

    # walk
    h, w = skel.shape
    visited = set()
    path = [start]
    visited.add(start)
    cur = start
    # 8-neighbor offsets
    nbrs = [(-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1)]
    while True:
        found = False
        cy, cx = cur
        for dy, dx in nbrs:
            ny, nx = cy + dy, cx + dx
            if 0 <= ny < h and 0 <= nx < w and skel[ny, nx]:
                if (ny, nx) not in visited:
                    visited.add((ny, nx))
                    path.append((ny, nx))
                    cur = (ny, nx)
                    found = True
                    break
        if not found:
            break
        if len(path) > 20000:
            break

    if len(path) < 3:
        return 0.0

    # compute angle changes
    angles = []
    for i in range(1, len(path) - 1):
        y0, x0 = path[i - 1]
        y1, x1 = path[i]
        y2, x2 = path[i + 1]
        v1 = (x1 - x0, y1 - y0)
        v2 = (x2 - x1, y2 - y1)
        # compute angle between v1 and v2
        dot = v1[0] * v2[0] + v1[1] * v2[1]
        n1 = math.hypot(v1[0], v1[1]) + 1e-9
        n2 = math.hypot(v2[0], v2[1]) + 1e-9
        cosang = max(-1.0, min(1.0, dot / (n1 * n2)))
        ang = math.acos(cosang)
        angles.append(abs(ang))

    if len(angles) == 0:
        return 0.0
    # curvature = average absolute angle change per step
    mean_ang = float(sum(angles) / len(angles))
    # normalize by pi so values roughly in 0..1.5 (typical lines small)
    return mean_ang / math.pi


def _bbox_ratio(points: np.ndarray) -> float:
    """
    points: Nx2 array of (x, y) or (col, row) depending on caller.
    We'll accept coords as (row, col) and compute bbox area ratio.
    """
    if len(points) == 0:
        return 0.0
    ys = points[:, 0]
    xs = points[:, 1]
    miny, maxy = int(ys.min()), int(ys.max())
    minx, maxx = int(xs.min()), int(xs.max())
    w = maxx - minx + 1
    h = maxy - miny + 1
    area = w * h
    if area <= 0:
        return 0.0
    return float(len(points)) / (area + 1e-9)


def extract_features_from_mask(mask: np.ndarray) -> Dict[str, float]:
    """
    mask: 2D numpy array with values {0,1,2,3}
    returns: dict with keys for each class (life/head/heart)
    """
    if mask is None:
        raise ValueError("mask is None")

    H, W = mask.shape
    diag = math.hypot(H, W) + 1e-9

    features = {}
    for cls_name, cls_id in [("life", 1), ("head", 2), ("heart", 3)]:
        bin_mask = (mask == cls_id).astype(np.uint8)
        pix_count = int(bin_mask.sum())
        features[f"{cls_name}_count"] = pix_count

        # skeleton length
        sk = _skeletonize_bin(bin_mask)
        sk_len = _skeleton_length(sk)
        # normalized length = pixels / diagonal
        features[f"{cls_name}_length"] = float(sk_len) / diag

        # curvature
        features[f"{cls_name}_curve"] = float(_compute_curvature(sk))

        # endpoints and intersections
        ep, junctions = _endpoints_and_junctions(sk)
        features[f"{cls_name}_endpoints"] = int(ep)
        features[f"{cls_name}_intersections"] = int(junctions)

        # bbox ratio
        coords = np.argwhere(bin_mask > 0)
        features[f"{cls_name}_bbox_ratio"] = float(_bbox_ratio(coords)) if coords.size else 0.0

    # Add a small global summary (optional but useful)
    features["mask_area"] = int((mask > 0).sum())
    features["h"] = int(H)
    features["w"] = int(W)

    return features
