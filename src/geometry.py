"""Coordinates are pixel centers on the resized evaluation images."""
import cv2
import numpy as np


def corners(size):
    w, h = size
    return np.array([(0, 0), (w-1, 0), (w-1, h-1), (0, h-1)], np.float64)


def project(homography, points):
    points = np.asarray(points).reshape(-1, 2)
    homogeneous = np.column_stack((points, np.ones(len(points)))) @ homography.T
    with np.errstate(divide="ignore", invalid="ignore"):
        return homogeneous[:, :2] / homogeneous[:, 2:]


def occupancy(points, size):
    cells = np.clip(np.floor(points * 4 / size).astype(int), 0, 3)
    return len(np.unique(cells, axis=0)) / 16


def assess(src, dst, src_size, dst_size, truth, ransac_px):
    result = dict(matches=len(src), inliers=0, inlier_ratio=0., coverage=0.,
                  reprojection=None, reliability=0., corner_error=None, homography=None)
    if len(src) < 4:
        return result
    h, mask = cv2.findHomography(src, dst, cv2.USAC_MAGSAC, ransac_px, maxIters=3000, confidence=.999)
    if h is None or mask is None or not np.isfinite(h).all():
        return result
    good = mask.reshape(-1).astype(bool)
    if good.sum() < 4:
        return result
    projected = project(h, src[good])
    pred_corners = project(h, corners(src_size))
    gt_corners = project(truth, corners(src_size))
    if not all(np.isfinite(x).all() for x in (projected, pred_corners, gt_corners)):
        return result
    err = float(np.median(np.linalg.norm(projected - dst[good], axis=1)))
    coverage = min(occupancy(src[good], src_size), occupancy(dst[good], dst_size))
    ratio = float(good.mean())
    score = ratio * coverage * (1 / (1 + err / ransac_px))
    result.update(inliers=int(good.sum()), inlier_ratio=ratio, coverage=coverage,
                  reprojection=err, reliability=score,
                  corner_error=float(np.linalg.norm(pred_corners - gt_corners, axis=1).mean()),
                  homography=h.tolist())
    return result


def resize_iso(image, max_side):
    h, w = image.shape[:2]
    scale = min(1., max_side / max(h, w))
    nw, nh = round(w * scale), round(h * scale)
    # cv2.resize uses a half-pixel translation as well as scale.
    sx, sy = nw/w, nh/h
    transform = np.array([[sx, 0, (sx-1)/2], [0, sy, (sy-1)/2], [0, 0, 1]])
    return cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA), transform
