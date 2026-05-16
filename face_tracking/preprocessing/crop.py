# coding: utf-8

"""Face-cropping helpers (vendored from LivePortrait).

These pure functions compute affine transformations / bounding boxes
from sparse facial landmark points. They are licence-compatible (MIT)
because LivePortrait itself releases ``src/utils/crop.py`` under MIT.
"""


import os.path as osp
from math import sin, cos, acos, degrees

import cv2
import numpy as np

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

DTYPE = np.float32
CV2_INTERP = cv2.INTER_LINEAR


def make_abs_path(fn: str) -> str:
    return osp.join(osp.dirname(osp.realpath(__file__)), fn)


def _transform_img(img, M, dsize, flags=CV2_INTERP, borderMode=None):
    if isinstance(dsize, (tuple, list)):
        _dsize = tuple(dsize)
    else:
        _dsize = (dsize, dsize)

    if borderMode is not None:
        return cv2.warpAffine(
            img, M[:2, :], dsize=_dsize, flags=flags,
            borderMode=borderMode, borderValue=(0, 0, 0),
        )
    return cv2.warpAffine(img, M[:2, :], dsize=_dsize, flags=flags)


def _transform_pts(pts, M):
    """``pts``: ``Nx2`` ndarray, ``M``: ``2x3`` or ``3x3`` matrix."""
    return pts @ M[:2, :2].T + M[:2, 2]


# ---------------------------------------------------------------------------
# 2-point (eye-center / lip-center) extractors for various landmark schemas
# ---------------------------------------------------------------------------

def parse_pt2_from_pt106(pt106, use_lip=True):
    pt_left_eye = np.mean(pt106[[33, 35, 40, 39]], axis=0)
    pt_right_eye = np.mean(pt106[[87, 89, 94, 93]], axis=0)
    if use_lip:
        pt_center_eye = (pt_left_eye + pt_right_eye) / 2
        pt_center_lip = (pt106[52] + pt106[61]) / 2
        return np.stack([pt_center_eye, pt_center_lip], axis=0)
    return np.stack([pt_left_eye, pt_right_eye], axis=0)


def parse_pt2_from_pt203(pt203, use_lip=True):
    pt_left_eye = np.mean(pt203[[0, 6, 12, 18]], axis=0)
    pt_right_eye = np.mean(pt203[[24, 30, 36, 42]], axis=0)
    if use_lip:
        pt_center_eye = (pt_left_eye + pt_right_eye) / 2
        pt_center_lip = (pt203[48] + pt203[66]) / 2
        return np.stack([pt_center_eye, pt_center_lip], axis=0)
    return np.stack([pt_left_eye, pt_right_eye], axis=0)


def parse_pt2_from_pt_x(pts, use_lip=True):
    if pts.shape[0] == 106:
        pt2 = parse_pt2_from_pt106(pts, use_lip=use_lip)
    elif pts.shape[0] == 203:
        pt2 = parse_pt2_from_pt203(pts, use_lip=use_lip)
    elif pts.shape[0] > 101:
        # Fallback: compute eye/lip centers from the leading prefix.
        pt2 = parse_pt2_from_pt106(pts[:106], use_lip=use_lip)
    else:
        raise ValueError(f"Unsupported landmark count: {pts.shape}")

    if not use_lip:
        v = pt2[1] - pt2[0]
        pt2[1, 0] = pt2[0, 0] - v[1]
        pt2[1, 1] = pt2[0, 1] + v[0]
    return pt2


# ---------------------------------------------------------------------------
# Bounding-box helpers
# ---------------------------------------------------------------------------

def parse_rect_from_landmark(
    pts, scale=1.5, need_square=True,
    vx_ratio=0, vy_ratio=0, use_deg_flag=False, **kwargs,
):
    pt2 = parse_pt2_from_pt_x(pts, use_lip=kwargs.get("use_lip", True))

    uy = pt2[1] - pt2[0]
    l = np.linalg.norm(uy)
    if l <= 1e-3:
        uy = np.array([0, 1], dtype=DTYPE)
    else:
        uy /= l
    ux = np.array((uy[1], -uy[0]), dtype=DTYPE)

    angle = acos(ux[0])
    if ux[1] < 0:
        angle = -angle

    M = np.array([ux, uy])

    center0 = np.mean(pts, axis=0)
    rpts = (pts - center0) @ M.T
    lt_pt = np.min(rpts, axis=0)
    rb_pt = np.max(rpts, axis=0)
    center1 = (lt_pt + rb_pt) / 2

    size = rb_pt - lt_pt
    if need_square:
        m = max(size[0], size[1])
        size[0] = m
        size[1] = m

    size *= scale
    center = center0 + ux * center1[0] + uy * center1[1]
    center = center + ux * (vx_ratio * size) + uy * (vy_ratio * size)

    if use_deg_flag:
        angle = degrees(angle)
    return center, size, angle


def parse_bbox_from_landmark(pts, **kwargs):
    center, size, angle = parse_rect_from_landmark(pts, **kwargs)
    cx, cy = center
    w, h = size

    bbox = np.array([
        [cx - w / 2, cy - h / 2],
        [cx + w / 2, cy - h / 2],
        [cx + w / 2, cy + h / 2],
        [cx - w / 2, cy + h / 2],
    ], dtype=DTYPE)

    R = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)],
    ], dtype=DTYPE)
    bbox_rot = (bbox - center) @ R.T + center

    return {
        "center": center,
        "size": size,
        "angle": angle,
        "bbox": bbox,
        "bbox_rot": bbox_rot,
    }


# ---------------------------------------------------------------------------
# Image cropping
# ---------------------------------------------------------------------------

def crop_image_by_bbox(img, bbox, lmk=None, dsize=512, angle=None,
                       flag_rot=False, **kwargs):
    left, top, right, bot = bbox
    size = right - left

    src_center = np.array([(left + right) / 2, (top + bot) / 2], dtype=DTYPE)
    tgt_center = np.array([dsize / 2, dsize / 2], dtype=DTYPE)

    s = dsize / size

    if flag_rot and angle is not None:
        costheta, sintheta = cos(angle), sin(angle)
        cx, cy = src_center
        tcx, tcy = tgt_center
        M_o2c = np.array(
            [
                [s * costheta, s * sintheta, tcx - s * (costheta * cx + sintheta * cy)],
                [-s * sintheta, s * costheta, tcy - s * (-sintheta * cx + costheta * cy)],
            ],
            dtype=DTYPE,
        )
    else:
        M_o2c = np.array(
            [
                [s, 0, tgt_center[0] - s * src_center[0]],
                [0, s, tgt_center[1] - s * src_center[1]],
            ],
            dtype=DTYPE,
        )

    img_crop = _transform_img(
        img, M_o2c, dsize=dsize, borderMode=kwargs.get("borderMode", None)
    )
    lmk_crop = _transform_pts(lmk, M_o2c) if lmk is not None else None

    M_o2c = np.vstack([M_o2c, np.array([0, 0, 1], dtype=DTYPE)])
    M_c2o = np.linalg.inv(M_o2c)

    return {
        "img_crop": img_crop,
        "lmk_crop": lmk_crop,
        "M_o2c": M_o2c,
        "M_c2o": M_c2o,
    }


def _estimate_similar_transform_from_pts(
    pts, dsize, scale=1.5, vx_ratio=0, vy_ratio=-0.1, flag_do_rot=True, **kwargs,
):
    center, size, angle = parse_rect_from_landmark(
        pts, scale=scale, vx_ratio=vx_ratio, vy_ratio=vy_ratio,
        use_lip=kwargs.get("use_lip", True),
    )

    s = dsize / size[0]
    tgt_center = np.array([dsize / 2, dsize / 2], dtype=DTYPE)

    if flag_do_rot:
        costheta, sintheta = cos(angle), sin(angle)
        cx, cy = center
        tcx, tcy = tgt_center
        M_INV = np.array(
            [
                [s * costheta, s * sintheta, tcx - s * (costheta * cx + sintheta * cy)],
                [-s * sintheta, s * costheta, tcy - s * (-sintheta * cx + costheta * cy)],
            ],
            dtype=DTYPE,
        )
    else:
        M_INV = np.array(
            [
                [s, 0, tgt_center[0] - s * center[0]],
                [0, s, tgt_center[1] - s * center[1]],
            ],
            dtype=DTYPE,
        )

    M_INV_H = np.vstack([M_INV, np.array([0, 0, 1])])
    M = np.linalg.inv(M_INV_H)
    return M_INV, M[:2, ...]


def crop_image(img, pts: np.ndarray, **kwargs):
    dsize = kwargs.get("dsize", 224)
    scale = kwargs.get("scale", 1.5)
    vy_ratio = kwargs.get("vy_ratio", -0.1)

    M_INV, _ = _estimate_similar_transform_from_pts(
        pts, dsize=dsize, scale=scale, vy_ratio=vy_ratio,
        flag_do_rot=kwargs.get("flag_do_rot", True),
    )

    img_crop = _transform_img(img, M_INV, dsize)
    pt_crop = _transform_pts(pts, M_INV)

    M_o2c = np.vstack([M_INV, np.array([0, 0, 1], dtype=DTYPE)])
    M_c2o = np.linalg.inv(M_o2c)

    return {
        "M_o2c": M_o2c,
        "M_c2o": M_c2o,
        "img_crop": img_crop,
        "pt_crop": pt_crop,
    }


def average_bbox_lst(bbox_lst):
    if len(bbox_lst) == 0:
        return None
    return np.mean(np.array(bbox_lst), axis=0).tolist()


def contiguous(arr: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(arr)
