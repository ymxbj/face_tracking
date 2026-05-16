# coding: utf-8

"""203-point human-face landmark refiner (ONNX, vendored from LivePortrait)."""


import time
from typing import Optional

import cv2
import numpy as np

from .crop import _transform_pts, crop_image

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)


def _to_ndarray(obj) -> np.ndarray:
    try:
        import torch
        if isinstance(obj, torch.Tensor):
            return obj.cpu().numpy()
    except ImportError:
        pass
    if isinstance(obj, np.ndarray):
        return obj
    return np.array(obj)


class LandmarkRunner(object):
    """Refines a coarse face landmark into 203 points via an ONNX model.

    The forward pass takes the original image plus a coarse landmark
    (typically the 106-point output from InsightFace) and returns 203
    landmarks in the **original image coordinate frame**.
    """

    def __init__(
        self,
        ckpt_path: str,
        onnx_provider: str = "cuda",
        device_id: int = 0,
        dsize: int = 224,
    ):
        import onnxruntime

        self.dsize = dsize

        if onnx_provider.lower() == "cuda":
            providers = [("CUDAExecutionProvider", {"device_id": device_id})]
        elif onnx_provider.lower() == "mps":
            providers = ["CoreMLExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]

        opts = onnxruntime.SessionOptions()
        opts.intra_op_num_threads = 4
        try:
            self.session = onnxruntime.InferenceSession(
                ckpt_path, providers=providers, sess_options=opts,
            )
        except Exception:
            # Provider unavailable — fall back to CPU.
            self.session = onnxruntime.InferenceSession(
                ckpt_path, providers=["CPUExecutionProvider"], sess_options=opts,
            )

    def _run(self, inp: np.ndarray):
        return self.session.run(None, {"input": inp})

    def run(self, img_rgb: np.ndarray, lmk: Optional[np.ndarray] = None) -> np.ndarray:
        """Return 203 landmarks for ``img_rgb``.

        Parameters
        ----------
        img_rgb : ndarray
            Full-resolution RGB frame.
        lmk : ndarray, optional
            Coarse landmark (e.g. 106-pt from InsightFace) used to crop a
            tight ``dsize x dsize`` patch around the face. If ``None`` the
            whole image is naively resized — not recommended.
        """
        if lmk is not None:
            crop_dct = crop_image(img_rgb, lmk, dsize=self.dsize, scale=1.5, vy_ratio=-0.1)
            img_crop_rgb = crop_dct["img_crop"]
        else:
            img_crop_rgb = cv2.resize(img_rgb, (self.dsize, self.dsize))
            scale = max(img_rgb.shape[:2]) / self.dsize
            crop_dct = {
                "M_c2o": np.array(
                    [[scale, 0.0, 0.0], [0.0, scale, 0.0], [0.0, 0.0, 1.0]],
                    dtype=np.float32,
                ),
            }

        inp = (img_crop_rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)[None, ...]
        out_lst = self._run(inp)
        out_pts = out_lst[2]

        # Output: 203 points in [0, 1] -> scale to dsize -> back to original frame.
        lmk_out = _to_ndarray(out_pts[0]).reshape(-1, 2) * self.dsize
        return _transform_pts(lmk_out, M=crop_dct["M_c2o"])

    def warmup(self) -> float:
        t0 = time.perf_counter()
        dummy = np.zeros((1, 3, self.dsize, self.dsize), dtype=np.float32)
        self._run(dummy)
        return time.perf_counter() - t0
