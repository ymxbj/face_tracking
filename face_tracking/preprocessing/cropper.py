# coding: utf-8

"""Cropper — face-tracking-based stable cropping with 203-pt landmarks.

Vendored and trimmed from LivePortrait (``src/utils/cropper.py``). The
animal-face / source-image variants have been removed; only the driving-
video flow used by ``face_tracking`` remains.

Workflow
--------

1. Frame 0: run RetinaFace + 2d-106 landmark via InsightFace.
2. Frame 0..N-1: refine the coarse 106-pt landmark into 203 points via
   the ONNX :class:`LandmarkRunner`. From frame 1 onwards, the previous
   frame's 203-pt landmark is used as the warm start, which gives
   smoother tracking than re-detecting every frame.
3. Average per-frame bounding boxes into one stable crop window, then
   re-crop every original frame to that window at the requested size
   (default 512x512) and transform the 203-pt landmarks into cropped
   image coordinates.
"""


from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

from .crop import (
    average_bbox_lst,
    contiguous,
    crop_image_by_bbox,
    parse_bbox_from_landmark,
)
from .face_analysis import FaceAnalysisDIY
from .landmark_runner import LandmarkRunner

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)


@dataclass
class CropConfig:
    insightface_root: str = "~/.insightface"
    landmark_ckpt_path: str = ""
    device_id: int = 0
    flag_force_cpu: bool = False
    det_thresh: float = 0.1
    dsize: int = 512
    direction: str = "large-small"
    # Driving-video crop window (size of a single face within ``dsize``).
    scale_crop_driving_video: float = 2.2
    vx_ratio_crop_driving_video: float = 0.0
    vy_ratio_crop_driving_video: float = -0.1


@dataclass
class _Trajectory:
    start: int = -1
    end: int = -1
    lmk_lst: List = field(default_factory=list)
    bbox_lst: List = field(default_factory=list)
    frame_rgb_lst: List = field(default_factory=list)
    lmk_crop_lst: List = field(default_factory=list)
    frame_rgb_crop_lst: List = field(default_factory=list)


class Cropper:
    """Detect, track, and stably crop a face out of a driving video."""

    def __init__(self, crop_cfg: CropConfig):
        self.crop_cfg = crop_cfg

        # Pick the right ONNX execution provider.
        if crop_cfg.flag_force_cpu:
            device = "cpu"
            face_providers: Sequence[str] = ["CPUExecutionProvider"]
        else:
            try:
                import torch
                if torch.backends.mps.is_available():
                    # InsightFace's RetinaFace shape inference fails on CoreML.
                    device = "mps"
                    face_providers = ["CPUExecutionProvider"]
                elif torch.cuda.is_available():
                    device = "cuda"
                    face_providers = ["CUDAExecutionProvider"]
                else:
                    device = "cpu"
                    face_providers = ["CPUExecutionProvider"]
            except Exception:
                device = "cuda"
                face_providers = ["CUDAExecutionProvider"]

        self.face_analysis_wrapper = FaceAnalysisDIY(
            name="buffalo_l",
            root=crop_cfg.insightface_root,
            providers=face_providers,
        )
        self.face_analysis_wrapper.prepare(
            ctx_id=crop_cfg.device_id,
            det_size=(512, 512),
            det_thresh=crop_cfg.det_thresh,
        )
        self.face_analysis_wrapper.warmup()

        self.human_landmark_runner = LandmarkRunner(
            ckpt_path=crop_cfg.landmark_ckpt_path,
            onnx_provider=device,
            device_id=crop_cfg.device_id,
        )
        self.human_landmark_runner.warmup()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def crop_driving_video(
        self,
        driving_rgb_lst: Sequence[np.ndarray],
        dsize: Optional[int] = None,
        direction: Optional[str] = None,
    ) -> dict:
        """Run face tracking, average the bbox, and re-crop every frame.

        Parameters
        ----------
        driving_rgb_lst
            Sequence of RGB frames (H, W, 3).
        dsize
            Output crop size. Defaults to :attr:`CropConfig.dsize`.
        direction
            Face-selection rule when multiple faces are present.

        Returns
        -------
        dict with keys:
            * ``frame_crop_lst``: list of cropped RGB frames at ``dsize``.
            * ``lmk_crop_lst``: list of ``(203, 2)`` landmarks in cropped
              frame coordinates.
        """
        dsize = dsize or self.crop_cfg.dsize
        direction = direction or self.crop_cfg.direction

        traj = _Trajectory()
        for idx, frame_rgb in enumerate(driving_rgb_lst):
            if idx == 0 or traj.start == -1:
                src_face = self.face_analysis_wrapper.get(
                    contiguous(frame_rgb[..., ::-1]),  # RGB -> BGR
                    flag_do_landmark_2d_106=True,
                    direction=direction,
                )
                if not src_face:
                    print(f"[Cropper] no face detected in frame #{idx}, skipping ...")
                    continue
                if len(src_face) > 1:
                    print(
                        f"[Cropper] {len(src_face)} faces in frame {idx}, "
                        f"picking by '{direction}'."
                    )
                src_face = src_face[0]
                lmk = src_face.landmark_2d_106
                lmk = self.human_landmark_runner.run(frame_rgb, lmk)
                traj.start = traj.end = idx
            else:
                lmk = self.human_landmark_runner.run(frame_rgb, traj.lmk_lst[-1])
                traj.end = idx

            traj.lmk_lst.append(lmk)
            ret_bbox = parse_bbox_from_landmark(
                lmk,
                scale=self.crop_cfg.scale_crop_driving_video,
                vx_ratio=self.crop_cfg.vx_ratio_crop_driving_video,
                vy_ratio=self.crop_cfg.vy_ratio_crop_driving_video,
            )["bbox"]
            traj.bbox_lst.append(
                [ret_bbox[0, 0], ret_bbox[0, 1], ret_bbox[2, 0], ret_bbox[2, 1]]
            )
            traj.frame_rgb_lst.append(frame_rgb)

        if not traj.frame_rgb_lst:
            raise RuntimeError("Cropper: no face detected in any frame of the video.")

        global_bbox = average_bbox_lst(traj.bbox_lst)

        for frame_rgb, lmk in zip(traj.frame_rgb_lst, traj.lmk_lst):
            ret = crop_image_by_bbox(
                frame_rgb, global_bbox, lmk=lmk,
                dsize=dsize, flag_rot=False, borderValue=(0, 0, 0),
            )
            traj.frame_rgb_crop_lst.append(ret["img_crop"])
            traj.lmk_crop_lst.append(ret["lmk_crop"])

        return {
            "frame_crop_lst": traj.frame_rgb_crop_lst,
            "lmk_crop_lst": traj.lmk_crop_lst,
        }

    def calc_lmks_from_cropped_video(
        self,
        cropped_rgb_lst: Sequence[np.ndarray],
        direction: str = "large-small",
    ) -> List[np.ndarray]:
        """Run 203-pt landmark detection on already-cropped frames."""
        traj = _Trajectory()
        for idx, frame_rgb in enumerate(cropped_rgb_lst):
            if idx == 0 or traj.start == -1:
                src_face = self.face_analysis_wrapper.get(
                    contiguous(frame_rgb[..., ::-1]),
                    flag_do_landmark_2d_106=True,
                    direction=direction,
                )
                if not src_face:
                    raise RuntimeError(f"No face detected in cropped frame #{idx}.")
                lmk = src_face[0].landmark_2d_106
                lmk = self.human_landmark_runner.run(frame_rgb, lmk)
                traj.start = traj.end = idx
            else:
                lmk = self.human_landmark_runner.run(frame_rgb, traj.lmk_lst[-1])
                traj.end = idx
            traj.lmk_lst.append(lmk)
        return traj.lmk_lst
