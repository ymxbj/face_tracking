"""Decode a video into per-frame face crops + 203-pt landmarks.

The cropping pipeline has two modes:

1. **External landmark mode** — the user already supplies a
   ``[T, 203, 2]`` landmark file. We just dump the video frames as
   ``00000.jpg`` ... and trust the caller to copy ``lmk.npy`` next to
   them. This matches the original Pixel3DMM pipeline.

2. **Auto-detect mode** — invoked when no landmark file is provided.
   We use a vendored slice of LivePortrait's :class:`Cropper` to:

       a. detect the face and refine to 203 keypoints per frame;
       b. average the per-frame bboxes into a single stable crop window;
       c. re-crop every frame to ``dsize x dsize``;
       d. transform the 203 landmarks into cropped image coordinates.

Both modes write to ``preprocessed_data/<video_name>/cropped/`` and
return the source FPS.
"""


import os
from typing import Optional, Tuple

import cv2
import mediapy
import numpy as np
from PIL import Image

from face_tracking import env_paths


_CROPPER_INSTANCE = None  # lazy singleton — initialising InsightFace is slow.


def _read_video(video_path: str) -> Tuple[np.ndarray, float]:
    """Return ``(frames [T, H, W, 3] RGB uint8, fps)``."""
    frames = mediapy.read_video(video_path)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return frames, fps


def _get_cropper(
    landmark_ckpt_path: Optional[str] = None,
    insightface_root: Optional[str] = None,
):
    global _CROPPER_INSTANCE
    if _CROPPER_INSTANCE is not None:
        return _CROPPER_INSTANCE

    from face_tracking.preprocessing.cropper import Cropper, CropConfig

    landmark_ckpt = landmark_ckpt_path or env_paths.CKPT_LANDMARK_ONNX
    if not os.path.exists(landmark_ckpt):
        raise FileNotFoundError(
            f"landmark.onnx not found at {landmark_ckpt}. "
            "Run scripts/download_weights.sh or set FACE_TRACKING_PRETRAINED."
        )

    cfg = CropConfig(
        insightface_root=insightface_root or env_paths.INSIGHTFACE_ROOT,
        landmark_ckpt_path=landmark_ckpt,
    )
    print("Initializing face cropper (RetinaFace + 106→203 landmark) ...")
    _CROPPER_INSTANCE = Cropper(cfg)
    return _CROPPER_INSTANCE


def _save_frames(target_dir: str, frames) -> None:
    os.makedirs(target_dir, exist_ok=True)
    existing = [f for f in os.listdir(target_dir) if f.endswith((".jpg", ".png"))]
    if len(existing) == len(frames):
        return
    for i, frame in enumerate(frames):
        Image.fromarray(frame).save(os.path.join(target_dir, f"{i:05d}.jpg"), quality=95)


def run(video_path: str, video_name: str) -> float:
    """External-landmark mode — extract every frame as a JPG.

    Used by ``api.track_video`` when the caller already has landmarks.
    """
    target_dir = os.path.join(env_paths.PREPROCESSED_DATA, video_name, "cropped")

    frames, fps = _read_video(video_path)
    _save_frames(target_dir, frames)
    return fps


def run_with_auto_landmarks(
    video_path: str,
    video_name: str,
    landmark_ckpt_path: Optional[str] = None,
    insightface_root: Optional[str] = None,
) -> Tuple[float, str]:
    """Auto-detect mode — crop the video and write both frames and ``lmk.npy``.

    Returns
    -------
    (fps, lmk_path)
        ``fps`` of the source video, and the path to the saved
        ``lmk.npy`` (shape ``[T, 203, 2]``) that the tracker expects.
    """
    out_dir = os.path.join(env_paths.PREPROCESSED_DATA, video_name)
    target_dir = os.path.join(out_dir, "cropped")
    lmk_path = os.path.join(out_dir, "lmk.npy")

    os.makedirs(out_dir, exist_ok=True)

    frames, fps = _read_video(video_path)

    if (
        os.path.exists(target_dir)
        and len([f for f in os.listdir(target_dir) if f.endswith((".jpg", ".png"))]) == len(frames)
        and os.path.exists(lmk_path)
    ):
        print(f"<<<<<<<< CROP+LMK ALREADY DONE for {video_name}, SKIPPING >>>>>>>>")
        return fps, lmk_path

    cropper = _get_cropper(
        landmark_ckpt_path=landmark_ckpt_path,
        insightface_root=insightface_root,
    )

    out = cropper.crop_driving_video(list(frames))
    crop_frames = out["frame_crop_lst"]
    crop_lmks = out["lmk_crop_lst"]

    _save_frames(target_dir, crop_frames)

    lmk_arr = np.stack([np.asarray(l, dtype=np.float32) for l in crop_lmks], axis=0)
    np.save(lmk_path, lmk_arr)

    print(
        f"[Cropper] {video_name}: {len(crop_frames)} frames cropped, "
        f"lmks {lmk_arr.shape} saved to {lmk_path}"
    )
    return fps, lmk_path
