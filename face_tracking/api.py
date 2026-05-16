"""High-level Python API for video face tracking.

The two main entry points are :func:`track_video` (single video) and
:func:`track_videos` (a list of videos). Both accept a video file
together with a pre-computed landmark file (``.npy``).

Example
-------

>>> from face_tracking import track_video
>>> track_video(
...     video_path="data/test.mp4",
...     output_dir="results",
... )
"""


import os
import shutil
import time
import traceback
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import torch

from face_tracking import env_paths


@dataclass
class TrackingResult:
    video_path: str
    landmark_path: Optional[str]
    video_name: str
    output_dir: str
    success: bool
    elapsed_seconds: float
    error: Optional[str] = None


def _derive_video_name(video_path: str) -> str:
    base = os.path.basename(video_path)
    stem, _ = os.path.splitext(base)
    return stem


def track_video(
    video_path: str,
    landmark_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    *,
    video_name: Optional[str] = None,
    save_meshes: bool = False,
    save_keypoints: bool = True,
    keep_intermediate: bool = True,
    flame_assets: Optional[str] = None,
    cfg_overrides: Optional[dict] = None,
) -> TrackingResult:
    """Run the full tracking pipeline for a single video.

    Parameters
    ----------
    video_path
        Path to the input video (``.mp4`` or any container readable by
        ``mediapy``).
    landmark_path
        Optional path to a pre-computed ``[T, 203, 2]`` landmark file.
        When ``None`` the pipeline runs LivePortrait's RetinaFace +
        203-pt landmark detector (``preprocessing.cropper``) to produce
        cropped frames and landmarks automatically.
    output_dir
        Directory in which the tracking artefacts are written. If
        ``None``, defaults to :data:`env_paths.TRACKING_OUTPUT`.
    video_name
        Logical name used for the per-video sub-folder. Derived from the
        video filename when omitted.
    save_meshes
        If ``True``, dump a per-frame ``.obj`` mesh.
    save_keypoints
        If ``True``, save canonical / expression / projected keypoints
        as ``.npy`` files under ``key_points/``.
    keep_intermediate
        If ``False``, remove the intermediate preprocessing artefacts
        (cropped frames, segmentations, network predictions) once
        tracking is complete.
    flame_assets
        Directory containing the FLAME 2020 assets. Defaults to
        :data:`env_paths.FLAME_ASSETS`.
    cfg_overrides
        Optional overrides for :file:`configs/tracking.yaml` (e.g.
        ``{"iters": 800}`` for high-quality single-image fitting).
    """
    # Local imports to avoid heavy dependencies at module import time.
    from face_tracking.preprocessing import cropping, network_inference, segmentation
    from face_tracking.tracking import run_track

    if not os.path.exists(video_path):
        raise FileNotFoundError(video_path)
    if landmark_path is not None and not os.path.exists(landmark_path):
        raise FileNotFoundError(landmark_path)

    name = video_name or _derive_video_name(video_path)
    base_output = output_dir or env_paths.TRACKING_OUTPUT
    os.makedirs(base_output, exist_ok=True)

    start_time = time.time()
    success = False
    error_msg: Optional[str] = None

    try:
        per_video_dir = os.path.join(env_paths.PREPROCESSED_DATA, name)
        os.makedirs(per_video_dir, exist_ok=True)

        if landmark_path is None:
            # Auto-detect 203-pt landmarks via the bundled cropper.
            fps, _ = cropping.run_with_auto_landmarks(
                video_path=video_path, video_name=name,
            )
        else:
            fps = cropping.run(video_path=video_path, video_name=name)
            shutil.copy(landmark_path, os.path.join(per_video_dir, "lmk.npy"))

        segmentation.run(video_name=name)
        network_inference.run(video_name=name)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        run_track.run(
            video_name=name,
            fps=fps,
            save_meshes=save_meshes,
            save_keypoints=save_keypoints,
            flame_assets=flame_assets,
            cfg_overrides={
                **({"output_folder": base_output} if base_output else {}),
                **(cfg_overrides or {}),
            },
        )
        success = True
    except Exception as exc:  # noqa: BLE001
        error_msg = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if not keep_intermediate:
        shutil.rmtree(os.path.join(env_paths.PREPROCESSED_DATA, name), ignore_errors=True)

    return TrackingResult(
        video_path=video_path,
        landmark_path=landmark_path,
        video_name=name,
        output_dir=os.path.join(base_output, name),
        success=success,
        elapsed_seconds=time.time() - start_time,
        error=error_msg,
    )


def track_videos(
    pairs: Iterable[Tuple[str, Optional[str]]],
    output_dir: Optional[str] = None,
    *,
    save_meshes: bool = False,
    save_keypoints: bool = True,
    keep_intermediate: bool = True,
    flame_assets: Optional[str] = None,
    cfg_overrides: Optional[dict] = None,
    failed_log_path: Optional[str] = None,
    success_log_path: Optional[str] = None,
) -> List[TrackingResult]:
    """Run the tracking pipeline on every ``(video_path, landmark_path)`` pair.

    Each pair may be ``(video_path, landmark_path)`` or
    ``(video_path, None)`` — passing ``None`` triggers automatic
    landmark detection via the bundled cropper.

    Failures are caught per-video so that one bad sample does not abort
    the whole batch. If ``failed_log_path`` / ``success_log_path`` are
    given, the corresponding entries are appended (one per line, space
    separated).
    """
    results: List[TrackingResult] = []
    for entry in pairs:
        if isinstance(entry, str):
            video_path, landmark_path = entry, None
        else:
            video_path, landmark_path = entry[0], (entry[1] if len(entry) > 1 else None)

        result = track_video(
            video_path=video_path,
            landmark_path=landmark_path,
            output_dir=output_dir,
            save_meshes=save_meshes,
            save_keypoints=save_keypoints,
            keep_intermediate=keep_intermediate,
            flame_assets=flame_assets,
            cfg_overrides=cfg_overrides,
        )
        results.append(result)

        log_path = success_log_path if result.success else failed_log_path
        if log_path:
            os.makedirs(os.path.dirname(os.path.abspath(log_path)) or ".", exist_ok=True)
            with open(log_path, "a", encoding="utf-8") as f:
                if landmark_path:
                    f.write(f"{video_path} {landmark_path}\n")
                else:
                    f.write(f"{video_path}\n")

    return results


def load_pairs_from_file(path: str) -> List[Tuple[str, Optional[str]]]:
    """Load a per-line video list.

    Each non-empty line is either:

    * ``video_path`` — auto-detect landmarks via the bundled cropper, or
    * ``video_path landmark_path`` — use the supplied ``[T, 203, 2]`` ``.npy``.
    """
    pairs: List[Tuple[str, Optional[str]]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            if len(parts) == 1:
                pairs.append((parts[0], None))
            else:
                pairs.append((parts[0], parts[1]))
    return pairs


def load_pairs_from_dir(
    video_dir: str,
    landmark_dir: Optional[str] = None,
    *,
    video_extensions: Sequence[str] = (".mp4", ".mov", ".avi", ".mkv"),
    landmark_extension: str = ".npy",
) -> List[Tuple[str, Optional[str]]]:
    """Pair every video in ``video_dir`` with a same-stem landmark file.

    When ``landmark_dir`` is ``None`` the loader yields ``(video, None)``
    pairs so the auto-detect cropper handles landmarks.
    """
    pairs: List[Tuple[str, Optional[str]]] = []
    for name in sorted(os.listdir(video_dir)):
        stem, ext = os.path.splitext(name)
        if ext.lower() not in video_extensions:
            continue
        video_path = os.path.join(video_dir, name)
        if landmark_dir is None:
            pairs.append((video_path, None))
            continue
        lmk = os.path.join(landmark_dir, stem + landmark_extension)
        pairs.append((video_path, lmk if os.path.exists(lmk) else None))
    return pairs
