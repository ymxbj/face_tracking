"""Run FLAME tracking on the (already preprocessed) video frames."""


from typing import Optional

from omegaconf import OmegaConf

from face_tracking import env_paths
from face_tracking.tracking.tracker import Tracker


_TRACKER_INSTANCE: Optional[Tracker] = None


def _get_tracker(flame_assets: Optional[str] = None) -> Tracker:
    global _TRACKER_INSTANCE
    if _TRACKER_INSTANCE is None:
        print("Initializing tracker ...")
        _TRACKER_INSTANCE = Tracker(flame_assets=flame_assets)
    return _TRACKER_INSTANCE


def run(
    video_name: str,
    fps: float = 30.0,
    save_meshes: bool = False,
    save_keypoints: bool = True,
    flame_assets: Optional[str] = None,
    cfg_overrides: Optional[dict] = None,
) -> None:
    """Run the full tracking pipeline for ``video_name``.

    The function expects all preprocessing artefacts (cropped frames,
    segmentation, normal/UV predictions, landmarks) to already exist
    under ``preprocessed_data/<video_name>/``.

    Parameters
    ----------
    video_name
        Logical name of the video (subdirectory name).
    fps
        Output FPS for the rendered debug videos.
    save_meshes
        If ``True``, dump per-frame OBJ meshes (slow, large).
    save_keypoints
        If ``True``, save per-frame canonical / expression / projected
        keypoints under ``tracking_output/<video_name>/key_points``.
    flame_assets
        Override the FLAME assets directory.
    cfg_overrides
        Optional dictionary of values to merge on top of
        ``configs/tracking.yaml`` (e.g. ``{"iters": 800}``).
    """
    cfg = OmegaConf.load(env_paths.get_tracking_yaml())
    if cfg_overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.create(cfg_overrides))
    cfg.video_name = video_name
    cfg.fps = fps

    tracker = _get_tracker(flame_assets=flame_assets)
    tracker.initialize(cfg)
    tracker.run(save_keypoints=save_keypoints, save_meshes=save_meshes, not_render=False)
