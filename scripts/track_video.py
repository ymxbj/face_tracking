#!/usr/bin/env python3
"""Track a single video with pre-computed landmarks.

Example
-------

    python scripts/track_video.py \
        --video data/test.mp4 \
        --landmark data/test.npy \
        --output-dir results/
"""


import argparse
import sys

from face_tracking import track_video


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", required=True, help="Path to the input video file (.mp4 / .mov / ...).")
    parser.add_argument(
        "--landmark",
        default=None,
        help="Optional pre-computed [T, 203, 2] landmark .npy. Omit to auto-detect.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Defaults to $FACE_TRACKING_OUTPUT or ./tracking_output.",
    )
    parser.add_argument("--video-name", default=None, help="Override the per-video sub-folder name.")
    parser.add_argument("--save-meshes", action="store_true", help="Save per-frame OBJ meshes.")
    parser.add_argument(
        "--no-save-keypoints",
        dest="save_keypoints",
        action="store_false",
        help="Disable saving per-frame canonical/expression keypoints.",
    )
    parser.add_argument(
        "--remove-intermediate",
        action="store_true",
        help="Remove preprocessing artefacts after tracking finishes.",
    )
    parser.add_argument(
        "--flame-assets",
        default=None,
        help="Override the FLAME 2020 assets directory.",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=None,
        help="Override the per-frame iteration count (use 800+ for single-image fitting).",
    )
    parser.set_defaults(save_keypoints=True)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    cfg_overrides = {}
    if args.iters is not None:
        cfg_overrides["iters"] = args.iters

    result = track_video(
        video_path=args.video,
        landmark_path=args.landmark,
        output_dir=args.output_dir,
        video_name=args.video_name,
        save_meshes=args.save_meshes,
        save_keypoints=args.save_keypoints,
        keep_intermediate=not args.remove_intermediate,
        flame_assets=args.flame_assets,
        cfg_overrides=cfg_overrides or None,
    )

    print()
    if result.success:
        print(f"[OK] {result.video_name}: {result.elapsed_seconds:.1f}s -> {result.output_dir}")
        return 0

    print(f"[FAIL] {result.video_name}: {result.error}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
