#!/usr/bin/env python3
"""Track many videos in one pass.

The list of videos to track can be supplied either as

* ``--video-dir`` + ``--landmark-dir`` — every video file in
  ``video-dir`` is paired with the same-stem ``.npy`` file from
  ``landmark-dir``;
* or ``--list`` — a text file with one ``video_path landmark_path`` pair
  per line (whitespace separated).

Example
-------

    python scripts/track_batch.py \\
        --video-dir data/videos --landmark-dir data/landmarks \\
        --output-dir results/

    python scripts/track_batch.py \\
        --list pairs.txt --output-dir results/
"""


import argparse
import sys

from face_tracking import track_videos
from face_tracking.api import load_pairs_from_dir, load_pairs_from_file


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    src = parser.add_argument_group("video source (pick one)")
    src.add_argument("--video-dir", default=None, help="Directory containing videos.")
    src.add_argument(
        "--landmark-dir",
        default=None,
        help="Optional directory of matching .npy landmarks. Omit to auto-detect per video.",
    )
    src.add_argument(
        "--list",
        dest="list_file",
        default=None,
        help="Text file with one `video_path [landmark_path]` per line. "
             "Omit the landmark column to auto-detect for that row.",
    )

    parser.add_argument("--output-dir", default=None, help="Output root directory.")
    parser.add_argument("--save-meshes", action="store_true", help="Save per-frame OBJ meshes.")
    parser.add_argument(
        "--no-save-keypoints",
        dest="save_keypoints",
        action="store_false",
        help="Disable saving keypoints.",
    )
    parser.add_argument("--remove-intermediate", action="store_true", help="Remove intermediate artefacts.")
    parser.add_argument("--flame-assets", default=None, help="Override the FLAME 2020 assets directory.")
    parser.add_argument(
        "--success-log",
        default=None,
        help="Append successful (video, landmark) pairs to this file.",
    )
    parser.add_argument(
        "--failed-log",
        default=None,
        help="Append failed (video, landmark) pairs to this file.",
    )
    parser.set_defaults(save_keypoints=True)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    if args.list_file:
        pairs = load_pairs_from_file(args.list_file)
    elif args.video_dir:
        pairs = load_pairs_from_dir(args.video_dir, args.landmark_dir)
    else:
        print("Error: pass either --list or --video-dir.", file=sys.stderr)
        return 2

    if not pairs:
        print("No (video, landmark) pairs were found.", file=sys.stderr)
        return 2

    print(f"Tracking {len(pairs)} videos ...")

    results = track_videos(
        pairs=pairs,
        output_dir=args.output_dir,
        save_meshes=args.save_meshes,
        save_keypoints=args.save_keypoints,
        keep_intermediate=not args.remove_intermediate,
        flame_assets=args.flame_assets,
        success_log_path=args.success_log,
        failed_log_path=args.failed_log,
    )

    n_ok = sum(1 for r in results if r.success)
    n_fail = len(results) - n_ok
    print()
    print(f"Done. success={n_ok}, failed={n_fail}, total={len(results)}.")
    for r in results:
        flag = "OK  " if r.success else "FAIL"
        msg = r.output_dir if r.success else (r.error or "")
        print(f"  [{flag}] {r.video_name} ({r.elapsed_seconds:.1f}s) {msg}")

    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
