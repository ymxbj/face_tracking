#!/usr/bin/env python3
"""Distributed batch tracking launcher.

Each launched process is a worker that pulls one task at a time from a
shared ``torch.distributed.TCPStore`` counter, so fast GPUs naturally
pick up more work and stragglers don't block the queue. Per-task
success/failure is appended to two log files (``success_<tag>.txt`` /
``failed_<tag>.txt``) under the output directory with ``fcntl`` file
locks, which makes ``--resume`` safe across crashes and re-runs.

Examples
--------

Single host, 4 GPUs (default — one process per GPU, NCCL)::

    torchrun --nproc_per_node=4 scripts/track_distributed.py \\
        --list pairs.txt --output-dir results/ --tag run_a --resume

Multi-host, 8 GPUs each::

    torchrun --nnodes=2 --nproc_per_node=8 --rdzv_endpoint=master:29500 \\
        scripts/track_distributed.py --list pairs.txt --output-dir results/

Multiple processes sharing one GPU (gloo backend)::

    torchrun --nproc_per_node=8 scripts/track_distributed.py \\
        --list pairs.txt --output-dir results/ --multi-process-per-gpu

Without ``torchrun`` the script falls back to a single-process run, which
is convenient for debugging and tiny lists.
"""


import argparse
import fcntl
import multiprocessing
import os
import sys
import time
from typing import List, Optional, Tuple

import torch
import torch.distributed as dist


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )

    src = parser.add_argument_group("video source (pick one)")
    src.add_argument("--list", dest="list_file", default=None,
                     help="Text file with `video_path [landmark_path]` per line. "
                          "Omit the landmark to auto-detect.")
    src.add_argument("--video-dir", default=None, help="Directory containing videos.")
    src.add_argument("--landmark-dir", default=None,
                     help="Optional directory of matching .npy landmarks. "
                          "Omit to auto-detect per video.")

    parser.add_argument("--output-dir", required=True, help="Output root directory.")
    parser.add_argument("--tag", default="run",
                        help="Tag used for the success/failed log filenames.")
    parser.add_argument("--resume", action="store_true",
                        help="Skip pairs that already appear in the success/failed log.")
    parser.add_argument("--save-meshes", action="store_true",
                        help="Save per-frame OBJ meshes.")
    parser.add_argument("--no-save-keypoints", dest="save_keypoints",
                        action="store_false",
                        help="Disable saving per-frame keypoints.")
    parser.add_argument("--remove-intermediate", action="store_true",
                        help="Remove preprocessing artefacts after each video.")
    parser.add_argument("--flame-assets", default=None,
                        help="Override the FLAME 2020 assets directory.")
    parser.add_argument("--counter-port", type=int, default=29501,
                        help="Port for the TCPStore-based task counter.")
    parser.add_argument("--multi-process-per-gpu", action="store_true",
                        help="Use the gloo backend and pin LOCAL_RANK to "
                             "rank // (world_size / n_gpu) so multiple ranks "
                             "share a single GPU.")
    parser.set_defaults(save_keypoints=True)
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def load_pairs(args: argparse.Namespace) -> List[Tuple[str, Optional[str]]]:
    from face_tracking.api import load_pairs_from_dir, load_pairs_from_file

    if args.list_file:
        return load_pairs_from_file(args.list_file)
    if args.video_dir:
        return load_pairs_from_dir(args.video_dir, args.landmark_dir)
    raise SystemExit("Error: pass either --list or --video-dir.")


def load_processed_set(log_path: str) -> set:
    """Read a ``video [landmark]`` log file into a set of pairs."""
    if not os.path.exists(log_path):
        return set()
    processed = set()
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            if len(parts) == 1:
                processed.add((parts[0], None))
            else:
                processed.add((parts[0], parts[1]))
    return processed


def append_locked(log_path: str, line: str) -> None:
    """Append ``line`` to ``log_path`` under an exclusive ``fcntl`` lock.

    Multiple ranks may write concurrently (especially when several
    processes share a single host filesystem), so the lock prevents
    interleaved or torn writes.
    """
    os.makedirs(os.path.dirname(os.path.abspath(log_path)) or ".", exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX)
            f.write(line)
            if not line.endswith("\n"):
                f.write("\n")
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


# ---------------------------------------------------------------------------
# Distributed setup
# ---------------------------------------------------------------------------


def is_torchrun() -> bool:
    return "RANK" in os.environ and "WORLD_SIZE" in os.environ


def init_distributed(args: argparse.Namespace) -> Tuple[int, int]:
    """Initialise the process group and pin this process to a GPU.

    Returns
    -------
    (rank, world_size)
        ``(0, 1)`` when launched outside ``torchrun`` (single-process mode).
    """
    if not is_torchrun():
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
        return 0, 1

    backend = "gloo" if args.multi_process_per_gpu else "nccl"
    dist.init_process_group(backend=backend)

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    if torch.cuda.is_available():
        if args.multi_process_per_gpu:
            n_gpu = max(1, torch.cuda.device_count())
            process_per_gpu = max(1, world_size // n_gpu)
            local_rank = rank // process_per_gpu
        else:
            local_rank = int(os.environ.get("LOCAL_RANK", rank))
        local_rank = local_rank % max(1, torch.cuda.device_count())
        torch.cuda.set_device(local_rank)
        os.environ["LOCAL_RANK"] = str(local_rank)
        print(f"[rank {rank}/{world_size}] pinned to CUDA device {local_rank}", flush=True)
    else:
        print(f"[rank {rank}/{world_size}] no CUDA available, running on CPU", flush=True)

    return rank, world_size


def broadcast_pairs(
    pairs: List[Tuple[str, str]], rank: int, world_size: int
) -> List[Tuple[str, str]]:
    """Broadcast the (resume-filtered) work list from rank 0."""
    if world_size == 1:
        return pairs
    payload = [pairs] if rank == 0 else [None]
    dist.broadcast_object_list(payload, src=0)
    return payload[0]


def make_counter_store(
    rank: int, world_size: int, port: int
) -> Optional["dist.TCPStore"]:
    """Create the shared ``task_idx`` counter on rank 0, others connect."""
    if world_size == 1:
        return None
    host = os.environ.get("MASTER_ADDR", "localhost")
    if rank == 0:
        store = dist.TCPStore(
            host_name=host, port=port, world_size=world_size, is_master=True
        )
        store.set("task_idx", "0")
        print(
            f"[rank 0] TCPStore counter listening on {host}:{port}", flush=True
        )
    else:
        time.sleep(2)  # let rank 0 bind first
        store = dist.TCPStore(
            host_name=host, port=port, world_size=world_size, is_master=False
        )
    return store


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def run_worker(
    args: argparse.Namespace,
    rank: int,
    world_size: int,
    pairs: List[Tuple[str, str]],
    counter: Optional["dist.TCPStore"],
    success_log: str,
    failed_log: str,
) -> int:
    # Lazy import so torch.cuda.set_device is in effect first.
    from face_tracking import track_video

    n_done = 0
    n_ok = 0
    n_fail = 0
    started = time.time()

    while True:
        if counter is None:
            idx = n_done
        else:
            idx = counter.add("task_idx", 1) - 1

        if idx >= len(pairs):
            break

        video_path, landmark_path = pairs[idx]

        result = track_video(
            video_path=video_path,
            landmark_path=landmark_path,
            output_dir=args.output_dir,
            save_meshes=args.save_meshes,
            save_keypoints=args.save_keypoints,
            keep_intermediate=not args.remove_intermediate,
            flame_assets=args.flame_assets,
        )

        log_path = success_log if result.success else failed_log
        if landmark_path:
            append_locked(log_path, f"{video_path} {landmark_path}")
        else:
            append_locked(log_path, video_path)

        flag = "OK  " if result.success else "FAIL"
        msg = result.output_dir if result.success else (result.error or "")
        print(
            f"[rank {rank}] [{flag}] task#{idx} {result.video_name} "
            f"({result.elapsed_seconds:.1f}s) {msg}",
            flush=True,
        )

        n_done += 1
        n_ok += int(result.success)
        n_fail += int(not result.success)

    elapsed = time.time() - started
    print(
        f"[rank {rank}] finished {n_done} task(s) "
        f"({n_ok} ok / {n_fail} failed) in {elapsed:.1f}s.",
        flush=True,
    )
    return n_done


def main(argv=None) -> int:
    multiprocessing.set_start_method("spawn", force=True)

    args = parse_args(argv)
    rank, world_size = init_distributed(args)

    os.makedirs(args.output_dir, exist_ok=True)
    success_log = os.path.join(args.output_dir, f"success_{args.tag}.txt")
    failed_log = os.path.join(args.output_dir, f"failed_{args.tag}.txt")

    # Rank 0 owns the work-list construction (single source of truth).
    pairs: List[Tuple[str, str]] = []
    if rank == 0:
        all_pairs = load_pairs(args)
        if args.resume:
            done = load_processed_set(success_log) | load_processed_set(failed_log)
            pairs = [p for p in all_pairs if p not in done]
            print(
                f"[rank 0] total={len(all_pairs)} already_processed={len(done)} "
                f"remaining={len(pairs)}",
                flush=True,
            )
        else:
            pairs = all_pairs
            print(f"[rank 0] {len(pairs)} pairs to track.", flush=True)

        if not pairs:
            print("[rank 0] nothing to do.", flush=True)

    if world_size > 1:
        dist.barrier()
    pairs = broadcast_pairs(pairs, rank, world_size)
    if not pairs:
        if world_size > 1:
            dist.destroy_process_group()
        return 0

    counter = make_counter_store(rank, world_size, args.counter_port)
    if world_size > 1:
        dist.barrier()

    n_done = run_worker(
        args, rank, world_size, pairs, counter, success_log, failed_log
    )

    if world_size > 1:
        dist.barrier()
        if rank == 0:
            success_n = sum(1 for _ in open(success_log)) if os.path.exists(success_log) else 0
            failed_n = sum(1 for _ in open(failed_log)) if os.path.exists(failed_log) else 0
            print(
                f"[rank 0] all ranks done. cumulative success={success_n} "
                f"failed={failed_n} (logs in {args.output_dir}).",
                flush=True,
            )
        dist.destroy_process_group()

    return 0 if n_done >= 0 else 1


if __name__ == "__main__":
    sys.exit(main())
