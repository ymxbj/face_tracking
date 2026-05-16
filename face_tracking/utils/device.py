"""Lightweight device helpers (drop-in replacement for ``pt_core.get_device``)."""


import torch


def get_device() -> torch.device:
    """Return the current default device, with the index pinned for CUDA."""
    if torch.cuda.is_available():
        return torch.device(f"cuda:{torch.cuda.current_device()}")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_device_str() -> str:
    """Return the device as a string, always including ``:<idx>`` for CUDA.

    Example: ``"cuda:0"`` (not just ``"cuda"``) — callers that parse the
    suffix (e.g. nvdiffrast bootstrap) rely on the index being present.
    """
    return str(get_device())
