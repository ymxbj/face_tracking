"""Inference-only Lightning wrapper around the Pixel3DMM normal-prediction network.

The original repository's ``system`` class also implemented training,
validation, optimizer configuration, and result visualisation. None of
those code paths are reachable from the open-source tracker (which only
calls ``model.net(batch)`` for inference), so they have been removed.
"""


import numpy as np
import pytorch_lightning as L
import torch

from face_tracking import env_paths
from face_tracking.lightning.p3dmm_network import Network


class system(L.LightningModule):
    """Inference-only wrapper used by ``preprocessing.network_inference``."""

    def __init__(self, cfg):
        super().__init__()

        self.glctx = None
        self.cfg = cfg
        self.net = Network(cfg)

        vertex_weight_mask = np.load(f"{env_paths.VERTEX_WEIGHT_MASK}")
        self.register_buffer(
            "vertex_weight_mask", torch.from_numpy(vertex_weight_mask).float()
        )

        # Kept for checkpoint hyper-parameter persistence.
        self.save_hyperparameters()
