"""Path configuration for the open-source ``face_tracking`` package.

All paths can be overridden via environment variables so that the package
remains usable without writing to the source tree:

* ``FACE_TRACKING_ROOT``                — repository root (default: parent of this file's parent).
* ``FACE_TRACKING_PREPROCESSED_DATA``   — where intermediate per-video artefacts are stored.
* ``FACE_TRACKING_OUTPUT``              — where the final per-video tracking outputs are saved.
* ``FACE_TRACKING_PRETRAINED``          — directory containing the downloaded model weights.
"""


import os

# ---------------------------------------------------------------------------
# Repository / asset roots
# ---------------------------------------------------------------------------

_THIS_FILE = os.path.abspath(__file__)
_PACKAGE_DIR = os.path.dirname(_THIS_FILE)
_DEFAULT_ROOT = os.path.dirname(_PACKAGE_DIR)

CODE_BASE = os.environ.get("FACE_TRACKING_ROOT", _DEFAULT_ROOT)

ASSETS_DIR = os.path.join(CODE_BASE, "assets")
CONFIG_DIR = os.path.join(CODE_BASE, "configs")

PREPROCESSED_DATA = os.environ.get(
    "FACE_TRACKING_PREPROCESSED_DATA",
    os.path.join(CODE_BASE, "preprocessed_data"),
)
TRACKING_OUTPUT = os.environ.get(
    "FACE_TRACKING_OUTPUT",
    os.path.join(CODE_BASE, "tracking_output"),
)
PRETRAINED_DIR = os.environ.get(
    "FACE_TRACKING_PRETRAINED",
    os.path.join(CODE_BASE, "pretrained_weights"),
)

# ---------------------------------------------------------------------------
# Static FLAME / pixel3dmm assets shipped with the repo
# ---------------------------------------------------------------------------

head_template = os.path.join(ASSETS_DIR, "head_template.obj")
head_template_color = os.path.join(ASSETS_DIR, "head_template_color.obj")

VERTEX_WEIGHT_MASK = os.path.join(ASSETS_DIR, "flame_vertex_weights.npy")
MIRROR_INDEX = os.path.join(ASSETS_DIR, "flame_mirror_index.npy")
EYE_MASK = os.path.join(ASSETS_DIR, "uv_mask_eyes.png")
FLAME_MASKS = os.path.join(ASSETS_DIR, "FLAME_masks.pkl")
FLAME_LANDMARK_EMBEDDING = os.path.join(ASSETS_DIR, "landmark_embedding.npy")

# ---------------------------------------------------------------------------
# Pretrained network weights
# ---------------------------------------------------------------------------

CKPT_N_PRED = os.path.join(PRETRAINED_DIR, "normals.ckpt")
CKPT_SEGFACE = os.path.join(PRETRAINED_DIR, "SegFace.pt")
CKPT_SWIN_BACKBONE = os.path.join(PRETRAINED_DIR, "swin_b-68c6b09e.pth")
DINO_BACKBONE_FILE = os.path.join(PRETRAINED_DIR, "vit_base_patch16_224.dino.safetensors")
CKPT_LANDMARK_ONNX = os.path.join(PRETRAINED_DIR, "landmark.onnx")
INSIGHTFACE_ROOT = os.path.join(PRETRAINED_DIR, "insightface")

# Directory in which the FLAME 3DMM assets (FLAME2020/, FLAME_masks/, ...) live.
FLAME_ASSETS = PRETRAINED_DIR


def get_tracking_yaml() -> str:
    return os.path.join(CONFIG_DIR, "tracking.yaml")


def get_base_yaml() -> str:
    return os.path.join(CONFIG_DIR, "base.yaml")
