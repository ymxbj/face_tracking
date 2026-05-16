"""Run SegFace face parsing on the cropped frames of a video."""


import os
from typing import Optional

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm

from face_tracking import env_paths
from face_tracking.preprocessing.SegFace import FaceParserInference
from face_tracking.utils.device import get_device


_face_parser_instance: Optional[FaceParserInference] = None


def get_face_parser(
    model_path: Optional[str] = None,
    swin_model_path: Optional[str] = None,
) -> FaceParserInference:
    """Initialize (and cache) the SegFace face parser."""
    global _face_parser_instance
    if _face_parser_instance is not None:
        return _face_parser_instance

    model_path = model_path or env_paths.CKPT_SEGFACE
    swin_model_path = swin_model_path or env_paths.CKPT_SWIN_BACKBONE

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Could not find SegFace checkpoint at {model_path}. "
            "Run scripts/download_weights.sh or set FACE_TRACKING_PRETRAINED."
        )
    if not os.path.exists(swin_model_path):
        raise FileNotFoundError(
            f"Could not find Swin backbone weights at {swin_model_path}. "
            "Run scripts/download_weights.sh or set FACE_TRACKING_PRETRAINED."
        )

    print("Initializing SegFace face parser ...")
    _face_parser_instance = FaceParserInference(
        model_path=model_path,
        swin_model_path=swin_model_path,
        device=get_device(),
    )
    return _face_parser_instance


def run(video_name: str, batch_size: Optional[int] = None) -> None:
    """Generate per-frame face parsing masks under ``preprocessed_data/<name>/seg_og/``."""
    cfg = OmegaConf.load(env_paths.get_base_yaml())
    if batch_size is None:
        batch_size = cfg.segmentation_batch_size

    device = get_device()
    face_parser = get_face_parser()

    out = os.path.join(env_paths.PREPROCESSED_DATA, video_name)
    out_seg = os.path.join(out, "seg_og")
    folder = os.path.join(out, "cropped")

    os.makedirs(out_seg, exist_ok=True)

    frames = [f for f in os.listdir(folder) if f.endswith((".png", ".jpg"))]
    frames.sort()

    if len(os.listdir(out_seg)) == len(frames):
        print(f"<<<<<<<< ALREADY COMPLETED SEGMENTATION FOR {video_name}, SKIPPING >>>>>>>>")
        return

    num_iter = (len(frames) + batch_size - 1) // batch_size

    for i in tqdm(range(num_iter), desc=f"SegFace[{video_name}]"):
        image_stack = []
        frame_stack = []
        for j in range(batch_size):
            index = i * batch_size + j
            if index >= len(frames):
                break

            file = frames[index]
            img = Image.open(os.path.join(folder, file))
            img = np.array(img)[..., :3]
            image = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device).float() / 255.0

            image_stack.append(image)
            frame_stack.append(file[:-4])

        image_batch = torch.cat(image_stack, dim=0)

        with torch.inference_mode():
            _, preds = face_parser.batch_predict(image_batch)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        for k in range(preds.shape[0]):
            frame = frame_stack[k]
            Image.fromarray(preds[k]).save(os.path.join(out_seg, f"{frame}.png"))

    print("Finish segmentation")
