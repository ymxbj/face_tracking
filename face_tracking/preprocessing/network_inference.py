"""Run the Pixel3DMM normal-prediction network on cropped frames.

The current tracker only consumes the per-pixel normal map; the original
Pixel3DMM UV-map prediction path is not used and has been removed.
"""


import os
from typing import Optional

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm

from face_tracking import env_paths
from face_tracking.lightning.p3dmm_system import system as p3dmm_system
from face_tracking.utils.device import get_device


def _gaussian_fn(M: int, std: float) -> torch.Tensor:
    n = torch.arange(0, M) - (M - 1.0) / 2.0
    sig2 = 2 * std * std
    return torch.exp(-(n ** 2) / sig2)


def _gkern(kernlen: int = 256, std: float = 128.0) -> torch.Tensor:
    g_x = _gaussian_fn(kernlen, std=std * 5)
    g_y = _gaussian_fn(kernlen, std=std)
    return torch.outer(g_y, g_x)


def _pad_to_3_channels(img: np.ndarray) -> np.ndarray:
    if img.shape[-1] == 3:
        return img
    if img.shape[-1] == 1:
        return np.concatenate([img, np.zeros_like(img[..., :1]), np.zeros_like(img[..., :1])], axis=-1)
    if img.shape[-1] == 2:
        return np.concatenate([img, np.zeros_like(img[..., :1])], axis=-1)
    raise ValueError("Unexpected number of prediction channels.")


_NORMAL_MODEL: Optional[torch.nn.Module] = None


def _initialize_normal_network() -> torch.nn.Module:
    """Lazily build and cache the normal-prediction network."""
    global _NORMAL_MODEL
    if _NORMAL_MODEL is not None:
        return _NORMAL_MODEL

    ckpt_path = env_paths.CKPT_N_PRED
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Could not find normal-prediction checkpoint at {ckpt_path}. "
            "Run scripts/download_weights.sh or set FACE_TRACKING_PRETRAINED."
        )

    print("Loading normal-prediction network ...")
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    hyper_parameters = checkpoint.get("hyper_parameters", None)
    model_cfg = hyper_parameters.get("cfg", None)

    model = p3dmm_system(model_cfg)
    state_dict = checkpoint.get("state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=False)
    model = model.to(get_device())

    _NORMAL_MODEL = model
    return _NORMAL_MODEL


def run(video_name: str, batch_size: Optional[int] = None) -> None:
    """Predict per-pixel surface normals for every cropped frame.

    Parameters
    ----------
    video_name
        Logical video name. Cropped frames are read from
        ``preprocessed_data/<video_name>/cropped`` and the predictions
        are written under ``preprocessed_data/<video_name>/p3dmm/normals``.
    batch_size
        Inference batch size. Defaults to the value in ``configs/base.yaml``.
    """
    cfg = OmegaConf.load(env_paths.get_base_yaml())
    if batch_size is None:
        batch_size = cfg.network_batch_size

    model = _initialize_normal_network()
    device = get_device()

    folder = os.path.join(env_paths.PREPROCESSED_DATA, video_name)
    image_folder = os.path.join(folder, "cropped")
    seg_folder = os.path.join(folder, "seg_og")
    out_folder = os.path.join(folder, "p3dmm", "normals")
    os.makedirs(out_folder, exist_ok=True)

    image_names = sorted(os.listdir(image_folder))

    if len(os.listdir(out_folder)) == len(image_names):
        print(f"<<<<<<<< ALREADY COMPLETED NORMAL INFERENCE for {video_name}, SKIPPING >>>>>>>>")
        return

    print(f"<<<<<<<< STARTING NORMAL INFERENCE for {video_name} >>>>>>>>")

    n_iter = (len(image_names) + batch_size - 1) // batch_size

    for num_iter in tqdm(range(n_iter), desc=f"p3dmm[normals]"):
        masks, imgs = [], []
        for i_batch in range(batch_size):
            index = num_iter * batch_size + i_batch
            if index >= len(image_names):
                break

            img = np.array(Image.open(os.path.join(image_folder, image_names[index])).resize((512, 512))) / 255.0
            img = torch.from_numpy(img)[None, None].float().to(device)

            seg_name = image_names[index][:-4] + ".png"
            img_seg = np.array(Image.open(os.path.join(seg_folder, seg_name)).resize((512, 512), Image.NEAREST))
            if img_seg.ndim == 3:
                img_seg = img_seg[..., 0]
            mask = ((img_seg == 2) | ((img_seg > 3) & (img_seg < 14))) & ~(img_seg == 11)
            mask = torch.from_numpy(mask).long().to(device)[None, None]

            masks.append(mask)
            imgs.append(img)

        masks = torch.cat(masks, dim=0)
        imgs = torch.cat(imgs, dim=0)

        batch = {"tar_msk": masks, "tar_rgb": imgs}
        batch_mirrored = {
            "tar_rgb": torch.flip(batch["tar_rgb"], dims=[3]).to(device),
            "tar_msk": torch.flip(batch["tar_msk"], dims=[3]).to(device),
        }

        with torch.no_grad():
            output, _ = model.net(batch)
            output_mirrored, _ = model.net(batch_mirrored)

            if "normals" in output:
                fliped = torch.flip(output_mirrored["normals"], dims=[4])
                fliped[:, :, 0, :, :] *= -1
                output["normals"] = (output["normals"] + fliped) / 2

        for i_batch in range(batch_size):
            index = num_iter * batch_size + i_batch
            if index >= len(image_names):
                break

            i_view = 0
            tmp = output["normals"][i_batch, i_view]
            tmp = tmp / torch.norm(tmp, dim=0).unsqueeze(0)
            tmp = torch.clamp((tmp + 1) / 2, 0, 1)
            tmp = torch.stack([tmp[0, ...], 1 - tmp[2, ...], 1 - tmp[1, ...]], dim=0)

            arr = (
                _pad_to_3_channels(tmp.permute(1, 2, 0).detach().cpu().float().numpy() * 255)
                .astype(np.uint8)
            )
            Image.fromarray(arr).save(os.path.join(out_folder, image_names[index][:-4] + ".png"))

    print(f"<<<<<<<< FINISHED NORMAL INFERENCE for {video_name} >>>>>>>>")
