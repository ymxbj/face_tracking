"""Visualize tracked keypoints on top of cropped frames and save as a video."""


import os

import cv2
import numpy as np

try:
    from moviepy import ImageSequenceClip  # moviepy >= 2.x
except ImportError:  # pragma: no cover
    from moviepy.editor import ImageSequenceClip  # moviepy 1.x

from face_tracking import env_paths


_mirror_index = np.load(env_paths.MIRROR_INDEX)

# FLAME vertex indices used for the 40-point keypoint visualisation
flame_lip_index = [2840, 2892, 3509, 1789, 1723, 1740, 3533, 2855]
flame_left_eyebrow_index = [3763, 336, 3153, 3705, 2178, 2177]
flame_right_eyebrow_index = [_mirror_index[i].item() for i in flame_left_eyebrow_index]
flame_left_eye_index = [2495]
flame_right_eye_index = [1344]
flame_left_iris_index = [4597]
flame_right_iris_index = [4051]
flame_left_face_index = [3710, 3743, 3116, 3467, 3465]
flame_right_face_index = [3866, 3881, 2081, 3717, 3715]
flame_nose_index = [3093, 3551, 2058]
flame_contour_index = [3408, 3404, 3624]

selected_indices_in_5023 = (
    flame_lip_index
    + flame_left_eyebrow_index
    + flame_right_eyebrow_index
    + flame_left_eye_index
    + flame_right_eye_index
    + flame_left_face_index
    + flame_right_face_index
    + flame_nose_index
    + flame_contour_index
)

flame_mesh_mask = np.load(env_paths.FLAME_MASKS, allow_pickle=True, encoding="latin1")
vertex_face = flame_mesh_mask["face"].tolist()
selected_indices_wo_iris = [vertex_face.index(idx) for idx in selected_indices_in_5023]
selected_indices = selected_indices_wo_iris + [-2, -1]


def vis_kp_video(preprocessed_dir: str, tracking_dir: str, video_save_path: str = "", fps: int = 25) -> str:
    """Render a side-by-side keypoint debug video to ``vis_kp.mp4``.

    Parameters
    ----------
    preprocessed_dir
        Folder containing the cropped frames (``cropped/``).
    tracking_dir
        Folder containing the tracking output (``key_points/*.npy``).
    video_save_path
        Optional explicit output path. If empty, defaults to
        ``{tracking_dir}/vis_kp.mp4``.
    fps
        Output video FPS.
    """

    kp_path = os.path.join(tracking_dir, "key_points", "kp.npy")
    exp_kp_path = os.path.join(tracking_dir, "key_points", "exp_kp.npy")

    kp = np.load(kp_path)[:, selected_indices, :]
    exp_kp = np.load(exp_kp_path)[:, selected_indices, :]

    image_dir = os.path.join(preprocessed_dir, "cropped")
    images = [f for f in os.listdir(image_dir) if f.lower().endswith((".jpg", ".png"))]
    images.sort(key=lambda x: int(x.split(".")[0]))

    frames = []
    for idx, image in enumerate(images):
        img = cv2.imread(os.path.join(image_dir, image))
        vis_image = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)

        vis_exp_kp = ((exp_kp[idx] + 1) * 256).astype(np.int32)
        vis_kp = ((kp[idx] + 1) * 256).astype(np.int32)

        cnv_xy_exp = np.full((512, 512, 3), 255, dtype=np.uint8)
        cnv_yz_exp = np.full((512, 512, 3), 255, dtype=np.uint8)

        for j in range(vis_exp_kp.shape[0]):
            cv2.circle(vis_image, (vis_kp[j, 0], vis_kp[j, 1]), 1, (255, 0, 0), 2)

            cv2.circle(cnv_xy_exp, (vis_exp_kp[j, 0], vis_exp_kp[j, 1] - 50), 1, (255, 0, 0), 2)
            cv2.circle(cnv_yz_exp, (vis_exp_kp[j, 2], vis_exp_kp[j, 1] - 50), 1, (255, 0, 0), 2)

        frames.append(np.concatenate([vis_image, cnv_xy_exp, cnv_yz_exp], axis=1))

    if not video_save_path:
        video_save_path = os.path.join(tracking_dir, "vis_kp.mp4")

    clip = ImageSequenceClip(frames, fps=fps)
    clip.write_videofile(video_save_path, codec="libx264")
    return video_save_path
