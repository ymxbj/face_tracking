import shutil

import mediapy
from PIL import Image
import os.path
from enum import Enum
from pathlib import Path
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import trimesh
from pytorch3d.io import load_obj
from tqdm import tqdm
from face_tracking.utils.device import get_device

import dreifus
from dreifus.matrix import Pose

from contextlib import nullcontext
from face_tracking import env_paths
from face_tracking.tracking import nvdiffrast_util
from face_tracking.tracking.renderer_nvdiffrast import NVDRenderer
from face_tracking.tracking.flame.FLAME import FLAME
from face_tracking.utils.utils_3d import rotation_6d_to_matrix, matrix_to_rotation_6d, euler_angles_to_matrix
from face_tracking.utils.vis_kp import vis_kp_video
from .opt_pre import FlameForwardMixin
from .opt_post import ComputeLoss

DEBUG = False

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
rank = 42
torch.manual_seed(rank)
torch.cuda.manual_seed(rank)
cudnn.benchmark = True
np.random.seed(rank)
I = torch.eye(3)[None].to(get_device()).detach()
I6D = matrix_to_rotation_6d(I)
                    
torch.set_float32_matmul_precision('high')

class View(Enum):
    GROUND_TRUTH = 1
    COLOR_OVERLAY = 2
    SHAPE_OVERLAY = 4
    SHAPE = 8
    LANDMARKS = 16
    HEATMAP = 32
    DEPTH = 64

def get_intrinsics(focal_length, principal_point, use_hack : bool = True, size : int = 512):
    
    # use_hack: 对相机内参矩阵中的主点x坐标进行翻转处理，以适应OpenGL的坐标系

    device = focal_length.device
    intrinsics = torch.eye(3)[None, ...].float().to(device).repeat(focal_length.shape[0], 1,1 )
    intrinsics[:, 0, 0] = focal_length.squeeze() * size
    intrinsics[:, 1, 1] = focal_length.squeeze() * size
    intrinsics[:, :2, 2] = size/2+0.5 + principal_point * (size/2+0.5)

    if use_hack:
        intrinsics[:, 0:1, 2:3] = size - intrinsics[:, 0:1, 2:3]  # TODO fix this hack

    return intrinsics

def get_extrinsics(R_base, t_base):
    timestep = 0
    device = R_base[timestep].device
    w2c_openGL = torch.eye(4)[None, ...].float().to(device)
    w2c_openGL[:, :3, :3] = R_base[timestep]
    w2c_openGL[:, :3, 3] = t_base[timestep]
    return w2c_openGL

COMPILE = True

class NullProfiler:
    """一个空的 Profiler，用于在不进行性能分析时替代 torch.profiler.profile。"""
    def __enter__(self):
        # 必须返回一个对象，这个对象要有 step 方法
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # 退出上下文时什么也不做
        pass

    def step(self):
        # 调用 prof.step()时什么也不做
        pass

class Tracker(FlameForwardMixin, ComputeLoss):
    def __init__(self, flame_assets: str | None = None):
        """Tracker entry point.

        Parameters
        ----------
        flame_assets
            Directory containing the FLAME 2020 assets
            (``FLAME2020/`` and ``FLAME_masks/``). If ``None``,
            falls back to :data:`face_tracking.env_paths.FLAME_ASSETS`.
        """
        self.device = get_device()
        self.FRAME_SKIP = 1
        self._flame_assets_override = flame_assets

        self.mirror_order = torch.from_numpy(np.load(f'{env_paths.MIRROR_INDEX}')).long().to(self.device)
        self.mirror = np.load(f'{env_paths.MIRROR_INDEX}')        

        # 参与loss计算的点在5023 mesh里的indices
        # lip [0, 8] 从左嘴角点开始，逆时针方向数8个点
        self.flame_lip_loss = [2840, 2892, 3509, 1789, 1723, 1740, 3533, 2855]
        # left eyebrow [8, 13]
        self.flame_left_eyebrow_loss = [3763, 336, 3153, 3705, 2178]
        # right eyebrow [13, 18]
        self.flame_right_eyebrow_loss = [self.mirror[i].item() for i in self.flame_left_eyebrow_loss]
        # left eye [18, 22]
        self.flame_left_eye_loss = [2437, 2495, 3619, 2355]
        # right eye [22, 26]
        self.flame_right_eye_loss = [self.mirror[i].item() for i in self.flame_left_eye_loss]
        # left iris [26, 27]
        self.flame_left_iris_loss = [4597]
        # right iris [27, 28]
        self.flame_right_iris_loss = [4051]
        # left nose [28, 34]
        self.flame_left_nose_loss = [3151, 2759, 3094, 3092, 3608, 2794]
        # right nose [34, 40]
        self.flame_right_nose_loss = [self.mirror[i].item() for i in self.flame_left_nose_loss[::-1]]
        # column nose [40, 44]
        self.flame_column_nose_loss = [3553, 3521, 3526, 3551]

        self.lmk_loss_in_flame_list = [self.flame_lip_loss, 
                                      self.flame_left_eyebrow_loss, self.flame_right_eyebrow_loss, 
                                      self.flame_left_eye_loss, self.flame_right_eye_loss, 
                                      self.flame_left_iris_loss, self.flame_right_iris_loss,
                                      self.flame_left_nose_loss, self.flame_right_nose_loss, self.flame_column_nose_loss]
    
        self.lip_len = len(self.flame_lip_loss)

        self.left_eyebrow_len = len(self.flame_left_eyebrow_loss)
        self.right_eyebrow_len = len(self.flame_right_eyebrow_loss)
        self.eyebrow_len = self.left_eyebrow_len + self.right_eyebrow_len

        self.left_eye_len = len(self.flame_left_eye_loss)
        self.right_eye_len = len(self.flame_right_eye_loss)
        self.eye_len = self.left_eye_len + self.right_eye_len

        self.left_iris_len = len(self.flame_left_iris_loss)
        self.right_iris_len = len(self.flame_right_iris_loss)
        self.iris_len = self.left_iris_len + self.right_iris_len

        self.left_nose_len = len(self.flame_left_nose_loss)
        self.right_nose_len = len(self.flame_right_nose_loss)
        self.column_nose_len = len(self.flame_column_nose_loss)
        self.nose_len = self.left_nose_len + self.right_nose_len + self.column_nose_len

        self.contour_side_num = 8

        self.left_contour_len = self.contour_side_num
        self.right_contour_len = self.contour_side_num

        self.contour_len = self.contour_side_num * 2 + 1

        # [0, 1, 7] + \

        self.left_face_indices = list(range(self.lip_len, self.lip_len + self.left_eyebrow_len)) + \
                            list(range(self.lip_len + self.eyebrow_len, self.lip_len + self.eyebrow_len + self.left_eye_len)) + \
                            list(range(self.lip_len + self.eyebrow_len + self.eye_len, self.lip_len + self.eyebrow_len + self.eye_len + self.left_iris_len)) + \
                            list(range(self.lip_len + self.eyebrow_len + self.eye_len + self.iris_len, self.lip_len + self.eyebrow_len + self.eye_len + self.iris_len + self.left_nose_len)) + \
                            list(range(self.lip_len + self.eyebrow_len + self.eye_len + self.iris_len + self.nose_len, self.lip_len + self.eyebrow_len + self.eye_len + self.iris_len + self.nose_len + self.left_contour_len))

        # [3, 4, 5] + \

        self.right_face_indices = list(range(self.lip_len + self.left_eyebrow_len, self.lip_len + self.eyebrow_len)) + \
                            list(range(self.lip_len + self.eyebrow_len + self.left_eye_len, self.lip_len + self.eyebrow_len + self.eye_len)) + \
                            list(range(self.lip_len + self.eyebrow_len + self.eye_len + self.left_iris_len, self.lip_len + self.eyebrow_len + self.eye_len + self.iris_len)) + \
                            list(range(self.lip_len + self.eyebrow_len + self.eye_len + self.iris_len + self.left_nose_len, self.lip_len + self.eyebrow_len + self.eye_len + self.iris_len + self.left_nose_len + self.right_nose_len)) + \
                            list(range(self.lip_len + self.eyebrow_len + self.eye_len + self.iris_len + self.nose_len + self.left_contour_len + 1, self.lip_len + self.eyebrow_len + self.eye_len + self.iris_len + self.nose_len + self.contour_len))

        # 参与loss计算的点在lmk203里的indices
        # lip [0, 8] 从左嘴角点开始，逆时针方向数8个点
        self.lmk_lip_index = [84, 106, 102, 98, 96, 94, 90, 86]
        # 左侧的嘴：84, 106, 86
        # left eyebrow [8, 13]
        self.lmk_left_up_eyebrow_index = [145, 148, 150, 152, 155] 
        self.lmk_left_bottom_eyebrow_index = [145, 162, 160, 158, 155]
        # right eyebrow [13, 18]
        self.lmk_right_up_eyebrow_index = [175, 172, 170, 168, 165] 
        self.lmk_right_bottom_eyebrow_index = [175, 178, 180, 182, 165]
        # left eye [18, 22]
        self.lmk_left_eye_index = [0, 6, 12, 18] 
        # right eye [22, 26]
        self.lmk_right_eye_index = [36, 30, 24, 42]
        # left iris [26, 27]
        self.lmk_left_iris_index = [197]
        # right iris [27, 28]
        self.lmk_right_iris_index = [198]
        # left nose [28, 34]
        self.lmk_left_nose_index = [185, 186, 187, 188, 189, 190]
        # right nose [34, 40]
        self.lmk_right_nose_index = [191, 192, 193, 194, 195, 196]
        # column nose [40, 44]
        self.lmk_column_nose_index = [199, 200, 201, 202]
        # contour [44, 47]

        self.all_lmk_contour_index = [109, 111, 113, 115, 117, 119, 121, 124, 126, 128, 131, 133, 135, 137, 139, 141, 143]

        mid_index = len(self.all_lmk_contour_index) // 2
        self.lmk_middle_point = self.all_lmk_contour_index[mid_index]

        self.lmk_left_column_index = self.all_lmk_contour_index[mid_index - self.contour_side_num: mid_index]
        self.lmk_right_column_index = self.all_lmk_contour_index[mid_index + 1: mid_index + 1 + self.contour_side_num]

        self.lmk_contour_index = self.lmk_left_column_index + [self.lmk_middle_point] + self.lmk_right_column_index

        # self.lmk_contour_index = [122, 126, 130]

        # 40点在5023 mesh里的indices
        # lip [0,8]
        self.flame_lip_index = [2840, 2892, 3509, 1789, 1723, 1740, 3533, 2855]
        # left eyebrow [8, 14]
        self.flame_left_eyebrow_index = [3763, 336, 3153, 3705, 2178, 2177]
        # right eyebrow [14, 20]
        self.flame_right_eyebrow_index = [self.mirror[i].item() for i in self.flame_left_eyebrow_index]
        # left eye [20, 21]
        self.flame_left_eye_index = [2495]
        # right eye [21, 22]
        self.flame_right_eye_index = [1344]
        # left iris [22, 23]
        self.flame_left_iris_index = [4597]
        # right iris [23, 24]
        self.flame_right_iris_index = [4051]
        # left face [24, 29]
        self.flame_left_face_index = [3710, 3743, 3116, 3467, 3465]
        # right face [29, 34]
        self.flame_right_face_index = [3866, 3881, 2081, 3717, 3715]
        # nose [34, 37]
        self.flame_nose_index = [3093, 3551, 2058]
        # contour [37, 40]
        self.flame_contour_index = [3408, 3404, 3624]

        self.selected_indices_in_flame = (self.flame_lip_index + 
                    self.flame_left_eyebrow_index + self.flame_right_eyebrow_index + 
                    self.flame_left_eye_index + self.flame_right_eye_index + 
                    self.flame_left_iris_index + self.flame_right_iris_index +
                    self.flame_left_face_index + self.flame_right_face_index +
                    self.flame_nose_index + self.flame_contour_index)
        
        self.debug_dir = os.path.join(env_paths.CODE_BASE, 'vis_debug')
        os.makedirs(self.debug_dir, exist_ok=True)

        self.global_step = 0

        # Latter will be set up
        self.frame = 0
        self.is_initializing = False

        self.cam_pose_nvd = {}
        self.R_base = {}
        self.t_base = {}

        self.flame_assets = self._flame_assets_override or env_paths.FLAME_ASSETS

        flame_masks_path = os.path.join(self.flame_assets, 'FLAME2020', 'FLAME_masks', 'FLAME_masks.pkl')
        if not os.path.exists(flame_masks_path):
            # Fallback to the lightweight copy that ships with this repo.
            flame_masks_path = env_paths.FLAME_MASKS
        flame_mesh_mask = np.load(flame_masks_path, allow_pickle=True, encoding='latin1')
        
        self.vertex_face_mask = torch.from_numpy(flame_mesh_mask['face']).to(self.device).long()

        self.vertex_face = flame_mesh_mask['face'].tolist()
        self.vertex_face = self.vertex_face + self.flame_left_iris_index + self.flame_right_iris_index
    
    def initialize(self, config):
        self.config = config
        self.actor_name = self.config.video_name  # actor name就是video name
        DATA_FOLDER = f'{env_paths.PREPROCESSED_DATA}/{self.actor_name}'
        self.MAX_STEPS = min(len([f for f in os.listdir(f'{DATA_FOLDER}/cropped/') if f.endswith('.jpg') or f.endswith('.png')]) - self.config.start_frame, 1000)
        self.fps = self.config.fps

        print(f'''
                <<<<<<<< INITIALIZING TRACKER INSTANCE FOR {self.actor_name} >>>>>>>>
                ''')
        
        self.BATCH_SIZE = self.config.batch_size

        self.no_sh = self.config.no_sh
        self.no_lm = self.config.no_lm
        self.no_pho = self.config.no_pho

        self.image_size = torch.tensor([[self.config.image_size[0], self.config.image_size[1]]]).to(self.device)

        self.global_step = 0

        # Latter will be set up
        self.frame = 0
        self.is_initializing = False
        if hasattr(self.config, 'output_folder'):
            self.save_folder = self.config.output_folder
        else:
            self.save_folder = env_paths.TRACKING_OUTPUT   # save_folder就是tracking_output
        self.output_folder = os.path.join(self.save_folder, self.actor_name)   # output_folder就是在tracking_output下建立视频目录
        self.checkpoint_folder = os.path.join(self.save_folder, self.actor_name, "checkpoint")   # 在视频目录下新建checkpoint目录
        self.mesh_folder = os.path.join(self.save_folder, self.actor_name, "mesh")   # 在视频目录下新建mesh目录
        self.exp_folder = os.path.join(self.mesh_folder, 'exp_mesh')
        self.kp_folder = os.path.join(self.mesh_folder, 'kp_mesh')
        self.kp_folder_eye = os.path.join(self.save_folder, self.actor_name, "key_points")   # 在视频目录下新建kp_eye目录
        self.create_output_folders()

        self.cam_pose_nvd = {}
        self.R_base = {}
        self.t_base = {}

        self.setup_renderer()

        self.intermediate_exprs = []
        self.intermediate_Rs = []
        self.intermediate_ts = []
        self.intermediate_eyes = []
        self.intermediate_eyelids = []
        self.intermediate_jaws = []
        self.intermediate_necks = []
        self.intermediate_fls = []
        self.intermediate_pps = []

        self.cached_data = {}

    def get_image_size(self):
        return self.image_size[0][0].item(), self.image_size[0][1].item()

    def create_output_folders(self):
        Path(self.save_folder).mkdir(parents=True, exist_ok=True)
        Path(self.output_folder).mkdir(parents=True, exist_ok=True)
        Path(self.checkpoint_folder).mkdir(parents=True, exist_ok=True)
        Path(self.mesh_folder).mkdir(parents=True, exist_ok=True)
        Path(self.exp_folder).mkdir(parents=True, exist_ok=True)
        Path(self.kp_folder).mkdir(parents=True, exist_ok=True)
        Path(self.kp_folder_eye).mkdir(parents=True, exist_ok=True)

    def setup_renderer(self):
        mesh_file = f'{env_paths.head_template}'
        self.config.image_size = self.get_image_size()
        self.flame = FLAME(self.config, self.flame_assets, self.contour_side_num, self.lmk_loss_in_flame_list).to(self.device)
        self.flame.vertex_face_mask = self.vertex_face_mask

        if COMPILE:
            print('Start compiling flame module')
            self.flame = torch.compile(self.flame)
            self.project_points_screen_space = torch.compile(self.project_points_screen_space)
            self.flame_forward = torch.compile(self.flame_forward)
            self.flame_forward_global = torch.compile(self.flame_forward_global)
            self.compute_loss = torch.compile(self.compute_loss)
            self.actual_smooth = torch.compile(self.actual_smooth)

        self.diff_renderer = NVDRenderer(self.config.size,
                                         obj_filename=mesh_file,
                                         no_sh=self.no_sh,
                                         white_bg= True,
                                         flame_assets=self.flame_assets).to(self.device)

        self.faces = load_obj(mesh_file)[1]
    
    def save_meshes(self, frame_id, vertices, vertices_noneck, vertices_can):
        f = self.diff_renderer.faces[0].cpu().numpy()
        bs = vertices.shape[0]
        sphere_radius = 0.001
        for b_i in range(bs):
            v_noneck = vertices_noneck[b_i].cpu().numpy()
            v = vertices[b_i].cpu().numpy()

            exp_mesh = trimesh.Trimesh(faces=f, vertices=v_noneck, process=False)
            sphere_meshes = []
            for idx in self.selected_indices_in_flame:
                vertex_pos = v_noneck[idx]

                sphere = trimesh.creation.icosphere(subdivisions=2, radius=sphere_radius)
                sphere.apply_translation(vertex_pos)

                sphere.visual.vertex_colors = [255, 0, 0, 255]

                sphere_meshes.append(sphere)
            
            exp_mesh.export(f'{self.exp_folder}/{frame_id:05d}.obj')
            exp_mesh = trimesh.util.concatenate([exp_mesh] + sphere_meshes)
            exp_mesh.export(f'{self.exp_folder}/{frame_id:05d}_keypoints.obj')

            kp_mesh = trimesh.Trimesh(faces=f, vertices = v, process=False)
            sphere_meshes = []
            for idx in self.selected_indices_in_flame:
                vertex_pos = v[idx]

                sphere = trimesh.creation.icosphere(subdivisions=2, radius=sphere_radius)
                sphere.apply_translation(vertex_pos)

                sphere.visual.vertex_colors = [255, 0, 0, 255]

                sphere_meshes.append(sphere)
            
            kp_mesh.export(f'{self.kp_folder}/{frame_id:05d}.obj')
            kp_mesh = trimesh.util.concatenate([kp_mesh] + sphere_meshes)
            kp_mesh.export(f'{self.kp_folder}/{frame_id:05d}_keypoints.obj')
        
        if frame_id == 0:
            v= vertices_can[0].detach().cpu().numpy()
            f = self.diff_renderer.faces[0].cpu().numpy()
            canonical_mesh = trimesh.Trimesh(faces=f, vertices=v, process=False)
            sphere_meshes = []
            for idx in self.selected_indices_in_flame:
                # 获取顶点位置
                vertex_pos = v[idx]
                
                # 创建小球
                sphere = trimesh.creation.icosphere(subdivisions=2, radius=sphere_radius)
                sphere.apply_translation(vertex_pos)
                
                # 设置球的颜色为红色
                sphere.visual.vertex_colors = [255, 0, 0, 255]
                
                sphere_meshes.append(sphere)

            # 合并原始网格和所有小球
            canonical_mesh.export(f'{self.mesh_folder}/canonical.obj')
            canonical_mesh = trimesh.util.concatenate([canonical_mesh] + sphere_meshes)
            canonical_mesh.export(f'{self.mesh_folder}/canonical_keypoints.obj')

    def save_checkpoint(self, frame_id, selected_frames = None):

        if selected_frames is None:
            exp = self.exp
            eyes = self.eyes
            eyelids = self.eyelids
            R = self.R
            t = self.t
            jaw = self.jaw
            neck = self.neck
            focal_length = self.focal_length
            principal_point = self.principal_point
        else:
            exp = self.exp(selected_frames)
            eyes = self.eyes(selected_frames)
            eyelids = self.eyelids(selected_frames)
            R = self.R(selected_frames)
            t = self.t(selected_frames)
            jaw = self.jaw(selected_frames)
            neck = self.neck(selected_frames)
            if self.config.global_camera:
                focal_length = self.focal_length
                principal_point = self.principal_point
            else:
                focal_length = self.focal_length(selected_frames)
                principal_point = self.principal_point(selected_frames)

        frame = {
            'flame': {
                'exp': exp.clone().detach().cpu().numpy(),
                'shape': self.shape.clone().detach().cpu().numpy(),
                'eyes': eyes.clone().detach().cpu().numpy(),
                'eyelids': eyelids.clone().detach().cpu().numpy(),
                'jaw': jaw.clone().detach().cpu().numpy(),
                'neck': neck.clone().detach().cpu().numpy(),
                'R': R.clone().detach().cpu().numpy(),
                'R_rotation_matrix': rotation_6d_to_matrix(R).detach().cpu().numpy(),
                't': t.clone().detach().cpu().numpy(),
            },
            'img_size': self.image_size.clone().detach().cpu().numpy()[0],
            'frame_id': frame_id,
            'global_step': self.global_step
        }

        cam_params = {
            f'R_base_{serial}': self.R_base[serial].clone().detach().cpu().numpy() for serial in self.R_base.keys()
        }
        cam_pos = {
                    f't_base_{serial}': self.t_base[serial].clone().detach().cpu().numpy() for serial in self.R_base.keys()
                }
        intr = {
                    'fl': focal_length.clone().detach().cpu().numpy(),
                    'pp': principal_point.clone().detach().cpu().numpy(),
                    }
        cam_params.update(cam_pos)
        cam_params.update(intr)
        frame.update(
            {
                f'camera': cam_params
            }
        )
        bs = exp.shape[0]
        vertices, lmks, joint_transforms, vertices_can, vertices_noneck, y_rot_angle = self.flame(cameras=torch.inverse(self.R_base[0])[:1, ...].repeat(bs, 1, 1),
                   shape_params=self.shape[:1, ...].repeat(bs, 1),
                   expression_params=exp,
                   eye_pose_params=eyes,
                   jaw_pose_params=jaw,
                   neck_pose_params=neck,
                   rot_params_lmk_shift=R,
                   eyelid_params=eyelids,
        )

        vertices = torch.einsum('bny,bxy->bnx', vertices, rotation_6d_to_matrix(R.repeat_interleave(1, dim=0))) \
            + t.repeat_interleave(1, dim=0).unsqueeze(1)   # flame空间到世界空间（也就是在头部加上旋转矩阵和平移）

        frame.update(
            {
                f'joint_transforms': joint_transforms.detach().cpu().numpy(),
            }
        )

        for b_i in range(bs):

            torch.save(frame, f'{self.checkpoint_folder}/{frame_id:05d}.frame')

            selction_indx = np.array([36, 39, 42, 45, 33, 48, 54])
            _lmks = lmks[b_i].detach().squeeze().cpu().numpy()

            if self.config.save_landmarks:
                np.save(f'{self.mesh_folder}/landmarks_{frame_id}_{b_i}.npy', _lmks[selction_indx])
    
    def project_points_screen_space(self, points3d):

        B = points3d.shape[0]
        reps_extr = B if self.w2c_openGL.shape[0] == 1 else 1
        reps_intr = B if self.intrinsics_hack.shape[0] == 1 else 1
        # apply w2c transformation
        cam_space = torch.bmm(
            torch.cat([points3d, torch.ones_like(points3d[..., :1])], dim=-1),
            self.w2c_openGL.permute(0, 2, 1).repeat(reps_extr, 1, 1).detach())
    
        # project from cam_space to screen_space
        cam_space_prime = cam_space[..., :3] / -cam_space[..., [2]]  # 透视除法，唯一一个不是刚性的
        screen_space = (-1) * torch.bmm(cam_space_prime, self.intrinsics_hack.permute(0, 2, 1).repeat(reps_intr, 1, 1).detach())[..., :2]
        screen_space = torch.stack([self.config.size - 1 - screen_space[..., 0], screen_space[..., 1], cam_space[..., 2]], dim=-1)

        return screen_space

    def to_cuda(self, batch, unsqueeze=False):
        for key in batch.keys():
            if torch.is_tensor(batch[key]):
                batch[key] = batch[key].to(self.device)
                if unsqueeze:
                    batch[key] = batch[key][None]

        return batch

    def create_parameters(self, timestep):
        pose_mat = np.eye(4)
        pose_mat[2, 3] = -1

        opencv_w2c_pose = Pose(pose_mat, camera_coordinate_convention=dreifus.matrix.CameraCoordinateConvention.OPEN_CV)
        opencv_w2c_pose = opencv_w2c_pose.change_pose_type(dreifus.matrix.PoseType.CAM_2_WORLD)

        opencv_w2c_pose.look_at(np.zeros(3), np.array([0, 1, 0]))

        opencv_w2c_pose = opencv_w2c_pose.change_pose_type(dreifus.matrix.PoseType.WORLD_2_CAM)
        self.debug_pose_init = opencv_w2c_pose.change_pose_type(dreifus.matrix.PoseType.WORLD_2_CAM).copy()

        cam_pose = opencv_w2c_pose
        cam_pose = cam_pose.change_pose_type(dreifus.matrix.PoseType.CAM_2_WORLD)
        cam_pose_nvd = cam_pose.copy()
        cam_pose_nvd = cam_pose_nvd.change_camera_coordinate_convention(new_camera_coordinate_convention=dreifus.matrix.CameraCoordinateConvention.OPEN_GL)
        cam_pose_nvd = cam_pose_nvd.change_pose_type(dreifus.matrix.PoseType.WORLD_2_CAM)
        self.cam_pose_nvd[timestep] = torch.from_numpy(cam_pose_nvd.copy()).float().to(self.device)

        R = torch.from_numpy(cam_pose_nvd.get_rotation_matrix()).unsqueeze(0).to(self.device)
        T = torch.from_numpy(cam_pose_nvd.get_translation()).unsqueeze(0).to(self.device)
        R.requires_grad = True
        T.requires_grad = True

        self.R_base[timestep] = R
        self.t_base[timestep] = T

        init_f = 2000 * self.config.size/512
        self.focal_length = torch.tensor([[init_f/self.config.size]]).float().to(self.device)
        self.principal_point = torch.tensor([[0, 0]]).float().to(self.device)
        self.focal_length.requires_grad = True
        self.principal_point.requires_grad = True
        intrinsics = torch.tensor([[init_f, 0, self.config.size//2],
                               [0, init_f, self.config.size//2],
                               [0, 0, 1]]).float().to(self.device)
        proj_512 = nvdiffrast_util.intrinsics2projection(intrinsics,
                                          znear=0.1, zfar=10,
                                          width=self.config.size,
                                          height=self.config.size)

        self.r_mvps = {}
        for serial in self.cam_pose_nvd.keys():
            self.r_mvps[serial] = (proj_512 @ self.cam_pose_nvd[serial])[None, ...]

        n_timesteps = 1
        expression_params = np.zeros([n_timesteps, 100])
        jaw_params = np.zeros([n_timesteps, 3])
        neck_params = np.zeros([n_timesteps, 3])
        flame_R = torch.from_numpy(np.stack([np.eye(3) for _ in range(n_timesteps)], axis=0))  # [1, 3, 3]
        flame_t = torch.from_numpy(np.stack([np.zeros([3]) for _ in range(n_timesteps)], axis=0))   # [1, 3]
        self.R = nn.Parameter(matrix_to_rotation_6d(flame_R.float().to(self.device)))  # [1, 6]
        self.t = nn.Parameter(flame_t.float().to(self.device))   # [1, 3]

        self.expression_params = expression_params
        self.jaw_params = jaw_params.astype(np.float32)
        self.neck_params = neck_params.astype(np.float32)

        self.shape = nn.Parameter(torch.zeros(1, 300).to(self.device))

        self.texture_observation_mask = None

        self.exp = nn.Parameter(torch.from_numpy(self.expression_params[[0] + self.config.keyframes,..., :]).float().to(self.device))  # [1, 100]
        self.jaw = nn.Parameter(matrix_to_rotation_6d(euler_angles_to_matrix(torch.from_numpy(self.jaw_params[[0]+ self.config.keyframes,..., :]).to(self.device), 'XYZ')))  # [1, 6]
        self.neck = nn.Parameter(matrix_to_rotation_6d(euler_angles_to_matrix(torch.from_numpy(self.neck_params[[0]+ self.config.keyframes,..., :]).to(self.device), 'XYZ')))  # [1, 6]
        self.eyes = nn.Parameter(torch.cat([matrix_to_rotation_6d(I), matrix_to_rotation_6d(I)], dim=1).repeat(1+len(self.config.keyframes), 1) )  # [1,12]
        self.eyelids = nn.Parameter(torch.zeros(1+len(self.config.keyframes), 2).float().to(self.device))   # [1, 2]

    def parse_mask(self, ops, batch, visualization=False):
        result = ops['mask_images_rendering']

        if visualization:
            result = ops['mask_images']

        return result.detach()

    def clone_params_keyframes_all(self, freeze_id : bool = False, freeze_cam : bool = False,
                                   include_neck : bool = False):

        lr_scale = 1.0
        lr_scale_id_related = 1.0
        if freeze_id:
            lr_scale_id_related = 0.1

        params = [
            {'params': [self.exp], 'lr': self.config.lr_exp * lr_scale, 'name': ['exp']},  # 0.025
            {'params': [self.eyes], 'lr': 0.005 * lr_scale, 'name': ['eyes']},
            {'params': [self.eyelids], 'lr': 0.002 * lr_scale, 'name': ['eyelids']},
            {'params': [self.t], 'lr': self.config.lr_t * lr_scale, 'name': ['t']},
            {'params': [self.R], 'lr': self.config.lr_R * lr_scale, 'name': ['R']},
        ]
        #params.append({'params': [self.shape.clone())], 'lr': self.config.lr_id * lr_scale, 'name': ['shape']})
        if not freeze_id:
            params.append({'params': [self.shape], 'lr': self.config.lr_id * lr_scale, 'name': ['shape']})

        params.append({'params': [self.jaw], 'lr': self.config.lr_jaw * lr_scale, 'name': ['jaw']})
        if include_neck:
            params.append({'params': [self.neck], 'lr': self.config.lr_neck, 'name': ['neck']})

        #if not self.config.load_intr:
        if not freeze_cam:
            params.append({'params': [self.focal_length], 'lr': self.config.lr_f * lr_scale_id_related, 'name': ['focal_length']})
            params.append({'params': [self.principal_point], 'lr': self.config.lr_pp * lr_scale_id_related, 'name': ['principal_point']})

        return params

    def clone_params_keyframes_all_joint(self, freeze_id : bool = False, include_neck : bool = False):

        lr_scale = 1.0
        lr_scale_id_related = 1.0
        if freeze_id:
            lr_scale_id_related = 0.1
        params = [
            {'params': self.exp.parameters(), 'lr': self.config.lr_exp * lr_scale, 'name': ['exp']},  # 0.025
            {'params': self.eyes.parameters(), 'lr': 0.005 * lr_scale, 'name': ['eyes']},
            {'params': self.eyelids.parameters(), 'lr': 0.002 * lr_scale, 'name': ['eyelids']},
            {'params': self.t.parameters(), 'lr': self.config.lr_t * lr_scale, 'name': ['t']},
            {'params': self.R.parameters(), 'lr': self.config.lr_R * lr_scale, 'name': ['R']},
        ]

        params.append({'params': self.jaw.parameters(), 'lr': self.config.lr_jaw * lr_scale, 'name': ['jaw']})
        if include_neck:
            params.append({'params': self.neck.parameters(), 'lr':  self.config.lr_neck, 'name': ['jaw']})

        if not self.config.global_camera: # False
            params.append({'params': self.focal_length.parameters(), 'lr': self.config.lr_f * lr_scale_id_related,
                           'name': ['camera_params']})
            params.append({'params': self.principal_point.parameters(), 'lr': self.config.lr_pp * lr_scale_id_related,
                           'name': ['camera_params']})
        #params.append({'params': [self.shape], 'lr': self.config.lr_id * lr_scale * 1, 'name': ['shape']})
        return params

    def reduce_loss(self, losses):
        all_loss = 0.
        for key in losses.keys():
            all_loss = all_loss + losses[key]
        losses['all_loss'] = all_loss
        return all_loss
    
    @torch.compiler.disable
    def select_frame(self, iters, p):
        
        with torch.no_grad():
            all_frames = np.array(
                range(self.config.start_frame, self.MAX_STEPS + self.config.start_frame, self.FRAME_SKIP))

            if self.MAX_STEPS < self.BATCH_SIZE:
                selected_frames = all_frames

            else:
                if (p < int(iters * 0.15) and (p % 2 == 0)):
                    selected_frames = np.sort(np.random.choice(np.arange(len(all_frames)), size=self.BATCH_SIZE,
                                                                replace=False))  # 在all_frames中随机选择BATCH_SIZE个帧，然后进行排序
                    
                else:
                    start = np.min(all_frames)
                    end = np.max(all_frames)
                    rnd_start = np.random.randint(start, end)
                    assert (end - start) >= self.BATCH_SIZE + 1
                    assert self.BATCH_SIZE % 2 == 0
                    if rnd_start - self.BATCH_SIZE // 2 < 0:
                        rnd_start = self.BATCH_SIZE // 2
                    if rnd_start + self.BATCH_SIZE // 2 + 1 > end:
                        rnd_start = end - self.BATCH_SIZE // 2 + 1
                    selected_frames = np.array(
                        list(range(rnd_start - self.BATCH_SIZE // 2, rnd_start + self.BATCH_SIZE // 2)))  
                    
                    # 以rnd_start中间的帧为中心，选择前后各BATCH_SIZE//2个帧，总共BATCH_SIZE个帧作为selected_frames

            selected_frames_th = torch.from_numpy(selected_frames).long()
            batch = {k: self.cached_data[k][selected_frames_th, ...] for k in self.cached_data.keys()}
            images, used_landmarks = self.parse_landmarks(batch)

            normal_mask = batch["normal_mask"]
            normal_map = batch["normals"] if "normals" in batch else None

            num_views = len(self.R_base.keys())
            bs = batch['normals'].shape[0] * num_views

            image_lmk = used_landmarks

        return selected_frames, batch, bs, num_views, image_lmk, normal_map, normal_mask

    #TODO: could be improved by compiling all the actuall smooth loss stuff

    #@torch.compile
    def actual_smooth(self, variables, losses):
        reg_smooth_exp = (variables['exp'][:-1, :] - variables['exp'][1:, :]).square().mean()
        reg_smooth_eyes = (variables['eyes'][:-1, :] - variables['eyes'][1:, :]).square().mean()
        reg_smooth_eyelids = (variables['eyelids'][:-1, :] - variables['eyelids'][1:, :]).square().mean()
        reg_smooth_R = (variables['R'][:-1, :] - variables['R'][1:, :]).square().mean()
        reg_smooth_t = (variables['t'][:-1, :] - variables['t'][1:, :]).square().mean()
        reg_smooth_jaw = (variables['jaw'][:-1, :] - variables['jaw'][1:, :]).square().mean()
        reg_smooth_neck = (variables['neck'][:-1, :] - variables['neck'][1:, :]).square().mean()
        if not self.config.global_camera:
            reg_smooth_principal_point = (
                    variables['principal_point'][:-1, :] - variables['principal_point'][1:, :]).square().mean()
            reg_smooth_focal_length = (
                    variables['focal_length'][:-1, :] - variables['focal_length'][1:, :]).square().mean()
        else:
            reg_smooth_principal_point = torch.zeros_like(reg_smooth_jaw)
            reg_smooth_focal_length = torch.zeros_like(reg_smooth_jaw)
        losses['smooth/exp'] = reg_smooth_exp * self.config.reg_smooth_exp * self.config.reg_smooth_mult
        losses['smooth/eyes'] = reg_smooth_eyes * self.config.reg_smooth_eyes * self.config.reg_smooth_mult
        losses['smooth/eyelids'] = reg_smooth_eyelids * self.config.reg_smooth_eyelids * self.config.reg_smooth_mult
        losses['smooth/jaw'] = reg_smooth_jaw * self.config.reg_smooth_jaw * self.config.reg_smooth_mult
        losses['smooth/neck'] = reg_smooth_neck * self.config.reg_smooth_neck * self.config.reg_smooth_mult
        losses['smooth/R'] = reg_smooth_R * self.config.reg_smooth_R * self.config.reg_smooth_mult
        losses['smooth/t'] = reg_smooth_t * self.config.reg_smooth_t * self.config.reg_smooth_mult
        losses['smooth/principal_point'] = reg_smooth_principal_point * self.config.reg_smooth_pp * self.config.reg_smooth_mult
        losses['smooth/focal_length'] = reg_smooth_focal_length * self.config.reg_smooth_fl * self.config.reg_smooth_mult
        return losses

    @torch.compiler.disable
    def add_smooth_loss(self, losses, is_joint, p, iters, variables):
        if is_joint and self.config.smooth and ((p >= int(iters * 0.15) and (p % 2 == 1)) ):  # and p % 2 != 0 and False:
            losses = self.actual_smooth(variables, losses)

        return losses

    def optimize_color_first_step(self, batch, params_func, is_joint = False):
        
        iters = self.config.iters

        batch = self.to_cuda(batch)

        images, used_landmarks = self.parse_landmarks(batch)

        self.focal_length.requires_grad = True
        self.principal_point.requires_grad = True

        normal_mask = batch["normal_mask"]

        normal_map = batch["normals"] if "normals" in batch else None

        # print(params_func())
        optimizer = torch.optim.Adam(params_func())

        optimizer.zero_grad()

        num_views = len(self.R_base.keys())

        image_lmk = used_landmarks

        self.diff_renderer.reset()

        best_loss = np.inf

        n_steps_stagnant = 0
        stagnant_window_size = 10
        past_k_steps = np.array([100.0 for _ in range(stagnant_window_size)])

        iterator = tqdm(range(iters), desc='', leave=True, miniters=100)

        for p in iterator:

            self.intrinsics_hack = get_intrinsics(self.focal_length, self.principal_point, size=self.config.size)
            self.w2c_openGL = get_extrinsics(self.R_base, self.t_base).repeat(self.focal_length.shape[0], 1, 1)

            losses, vertices, vertices_noneck, proj_vertices, proj_lmk, variables, num_views = self.flame_forward(iters, p, image_lmk)
            
            if DEBUG == True:
                img_tensor = images[0].clone().detach().cpu()

                if img_tensor.min() < 0:
                    img_tensor = (img_tensor + 1) / 2.0
                img_tensor = torch.clamp(img_tensor, 0, 1)

                img_np = img_tensor.permute(1, 2, 0).numpy()
                img_np = (img_np * 255).astype(np.uint8)
                img_vis = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

                # 2. 处理关键点
                # proj_lmk 取前两维 (x, y)，忽略第三维深度
                pred_pts = proj_lmk[0, :, :2].detach().cpu().numpy()
                gt_pts = image_lmk[0].detach().cpu().numpy()

                # 绿色 (Green) 画 GT landmarks
                for pt in gt_pts:
                    cv2.circle(img_vis, (int(pt[0]), int(pt[1])), 2, (0, 255, 0), -1)
                
                # 红色 (Red) 画 Projected landmarks
                for pt in pred_pts:
                    cv2.circle(img_vis, (int(pt[0]), int(pt[1])), 2, (0, 0, 255), -1)

                # 4. 保存
                cv2.imwrite(f'{self.debug_dir}/landmark_0.png', img_vis)

            timestep = 0

            ops = self.diff_renderer(vertices,
                                self.r_mvps[timestep].detach(), self.R_base[timestep], self.t_base[timestep],
                                verts_noneck=vertices_noneck,
                                verts_depth=proj_vertices[:, :, 2:3],
                            )

            all_loss, losses = self.compute_loss(variables, ops, proj_vertices, batch, is_joint, True, losses, p, iters, num_views, normal_mask, normal_map)
            
            all_loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            self.global_step += 1
            loss_color = all_loss.item()

            if loss_color < best_loss - 1.0:
                best_loss = loss_color
                n_steps_stagnant = 0
            elif p > 25: # only start counting after n steps
                n_steps_stagnant += 1

            if p > 0:
                past_k_steps[p%stagnant_window_size] = np.abs(all_loss.item() - prev_loss)
            prev_loss = all_loss.item()

            self.frame += 1

            iterator.set_description(f'Timestep 0; Loss {all_loss.item():.4f}')
            # print({k: f"{v.item():.4f}" for k, v in losses.items()})

    def optimize_color(self, batch, params_func,
                       save_timestep=0,
                       is_joint : bool = False,
                       enable_profiling: bool = False,
                       ):

        # if enable_profiling:

        #     profile_steps = 3
        #     prof_schedule = torch.profiler.schedule(wait=1, warmup=1, active=profile_steps, repeat=1)
        #     profiler_log_dir = f"{self.save_folder}/{self.actor_name}/optimize_color"
        #     # self.profiler_log_file = os.path.join(profiler_log_dir, 'fcd81199db73_3710790.1755866321613231646.pt.trace.json')
        #     trace_handler = torch.profiler.tensorboard_trace_handler(profiler_log_dir)
        #     prof = torch.profiler.profile(
        #         activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        #         schedule=prof_schedule,
        #         on_trace_ready = trace_handler,
        #         record_shapes=True,
        #         with_stack=True,
        #         profile_memory=True,
        #     )

        #     record = torch.profiler.record_function
        
        # else:
        #     prof = NullProfiler() 
        #     record = lambda name: nullcontext()

        iters = self.config.iters
        
        images, used_landmarks = self.parse_landmarks(batch)   # images: [1, 3, 512, 512] used_landmarks: [1, N, 2]

        normal_mask = batch["normal_mask"]

        normal_map = batch["normals"] if "normals" in batch else None
        optimizer = torch.optim.Adam(params_func())

        optimizer.zero_grad()

        num_views = len(self.R_base.keys())

        image_lmk = used_landmarks
        
        self.diff_renderer.reset()

        best_loss = np.inf

        n_steps_stagnant = 0
        stagnant_window_size = 10
        past_k_steps = np.array([100.0 for _ in range(stagnant_window_size)])

        iterator = tqdm(range(iters), desc='', leave=True, miniters=100)

        # with prof:

        for p in iterator:

            # with record("FLAME forward"):
            
            # 这一步是flame的前向 包括FK+LBS
            losses, vertices, vertices_noneck, proj_vertices, proj_lmk, variables, num_views = self.flame_forward(iters, p, image_lmk)
            # proj_vertices: [1, 5023, 3]

            # 帮我把proj_lmk，image_lmk在images上可视化一下，并存下来
            # proj_lmk的形状是[1, N, 3]，image_lmk的形状是[1, N, 2]，images的形状是[1, 3, 512, 512]
            if DEBUG == True:
                img_tensor = images[0].clone().detach().cpu()

                if img_tensor.min() < 0:
                    img_tensor = (img_tensor + 1) / 2.0
                img_tensor = torch.clamp(img_tensor, 0, 1)

                img_np = img_tensor.permute(1, 2, 0).numpy()
                img_np = (img_np * 255).astype(np.uint8)
                img_vis = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

                # 2. 处理关键点
                # proj_lmk 取前两维 (x, y)，忽略第三维深度
                pred_pts = proj_lmk[0, :, :2].detach().cpu().numpy()
                gt_pts = image_lmk[0].detach().cpu().numpy()

                # 绿色 (Green) 画 GT landmarks
                for pt in gt_pts:
                    cv2.circle(img_vis, (int(pt[0]), int(pt[1])), 2, (0, 255, 0), -1)
                
                # 红色 (Red) 画 Projected landmarks
                for pt in pred_pts:
                    cv2.circle(img_vis, (int(pt[0]), int(pt[1])), 2, (0, 0, 255), -1)

                # 4. 保存
                cv2.imwrite(f'{self.debug_dir}/landmark_{save_timestep}.png', img_vis)

            timestep = 0

            # with record("diff_renderer (nvdiffrast)"):

            # 这一步是通过nvdiffrast把前向出来的flame mesh渲染成图片
            ops = self.diff_renderer(vertices,
                                self.r_mvps[timestep].detach(), self.R_base[timestep], self.t_base[timestep],
                                verts_noneck=vertices_noneck,
                                verts_depth=proj_vertices[:, :, 2:3],
                                )

            # with record("Loss Calculation"):

            # 这一步是根据渲染出的图片和真实图片计算loss
            all_loss, losses = self.compute_loss(variables, ops, proj_vertices, batch, is_joint, False, losses, p, iters, num_views, normal_mask, normal_map)
            
            # with record("Backward Pass and Optimizer Step"):

            all_loss.backward()
            
            optimizer.step()
            optimizer.zero_grad()

            self.global_step += 1
            loss_color = all_loss.item()

            if loss_color < best_loss - 1.0:
                best_loss = loss_color
                n_steps_stagnant = 0
            elif p > 25: # only start counting after n steps
                n_steps_stagnant += 1

            if p > 0:
                past_k_steps[p%stagnant_window_size] = np.abs(all_loss.item() - prev_loss)
            prev_loss = all_loss.item()

            self.frame += 1

            iterator.set_description(f'Timestep {save_timestep}; Loss {all_loss.item():.4f}')
            # print({k: f"{v.item():.4f}" for k, v in losses.items()})

            # prof.step()

            if self.config.early_stopping:
                if p > stagnant_window_size and np.mean(past_k_steps) < self.config.early_stopping_delta:
                    print('Early Stopping, go to next frame!')
                    break
    
    def optimize_color_global(self, batch, params_func,
                       no_lm : bool = False,
                       save_timestep=0,
                       is_joint : bool = False,
                       enable_profiling: bool = False,
                       ):

        # if enable_profiling:

        #     profile_steps = 3
        #     prof_schedule = torch.profiler.schedule(wait=1, warmup=1, active=profile_steps, repeat=1)
        #     profiler_log_dir = f"{self.save_folder}/{self.actor_name}/optimize_color_global"
        #     # self.profiler_log_file = os.path.join(profiler_log_dir, 'fcd81199db73_3710790.1755866321613231646.pt.trace.json')
        #     trace_handler = torch.profiler.tensorboard_trace_handler(profiler_log_dir)
        #     prof = torch.profiler.profile(
        #         activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        #         schedule=prof_schedule,
        #         on_trace_ready = trace_handler,
        #         record_shapes=True,
        #         with_stack=True,
        #         profile_memory=True,
        #     )

        #     record = torch.profiler.record_function
        
        # else:
        #     prof = NullProfiler() 
        #     record = lambda name: nullcontext()

        iters = self.config.iters
        
        # Optimizer per step
        optimizer = torch.optim.SparseAdam(params_func())
        params_global = [
            {'params': [self.shape], 'lr': self.config.lr_id * 1.0, 'name': ['shape']}
        ]
        if self.config.global_camera:
            params_global.append({'params': [self.focal_length], 'lr': self.config.lr_f * 1.0,
                        'name': ['camera_params']})
            params_global.append({'params': [self.principal_point], 'lr': self.config.lr_pp * 1.0,
                        'name': ['camera_params']})
        optimizer_id = torch.optim.Adam(params_global)

        optimizer_id.zero_grad()

        optimizer.zero_grad()

        self.diff_renderer.reset()

        best_loss = np.inf

        n_steps_stagnant = 0
        stagnant_window_size = 10
        past_k_steps = np.array([100.0 for _ in range(stagnant_window_size)])

        iterator = tqdm(range(iters), desc='', leave=True, miniters=100)

        # with prof:

        for p in iterator:

            if p == int(iters*0.5):

                for pgroup in optimizer.param_groups:
                    if pgroup['name'] in ['t', 'R', 'jaw']:
                        pgroup['lr'] = pgroup['lr'] / 10
                        print(f'LR Reduce at iter {p}, for pgroup {pgroup["name"]}')
                    else:
                        pgroup['lr'] = pgroup['lr'] / 2
            if p == int(iters *0.75):
                for pgroup in optimizer.param_groups:
                    if pgroup['name'] in ['t', 'R', 'jaw']:
                        pgroup['lr'] = pgroup['lr'] / 5
                        print(f'LR Reduce at iter {p}, for pgroup {pgroup["name"]}')
                    else:
                        pgroup['lr'] = pgroup['lr'] / 2

            if p == int(iters *0.9):
                for pgroup in optimizer.param_groups:
                    if pgroup['name'] in ['t', 'R', 'jaw']:
                        pgroup['lr'] = pgroup['lr'] / 2
                        print(f'LR Reduce at iter {p}, for pgroup {pgroup["name"]}')
                    else:
                        pgroup['lr'] = pgroup['lr'] / 5

            selected_frames, batch, bs, num_views, image_lmk, normal_map, normal_mask = self.select_frame(iters, p)

            selected_frames = torch.from_numpy(selected_frames).long().to(self.device)

            # with record("FLAME forward"):

            # 这一步是flame的前向 包括FK+LBS
            losses, vertices, vertices_noneck, proj_vertices, proj_lmk, variables = self.flame_forward_global(iters, p, selected_frames, bs, num_views, image_lmk)

            timestep = 0

            # with record("diff_renderer (nvdiffrast)"):

            # 这一步是通过nvdiffrast把前向出来的flame mesh渲染成图片
            ops = self.diff_renderer(vertices,
                                    self.r_mvps[timestep].detach(), self.R_base[timestep], self.t_base[timestep],
                                    verts_noneck=vertices_noneck,
                                    verts_depth=proj_vertices[:, :, 2:3],
                                    )

            # with record("Loss Calculation"):

            # 这一步是根据渲染出的图片和真实图片计算loss
            all_loss, losses = self.compute_loss(variables, ops, proj_vertices, batch, is_joint, False, losses, p, iters, num_views, normal_mask, normal_map)
            
            # with record("Backward Pass and Optimizer Step"):

            all_loss.backward()
            
            optimizer.step()
            optimizer.zero_grad()
            optimizer_id.step()
            optimizer_id.zero_grad()

            self.global_step += 1
            loss_color = all_loss.item()

            if loss_color < best_loss - 1.0:
                best_loss = loss_color
                n_steps_stagnant = 0
            elif p > 25: # only start counting after n steps
                n_steps_stagnant += 1

            if p > 0:
                past_k_steps[p%stagnant_window_size] = np.abs(all_loss.item() - prev_loss)
            prev_loss = all_loss.item()

            self.frame += 1

            iterator.set_description(f'Timestep {save_timestep}; Loss {all_loss.item():.4f}')

            # prof.step()
    
    def render_and_save(self, batch,
                        save=True,
                        save_meshes: bool = False,
                        timestep : int = 0,
                        selected_frames = None,
                        load_checkpoint: bool = False,
                        save_keypoints: bool = False,
                        not_render: bool = False,
                        ):
        batch = self.to_cuda(batch)
        images, used_landmarks = self.parse_landmarks(batch)

        num_keyframes = 1  #1 + len(self.config.keyframes)

        with torch.no_grad():
            self.diff_renderer.reset()
            num_views = len(self.R_base.keys())
            bs = batch['normals'].shape[0] * num_keyframes #self.shape.shape[0]

            if selected_frames is None:
                exp = self.exp
                eyes = self.eyes
                eyelids = self.eyelids
                R = self.R
                t = self.t
                jaw = self.jaw
                neck = self.neck
                focal_length = self.focal_length
                principal_point = self.principal_point
            else:
                if not load_checkpoint:
                    exp = self.exp(selected_frames)
                    eyes = self.eyes(selected_frames)
                    eyelids = self.eyelids(selected_frames)
                    R = self.R(selected_frames)
                    t = self.t(selected_frames)
                    jaw = self.jaw(selected_frames)
                    neck = self.neck(selected_frames)
                    if not self.config.global_camera:
                        focal_length = self.focal_length(selected_frames)
                        principal_point = self.principal_point(selected_frames)
                    else:
                        focal_length = self.focal_length
                        principal_point = self.principal_point
                else:
                    frame = torch.load(f'{self.checkpoint_folder}/{timestep:05d}.frame', weights_only = False)
                    exp = torch.from_numpy(frame['flame']['exp']).to(self.device)  #  [1, 100] 表情
                    eyes = torch.from_numpy(frame['flame']['eyes']).to(self.device)   # [1, 12] 眼睛
                    self.shape = torch.from_numpy(frame['flame']['shape']).to(self.device)  # [1, 300] ID
                    eyelids = torch.from_numpy(frame['flame']['eyelids']).to(self.device)  # [1, 2] 瞳孔
                    R = torch.from_numpy(frame['flame']['R']).to(self.device)  # [1, 6] 全局旋转
                    t = torch.from_numpy(frame['flame']['t']).to(self.device)  # [1, 3] 全局平移
                    jaw = torch.from_numpy(frame['flame']['jaw']).to(self.device)  # [1, 6] 下颌旋转
                    neck = torch.from_numpy(frame['flame']['neck']).to(self.device)  # [1, 6] 脖子旋转
                    
                    focal_length = torch.from_numpy(frame['camera']['fl']).to(self.device)  # 焦距
                    principal_point = torch.from_numpy(frame['camera']['pp']).to(self.device)  # 主点

            intrinsics = get_intrinsics(focal_length, principal_point, use_hack=False, size=self.config.size)

            proj_512 = nvdiffrast_util.intrinsics2projection(intrinsics,
                                                                znear=0.1, zfar=5,
                                                                width=self.config.size,
                                                                height=self.config.size)
            for serial in self.cam_pose_nvd.keys():
                if not load_checkpoint:
                    extr = get_extrinsics(self.R_base[serial], self.t_base[serial])
                else:
                    R_base_0 = torch.from_numpy(frame['camera']['R_base_0']).to(self.device)
                    t_base_0 = torch.from_numpy(frame['camera']['t_base_0']).to(self.device)
                    extr = get_extrinsics(R_base_0, t_base_0)
                r_mvps = torch.matmul(proj_512, extr.repeat(bs, 1, 1))
                self.r_mvps[serial] = r_mvps
            
            self.intrinsics_hack = get_intrinsics(focal_length, principal_point, size=self.config.size)
            self.w2c_openGL = get_extrinsics(self.R_base, self.t_base).repeat(self.focal_length.shape[0], 1, 1)

            # 设置为不旋转状态
            # R: 使用单位矩阵的6D表示（表示无旋转）
            batch_size = R.shape[0]
            identity_matrix = torch.eye(3).unsqueeze(0).repeat(batch_size, 1, 1).to(self.device)
            identity_6D = matrix_to_rotation_6d(identity_matrix)  # 确保是float类型

            # 最后得到canonical space中的顶点需要把除了shape之外的所有项都设置为0
            # R = identity_6D
            # jaw = identity_6D
            # neck = identity_6D
            # exp = torch.zeros_like(exp)  # [1, 100] 全部置为0
            # eyes = torch.cat([identity_6D, identity_6D], dim = 1)  # [1, 12] 全部置为0
            # eyelids = torch.zeros_like(eyelids)  # [1, 2] 全部置为0

            if not load_checkpoint:
                vertices_can, pred_lmk, joint_transforms, vertices_can_can, vertices_noneck, y_rot_angle = self.flame(
                    cameras=torch.inverse(self.R_base[0]).repeat(bs, 1, 1),
                    shape_params=self.shape.repeat(bs, 1),
                    expression_params=exp.repeat_interleave(num_views, dim=0), #torch.from_numpy(self.expression_params[:1, :]).to(self.device).repeat(bs, 1), #self.exp,
                    eye_pose_params=eyes.repeat_interleave(num_views, dim=0),
                    #euler_angles_to_matrix(x_opts['rotation'][i], 'XYZ')
                    jaw_pose_params=jaw.repeat_interleave(num_views, dim=0), #matrix_to_rotation_6d(euler_angles_to_matrix(torch.from_numpy(self.jaw_params[:1, :]).to(self.device), 'XYZ')).repeat(bs, 1), #self.jaw,
                    neck_pose_params=neck.repeat_interleave(num_views, dim=0), #matrix_to_rotation_6d(euler_angles_to_matrix(torch.from_numpy(self.jaw_params[:1, :]).to(self.device), 'XYZ')).repeat(bs, 1), #self.jaw,
                    eyelid_params=eyelids.repeat_interleave(num_views, dim=0),
                    rot_params_lmk_shift=(matrix_to_rotation_6d(torch.inverse(rotation_6d_to_matrix(R)))).repeat_interleave(num_views, dim=0),
                )   # vertices_can是不带全局旋转，带neck pose的flame顶点，vertices_noneck是不带全局旋转且不带neck pose的flame顶点。
            else:
                vertices_can, pred_lmk, joint_transforms, vertices_can_can, vertices_noneck, y_rot_angle = self.flame(
                    cameras=torch.inverse(R_base_0).repeat(bs, 1, 1),
                    shape_params=self.shape.repeat(bs, 1),
                    expression_params=exp.repeat_interleave(num_views, dim=0), #torch.from_numpy(self.expression_params[:1, :]).to(self.device).repeat(bs, 1), #self.exp,
                    eye_pose_params=eyes.repeat_interleave(num_views, dim=0),
                    #euler_angles_to_matrix(x_opts['rotation'][i], 'XYZ')
                    jaw_pose_params=jaw.repeat_interleave(num_views, dim=0), #matrix_to_rotation_6d(euler_angles_to_matrix(torch.from_numpy(self.jaw_params[:1, :]).to(self.device), 'XYZ')).repeat(bs, 1), #self.jaw,
                    neck_pose_params=neck.repeat_interleave(num_views, dim=0), #matrix_to_rotation_6d(euler_angles_to_matrix(torch.from_numpy(self.jaw_params[:1, :]).to(self.device), 'XYZ')).repeat(bs, 1), #self.jaw,
                    eyelid_params=eyelids.repeat_interleave(num_views, dim=0),
                    rot_params_lmk_shift=(matrix_to_rotation_6d(torch.inverse(rotation_6d_to_matrix(R)))).repeat_interleave(num_views, dim=0),
                )   # vertices_can是不带全局旋转，带neck pose的flame顶点，vertices_noneck是不带全局旋转且不带neck pose的flame顶点。

            # print(joint_transforms.shape)  # [1, 5, 4, 4]

            kp_results = []

            # 以下部分是存储训练所需要的三组keypoints的代码

            if save_keypoints:

                lmk_path = f'{env_paths.PREPROCESSED_DATA}/{self.config.video_name}/lmk.npy'   # 512分辨率大小

                if self.config.size == 256:
                    lmk = np.load(lmk_path) / 2.0   # [N, 203, 2]  # [0 - img_size] [0 - 256]
                elif self.config.size == 512:
                    lmk = np.load(lmk_path)

                lmk = lmk[timestep]  # 取第一帧 [203, 2]
                lmk = torch.from_numpy(lmk).to(self.device)  # [203, 2]

                # canonical space
                R_zero = identity_6D
                jaw_zero = identity_6D
                neck_zero = identity_6D
                exp_zero = torch.zeros_like(exp)
                eyes_zero = torch.cat([identity_6D, identity_6D], dim = 1)
                eyelids_zero = torch.zeros_like(eyelids)

                if not load_checkpoint:
                    vertices_wo_exp, _, _ , _ , _ , _ = self.flame(
                        cameras=torch.inverse(self.R_base[0]).repeat(bs, 1, 1),
                        shape_params=self.shape.repeat(bs, 1),
                        expression_params=exp_zero.repeat_interleave(num_views, dim=0), #torch.from_numpy(self.expression_params[:1, :]).to(self.device).repeat(bs, 1), #self.exp,
                        eye_pose_params=eyes_zero.repeat_interleave(num_views, dim=0),
                        jaw_pose_params=jaw_zero.repeat_interleave(num_views, dim=0), #matrix_to_rotation_6d(euler_angles_to_matrix(torch.from_numpy(self.jaw_params[:1, :]).to(self.device), 'XYZ')).repeat(bs, 1), #self.jaw,
                        neck_pose_params=neck_zero.repeat_interleave(num_views, dim=0), #matrix_to_rotation_6d(euler_angles_to_matrix(torch.from_numpy(self.jaw_params[:1, :]).to(self.device), 'XYZ')).repeat(bs, 1), #self.jaw,
                        eyelid_params=eyelids_zero.repeat_interleave(num_views, dim=0),
                        rot_params_lmk_shift=(matrix_to_rotation_6d(torch.inverse(rotation_6d_to_matrix(R_zero)))).repeat_interleave(num_views, dim=0),
                    )  # 得到canonical空间下的人脸mesh，没有头部姿态，没有表情

                else:
                    vertices_wo_exp, _, _ , _ , _, _ = self.flame(
                        cameras=torch.inverse(R_base_0).repeat(bs, 1, 1),
                        shape_params=self.shape.repeat(bs, 1),
                        expression_params=exp_zero.repeat_interleave(num_views, dim=0), #torch.from_numpy(self.expression_params[:1, :]).to(self.device).repeat(bs, 1), #self.exp,
                        eye_pose_params=eyes_zero.repeat_interleave(num_views, dim=0),
                        jaw_pose_params=jaw_zero.repeat_interleave(num_views, dim=0), #matrix_to_rotation_6d(euler_angles_to_matrix(torch.from_numpy(self.jaw_params[:1, :]).to(self.device), 'XYZ')).repeat(bs, 1), #self.jaw,
                        neck_pose_params=neck_zero.repeat_interleave(num_views, dim=0), #matrix_to_rotation_6d(euler_angles_to_matrix(torch.from_numpy(self.jaw_params[:1, :]).to(self.device), 'XYZ')).repeat(bs, 1), #self.jaw,
                        eyelid_params=eyelids_zero.repeat_interleave(num_views, dim=0),
                        rot_params_lmk_shift=(matrix_to_rotation_6d(torch.inverse(rotation_6d_to_matrix(R_zero)))).repeat_interleave(num_views, dim=0),
                    )  # 得到canonical空间下的人脸mesh，没有头部姿态，没有表情

                canonical_kp = vertices_wo_exp.clone()

                canonical_kp = canonical_kp[:, self.vertex_face, :]

                canonical_kp[..., 1:] = - canonical_kp[..., 1:]  # 对yz轴取反，转换到图像坐标系下

                canonical_kp = canonical_kp * 10

                ## without pose
                if not load_checkpoint:
                    # 有neck pose的joint transforms
                    vertices, _, joint_transforms_with_neck, joint_transforms_no_neck,  _ , vertices_noneck, y_rot_angle = self.flame(
                        cameras=torch.inverse(self.R_base[0]).repeat(bs, 1, 1),
                        shape_params=self.shape.repeat(bs, 1),
                        expression_params=exp.repeat_interleave(num_views, dim=0), #torch.from_numpy(self.expression_params[:1, :]).to(self.device).repeat(bs, 1), #self.exp,
                        eye_pose_params=eyes.repeat_interleave(num_views, dim=0),
                        jaw_pose_params=jaw.repeat_interleave(num_views, dim=0), #matrix_to_rotation_6d(euler_angles_to_matrix(torch.from_numpy(self.jaw_params[:1, :]).to(self.device), 'XYZ')).repeat(bs, 1), #self.jaw,
                        neck_pose_params=neck.repeat_interleave(num_views, dim=0), #matrix_to_rotation_6d(euler_angles_to_matrix(torch.from_numpy(self.jaw_params[:1, :]).to(self.device), 'XYZ')).repeat(bs, 1), #self.jaw,
                        eyelid_params=eyelids.repeat_interleave(num_views, dim=0),
                        rot_params_lmk_shift=(matrix_to_rotation_6d(torch.inverse(rotation_6d_to_matrix(R)))).repeat_interleave(num_views, dim=0),
                        return_noneck = True,
                    )  # 不带全局头部姿态，带neck pose的vertices,以及不带全局头部姿态并且不带neck pose的joint_transforms

                else:
                    vertices, _, joint_transforms_with_neck, joint_transforms_no_neck,  _ , vertices_noneck, y_rot_angle = self.flame(
                        cameras=torch.inverse(R_base_0).repeat(bs, 1, 1),
                        shape_params=self.shape.repeat(bs, 1),
                        expression_params=exp.repeat_interleave(num_views, dim=0), #torch.from_numpy(self.expression_params[:1, :]).to(self.device).repeat(bs, 1), #self.exp,
                        eye_pose_params=eyes.repeat_interleave(num_views, dim=0),
                        jaw_pose_params=jaw.repeat_interleave(num_views, dim=0), #matrix_to_rotation_6d(euler_angles_to_matrix(torch.from_numpy(self.jaw_params[:1, :]).to(self.device), 'XYZ')).repeat(bs, 1), #self.jaw,
                        neck_pose_params=neck.repeat_interleave(num_views, dim=0), #matrix_to_rotation_6d(euler_angles_to_matrix(torch.from_numpy(self.jaw_params[:1, :]).to(self.device), 'XYZ')).repeat(bs, 1), #self.jaw,
                        eyelid_params=eyelids.repeat_interleave(num_views, dim=0),
                        rot_params_lmk_shift=(matrix_to_rotation_6d(torch.inverse(rotation_6d_to_matrix(R)))).repeat_interleave(num_views, dim=0),
                        return_noneck = True,
                    )  # 不带全局头部姿态，带neck pose的vertices,以及不带全局头部姿态并且不带neck pose的joint_transforms

                vertices = vertices[:, self.vertex_face, :]  # [1, 1787, 3]  这里是取了1787个在人脸上的点，如果把整个flame人头都算进去，很有可能超出屏幕范围，因为人头是带有脖子的

                vertices = torch.einsum('bny,bxy->bnx', vertices, rotation_6d_to_matrix(R.repeat_interleave(num_views, dim=0)))  # 加上global的全局旋转R
                t_move = t.repeat_interleave(num_views, dim=0).unsqueeze(1)
                t_move_zero = t_move.clone()
                t_move_zero[..., 2] = 0.0   # z方向不加
                vertices_pose = vertices + t_move_zero   # 只加平移的x,y分量，z方向不加平移
                vertices = vertices + t_move  # 加上完整的平移，得到完整变换后的人脸坐标

                vertices_pose[..., 2] = - vertices_pose[..., 2] 
                z = vertices_pose[...,2] * focal_length.squeeze() * self.config.size  # 这是图像空间下的z坐标

                proj_vertices = self.project_points_screen_space(vertices)  # 世界空间到屏幕空间
                proj_vertices_with_z = torch.stack([proj_vertices[...,0], proj_vertices[...,1], z], dim=-1)  # [1, 1787, 3]  图像空间下带有z的顶点坐标

                kp = proj_vertices_with_z
                kp[..., :2] = kp[..., :2] / self.config.size
                kp[..., :2] = 2 * kp[..., :2] - 1
                kp[..., 2] = kp[..., 2] / (self.config.size / 2)  # 这一步得到图像空间中的kp

                # vertices_noneck就是flame空间中带有表情，不带头部姿态的exp_kp

                exp_kp = vertices_noneck.clone()

                exp_kp = exp_kp[:, self.vertex_face, :]

                exp_kp[..., 1:] = - exp_kp[..., 1:]

                exp_kp = exp_kp * 10

                exp_delta = exp_kp - canonical_kp

                kp_results = [canonical_kp, exp_kp, exp_delta, kp]
            
            if not_render:
                return [], [], kp_results

            pred_lmk = torch.einsum('bny,bxy->bnx', pred_lmk, rotation_6d_to_matrix(R.repeat_interleave(num_views, dim=0))) + t.repeat_interleave(num_views, dim=0).unsqueeze(1)   # canonical空间到世界空间（也就是在头部加上旋转矩阵和平移）
            
            # 这里的vertices_can是不带全局旋转，带neck pose的flame顶点
            # 得到的vertices是带全局旋转和neck pose的顶点
            vertices = torch.einsum('bny,bxy->bnx', vertices_can, rotation_6d_to_matrix(R.repeat_interleave(num_views, dim=0))) + t.repeat_interleave(num_views, dim=0).unsqueeze(1)   # canonical空间到世界空间（也就是在头部加上旋转矩阵和平移）
            
            if save_meshes:
                # 在这里可视化了vertices：带有全局旋转和neck pose的顶点，以及vertices_noneck: 不带全局旋转且没有neck pose的顶点，也就是exp_kp
                self.save_meshes(timestep, vertices = vertices, vertices_noneck = vertices_noneck, vertices_can = vertices_wo_exp)
            
            vertices_noneck = torch.einsum('bny,bxy->bnx', vertices_noneck, rotation_6d_to_matrix(R.repeat_interleave(num_views, dim=0))) + t.repeat_interleave(num_views, dim=0).unsqueeze(1)  # canonical空间到世界空间（也就是在头部加上旋转矩阵和平移）

            pred_lmk = self.project_points_screen_space(pred_lmk)  # 世界空间到屏幕空间
            proj_vertices = self.project_points_screen_space(vertices)  # 世界空间到屏幕空间

            _timestep = 0
            ops = self.diff_renderer(vertices,
                                     self.r_mvps[_timestep], self.R_base[_timestep], self.t_base[_timestep],
                                     verts_noneck=vertices_noneck,
                                     verts_depth=proj_vertices[:, :, 2:3],
                                     is_viz=True,
                                     )

            grabbed_depth = ops['actual_rendered_depth'][0, 0,
                torch.clamp(proj_vertices[0, :, 1].long(), 0, self.config.size-1),
                torch.clamp(proj_vertices[0, :, 0].long(), 0, self.config.size-1),
            ]
            is_visible_verts_idx = grabbed_depth < proj_vertices[0, :, 2] + 1e-2
            if not self.config.occ_filter:
                is_visible_verts_idx = torch.ones_like(is_visible_verts_idx)

            pred_normals = ops['normal_images']  # 1 3 512 512 normals in world space
            rot_mat = rotation_6d_to_matrix(R.detach().repeat_interleave(num_views, dim=0))  # 1 3 3
            pred_normals_flame_space = torch.einsum('bxy,bxhw->byhw', rot_mat, pred_normals)
            
            output_rows = []

            for b_i in range(bs):

                # 第一列：原始图像
                original_img = (images[b_i].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            
                # 第二列：融合后的图像
                shape_mask = ((ops['alpha_images'] * ops['mask_images_mesh']) > 0.).int()[b_i]
                shape = (pred_normals_flame_space[b_i] + 1) / 2 * shape_mask
                blended = (images[b_i] * (1 - shape_mask) + images[b_i] * shape_mask * 0.3 + shape * 0.7 * shape_mask)
                blended_img = (blended.permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)
                
                # 第三列：预测的法向贴图
                predicted_normal = ((pred_normals_flame_space[b_i].permute(1, 2, 0) + 1) / 2 * 255).detach().cpu().numpy().astype(np.uint8)
                
                # ===== 拼接成一行 =====
                row = np.concatenate([original_img, blended_img, predicted_normal], axis=1)
                output_rows.append(row)

            # ===== 最终输出 =====
            # 多个batch叠在一起
            final_output = np.concatenate(output_rows, axis=0)
            catted = Image.fromarray(final_output)

            # ===== Single输出（中间列） =====
            single_output = final_output[:, 512:1024, :]  # 假设宽度512
            catted_single = Image.fromarray(single_output)

            if not save:
                return

            # CHECKPOINT
            if not load_checkpoint:
                self.save_checkpoint(timestep, selected_frames=selected_frames)

        return catted, catted_single, kp_results

    def parse_landmarks(self, batch):
        images = batch['rgb']
        if 'lmk' in batch:
            landmarks = batch['lmk']  # [B, 203, 2]

            lip_landmarks = landmarks[:, self.lmk_lip_index, :]  # [B, 8, 2]

            left_up_eyebrow_landmarks = landmarks[:, self.lmk_left_up_eyebrow_index, :]  # [B, 5, 2]
            left_bottom_eyebrow_landmarks = landmarks[:, self.lmk_left_bottom_eyebrow_index, :]   # [B, 5, 2]

            left_eyebrow_landmarks = (left_up_eyebrow_landmarks + left_bottom_eyebrow_landmarks) / 2  # [B, 5, 2]

            right_up_eyebrow_landmarks = landmarks[:, self.lmk_right_up_eyebrow_index, :]  # [B, 5, 2]
            right_bottom_eyebrow_landmarks = landmarks[:, self.lmk_right_bottom_eyebrow_index, :]  # [B, 5, 2]

            right_eyebrow_landmarks = (right_up_eyebrow_landmarks + right_bottom_eyebrow_landmarks) / 2  # [B, 5, 2]

            left_eye_landmarks = landmarks[:, self.lmk_left_eye_index, :]  # [B, 1, 2]
            right_eye_landmarks = landmarks[:, self.lmk_right_eye_index, :]  # [B, 1, 2]

            left_iris_landmarks = landmarks[:, self.lmk_left_iris_index, :]  # [B, 1, 2]
            right_iris_landmarks = landmarks[:, self.lmk_right_iris_index, :]  # [B, 1, 2]

            left_nose_landmarks = landmarks[:, self.lmk_left_nose_index, :]  # [B, 6, 2]
            right_nose_landmarks = landmarks[:, self.lmk_right_nose_index, :]  # [B, 6, 2]
            column_nose_landmarks = landmarks[:, self.lmk_column_nose_index, :]  # [B, 4, 2]

            contour_landmarks = landmarks[:, self.lmk_contour_index, :]  # [B, 3, 2]

            used_landmarks = torch.cat([lip_landmarks,
                                        left_eyebrow_landmarks, right_eyebrow_landmarks,
                                        left_eye_landmarks, right_eye_landmarks,
                                        left_iris_landmarks, right_iris_landmarks,
                                        left_nose_landmarks, right_nose_landmarks, column_nose_landmarks,
                                        contour_landmarks], dim=1)  # [B, 28, 2]

            batch['left_iris'] = landmarks[:, 197:198, :]
            batch['right_iris'] = landmarks[:, 198:199, :]

        else:
            landmarks = None

        return images,  used_landmarks

    def read_all_data(self):

        DATA_FOLDER = f'{env_paths.PREPROCESSED_DATA}/{self.config.video_name}'
        P3DMM_FOLDER = f'{env_paths.PREPROCESSED_DATA}/{self.config.video_name}/p3dmm'

        lmk_path = f'{DATA_FOLDER}/lmk.npy'
        lmk = np.load(lmk_path)  # [n, 203, 2] 范围是在0-512之间

        all_batches = []

        for timestep in range(self.MAX_STEPS):
            try:
                rgb = np.array(Image.open(f'{DATA_FOLDER}/cropped/{timestep:05d}.jpg').resize((self.config.size, self.config.size))) / 255  # 256 * 256大小，0-1之间
            except Exception as ex:
                rgb = np.array(Image.open(f'{DATA_FOLDER}/cropped/{timestep:05d}.png').resize((self.config.size, self.config.size))) / 255  # 256 * 256大小，0-1之间

            seg = np.array(Image.open(f'{DATA_FOLDER}/seg_og/{timestep:05d}.png').resize((self.config.size, self.config.size), Image.NEAREST))  # 读取通过facer分割的结果，分割图的大小位于[0-17]，代表总共18个类别
            if len(seg.shape) == 3:
                seg = seg[..., 0]

            normal_mask = ((seg == 2) | (seg == 6) | (seg == 7) |
                    (seg == 10) | (seg == 12) | (seg == 13)
                    ) | (seg == 11)  # mouth interior
            if self.config.big_normal_mask:
                normal_mask = normal_mask | (seg==1) | (seg == 4) | (seg==5) # add neck and ears

            fg_mask = ((seg == 2) | (seg == 6) | (seg == 7) | (seg == 8) | (seg == 9) | #(seg == 4) | (seg == 5) |
                    (seg == 10) | (seg == 12) | (seg == 13)
                    )

            valid_bg = seg <= 1

            normals = ((np.array(Image.open(f'{P3DMM_FOLDER}/normals/{timestep:05d}.png').resize((self.config.size, self.config.size))) / 255).astype(np.float32) - 0.5 )*2

            lms = lmk[timestep]
            lms = lms / 512
            lms = lms * self.config.size

            ret_dict = {
                'rgb': rgb,   # 图像rgb 
                'normals': normals,    # pixel3dmm预测的法线图
                'normal_mask': normal_mask,
                'fg_mask': fg_mask,
                'valid_bg': valid_bg,
            }
            if lms is not None:
                ret_dict['lmk'] = lms

            ret_dict = {k: torch.from_numpy(v).float().unsqueeze(0).to(self.device) for k,v in ret_dict.items()}

            ret_dict['normal_mask'] = ret_dict['normal_mask'][:, :, :, None].repeat(1, 1, 1, 3)
            ret_dict['fg_mask'] = ret_dict['fg_mask'][:, :, :, None].repeat(1, 1, 1, 3)

            channels_first = ['rgb', 'normal_mask', 'normals', 'fg_mask']
            for k in channels_first:
                ret_dict[k] = ret_dict[k].permute(0, 3, 1, 2)

            all_batches.append(ret_dict)

        return all_batches

    def prepare_global_optimization(self, N_FRAMES):
        is_sparse=True
        self.exp = nn.Embedding(num_embeddings=N_FRAMES, embedding_dim=100, sparse=is_sparse, ).to(self.device)
        self.R = nn.Embedding(num_embeddings=N_FRAMES, embedding_dim=6, sparse=is_sparse).to(self.device)
        self.t = nn.Embedding(num_embeddings=N_FRAMES, embedding_dim=3, sparse=is_sparse).to(self.device)
        self.eyes = nn.Embedding(num_embeddings=N_FRAMES, embedding_dim=12, sparse=is_sparse).to(self.device)
        self.eyelids = nn.Embedding(num_embeddings=N_FRAMES, embedding_dim=12, sparse=is_sparse).to(self.device)
        self.jaw = nn.Embedding(num_embeddings=N_FRAMES, embedding_dim=6, sparse=is_sparse).to(self.device)
        self.neck = nn.Embedding(num_embeddings=N_FRAMES, embedding_dim=6, sparse=is_sparse).to(self.device)
        if not self.config.global_camera:
            self.focal_length = nn.Embedding(num_embeddings=N_FRAMES, embedding_dim=1, sparse=is_sparse).to(self.device)
            self.principal_point = nn.Embedding(num_embeddings=N_FRAMES, embedding_dim=2, sparse=is_sparse).to(self.device)

        exp = torch.cat(self.intermediate_exprs, dim=0)
        R = torch.cat(self.intermediate_Rs, dim=0)
        t = torch.cat(self.intermediate_ts, dim=0)
        eyes = torch.cat(self.intermediate_eyes, dim=0)
        eyelids = torch.cat(self.intermediate_eyelids, dim=0)
        jaw = torch.cat(self.intermediate_jaws, dim=0)
        neck = torch.cat(self.intermediate_necks, dim=0)
        if not self.config.global_camera:
            focal_length = torch.cat(self.intermediate_fls, dim=0)
            principal_point = torch.cat(self.intermediate_pps, dim=0)

        with torch.no_grad():
            self.exp.weight = torch.nn.Parameter(exp)
            self.R.weight = torch.nn.Parameter(R)
            self.t.weight = torch.nn.Parameter(t)
            self.eyes.weight = torch.nn.Parameter(eyes)
            self.eyelids.weight = torch.nn.Parameter(eyelids)
            self.jaw.weight = torch.nn.Parameter(jaw)
            self.neck.weight = torch.nn.Parameter(neck)
            if not self.config.global_camera:
                self.focal_length.weight = torch.nn.Parameter(focal_length)
                self.principal_point.weight = torch.nn.Parameter(principal_point)
    
    def load_checkpoint_and_render(self, save_keypoints = True, save_meshes = False, not_render = False):
        all_batches = self.read_all_data()  # 读取所有帧的数据

        # Important to initialize
        self.create_parameters(0)

        if save_keypoints:
            canonical_kp_list = []
            exp_kp_list = []
            exp_delta_list = []
            kp_list = []

        # render result and save it as a video to get some viusal feedback
        video_frames = []
        video_single = []
        for it, timestep in enumerate(range(self.config.start_frame, self.config.start_frame + self.MAX_STEPS, self.FRAME_SKIP)):
            selected_frames = []
            selected_frames_loading = []
            batches = []
            batch = all_batches[timestep]
            batches.append(batch)
            selected_frames.append(it)
            selected_frames_loading.append(timestep)
            batches = {k: torch.cat([x[k] for x in batches], dim=0) for k in batch.keys()}
            selected_frames = torch.from_numpy(np.array(selected_frames)).long().to(self.device)

            result_rendering, result_rendering_single, kp_results = self.render_and_save(batches,
                                                    save_meshes = save_meshes, timestep=timestep, selected_frames=selected_frames,
                                                    load_checkpoint=True, save_keypoints=save_keypoints, not_render = not_render)
            
            if save_keypoints:
                canonical_kp, exp_kp, exp_delta, kp = kp_results
                canonical_kp_list.append(canonical_kp)
                exp_kp_list.append(exp_kp)
                exp_delta_list.append(exp_delta)
                kp_list.append(kp)

            video_frames.append(np.array(result_rendering))
            video_single.append(np.array(result_rendering_single))
            self.frame += 1
        
        if not not_render:
            mediapy.write_video(f'{self.save_folder}/{self.actor_name}/result.mp4', images=video_frames, crf=15, fps=self.fps)
            mediapy.write_video(f'{self.save_folder}/{self.actor_name}/single.mp4', images=video_single, crf=15, fps=self.fps)
        
        if save_keypoints:
            canonical_kp = torch.cat(canonical_kp_list, dim=0).cpu().numpy()
            exp_kp = torch.cat(exp_kp_list, dim=0).cpu().numpy()
            exp_delta = torch.cat(exp_delta_list, dim=0).cpu().numpy()
            kp = torch.cat(kp_list, dim=0).cpu().numpy()

            np.save(os.path.join(self.kp_folder_eye, 'canonical_kp.npy'), canonical_kp)
            np.save(os.path.join(self.kp_folder_eye, 'exp_kp.npy'), exp_kp)
            np.save(os.path.join(self.kp_folder_eye, 'exp_delta.npy'), exp_delta)
            np.save(os.path.join(self.kp_folder_eye, 'kp.npy'), kp)
        
        vis_kp_video(preprocessed_dir = f"{env_paths.PREPROCESSED_DATA}/{self.actor_name}/", tracking_dir = f"{self.save_folder}/{self.actor_name}/", fps = self.fps)
        
        print(f'''
                <<<<<<<< DONE WITH RENDERING {self.actor_name} >>>>>>>>
                ''')

    def run(self, save_keypoints = True, save_meshes = False, not_render = False):
        all_batches = self.read_all_data()  # 读取所有帧的数据

        # Important to initialize
        self.create_parameters(0)
        self.frame = 0

        print('''
        <<<<<<<< STARTING ONLINE TRACKING PHASE >>>>>>>>
        ''')

        for timestep in range(self.config.start_frame, self.MAX_STEPS + self.config.start_frame, self.FRAME_SKIP):  # MAX_STEPS就是视频帧数  FRAME_SKIP = 1
            batch = all_batches[timestep]
            for k in batch.keys():
                if k not in self.cached_data:
                    self.cached_data[k] = [batch[k]]
                else:
                    self.cached_data[k].append(batch[k])
            if timestep == self.config.start_frame:

                params = lambda: self.clone_params_keyframes_all(freeze_id=False, freeze_cam=False, include_neck=self.config.include_neck)  # freeze_cam为True

                # 在第一步优化所有的参数，包括ID和相机位姿
                self.optimize_color_first_step(batch, params, is_joint = False)
                
                bs = 1
                self.intrinsics = get_intrinsics(self.focal_length, self.principal_point, use_hack=False, size=self.config.size)

                self.proj_512 = nvdiffrast_util.intrinsics2projection(self.intrinsics,
                                                         znear=0.1, zfar=5,
                                                         width=self.config.size,
                                                         height=self.config.size)

                for serial in self.cam_pose_nvd.keys():
                    extr = get_extrinsics(self.R_base[serial], self.t_base[serial])
                    r_mvps = torch.matmul(self.proj_512, extr.repeat(bs, 1, 1))
                    self.r_mvps[serial] = r_mvps
                
                self.extrinsics = extr

                self.intrinsics_hack = get_intrinsics(self.focal_length, self.principal_point, size=self.config.size)
                self.w2c_openGL = get_extrinsics(self.R_base, self.t_base).repeat(self.focal_length.shape[0], 1, 1)

            else:
                params = lambda: self.clone_params_keyframes_all(freeze_id=True, freeze_cam=self.config.global_camera, include_neck=self.config.include_neck)  # freeze_cam为True

            # 这里的batch size是1，单帧数据
            self.optimize_color(batch, params, save_timestep=timestep, is_joint = False)

            self.frame += 1

            # save results for global optimization later
            self.intermediate_exprs.append(self.exp.detach().clone())
            self.intermediate_Rs.append(self.R.detach().clone())
            self.intermediate_ts.append(self.t.detach().clone())
            self.intermediate_eyes.append(self.eyes.detach().clone())
            self.intermediate_eyelids.append(self.eyelids.detach().clone())
            self.intermediate_jaws.append(self.jaw.detach().clone())
            self.intermediate_necks.append(self.neck.detach().clone())
            if not self.config.global_camera:
                self.intermediate_fls.append(self.focal_length.detach().clone())
                self.intermediate_pps.append(self.principal_point.detach().clone())

            if self.config.early_exit:  # False
                exit()
        
        for k in self.cached_data.keys():
            self.cached_data[k] = torch.cat(self.cached_data[k], dim=0)

        params = lambda: self.clone_params_keyframes_all_joint(freeze_id=False, include_neck=self.config.include_neck)

        self.config.iters = self.config.global_iters

        N_FRAMES = len(self.intermediate_exprs)
        
        self.prepare_global_optimization(N_FRAMES=N_FRAMES)  # 这一步是把所有帧的参数合并在一起，创建一个大的nn.Embedding，然后用之前逐帧优化出来的每一帧的参数初始化这个大的nn.Embedding

        print('''
                <<<<<<<< STARTING GLOBAL TRACKING PHASE >>>>>>>>
                ''')

        if N_FRAMES > 1:
            
            # 这里optimize_color的batch size是16，是在所有帧里随机选16帧
            self.optimize_color_global(None, params,
                                save_timestep=1000,   # 在做全局优化的时候，默认timestep是1000
                                is_joint=True,
                                )

        if save_keypoints:
            canonical_kp_list = []
            exp_kp_list = []
            exp_delta_list = []
            kp_list = []

        # render result and save it as a video to get some viusal feedback
        video_frames = []
        video_single = []
        for it, timestep in enumerate(range(self.config.start_frame, self.MAX_STEPS + self.config.start_frame, self.FRAME_SKIP)):
            selected_frames = []
            selected_frames_loading = []
            batches = []
            batch = all_batches[timestep]
            batches.append(batch)
            selected_frames.append(it)
            selected_frames_loading.append(timestep)
            batches = {k: torch.cat([x[k] for x in batches], dim=0) for k in batch.keys()}
            selected_frames = torch.from_numpy(np.array(selected_frames)).long().to(self.device)

            result_rendering, result_rendering_single, kp_results = self.render_and_save(batches,
                                                    save_meshes = save_meshes, timestep=timestep, selected_frames=selected_frames,
                                                    save_keypoints = save_keypoints, not_render = not_render)
            if save_keypoints:
                canonical_kp, exp_kp, exp_delta, kp = kp_results
                canonical_kp_list.append(canonical_kp)
                exp_kp_list.append(exp_kp)
                exp_delta_list.append(exp_delta)
                kp_list.append(kp)

            video_frames.append(np.array(result_rendering))
            video_single.append(np.array(result_rendering_single))
            self.frame += 1

        if not not_render:
            mediapy.write_video(f'{self.save_folder}/{self.actor_name}/result.mp4', images=video_frames, crf=15, fps=self.fps)
            mediapy.write_video(f'{self.save_folder}/{self.actor_name}/single.mp4', images=video_single, crf=15, fps=self.fps)

        if save_keypoints:
            canonical_kp = torch.cat(canonical_kp_list, dim=0).cpu().numpy()
            exp_kp = torch.cat(exp_kp_list, dim=0).cpu().numpy()
            exp_delta = torch.cat(exp_delta_list, dim=0).cpu().numpy()
            kp = torch.cat(kp_list, dim=0).cpu().numpy()

            np.save(os.path.join(self.kp_folder_eye, 'canonical_kp.npy'), canonical_kp)
            np.save(os.path.join(self.kp_folder_eye, 'exp_kp.npy'), exp_kp)
            np.save(os.path.join(self.kp_folder_eye, 'exp_delta.npy'), exp_delta)
            np.save(os.path.join(self.kp_folder_eye, 'kp.npy'), kp)
    
        # Optionally delete all preoprocessing artifacts, once tracking is done (only keep cropped images)
        if self.config.delete_preprocessing:
            shutil.rmtree(f'{env_paths.PREPROCESSED_DATA}/{self.config.video_name}/cropped')
            shutil.rmtree(f'{env_paths.PREPROCESSED_DATA}/{self.config.video_name}/p3dmm')
            shutil.rmtree(f'{env_paths.PREPROCESSED_DATA}/{self.config.video_name}/seg_og')
        
        vis_kp_video(preprocessed_dir = f"{env_paths.PREPROCESSED_DATA}/{self.actor_name}/", tracking_dir = f"{self.save_folder}/{self.actor_name}/", fps = self.fps)

        print(f'''
                <<<<<<<< DONE WITH TRACKING {self.actor_name} >>>>>>>>
                ''')