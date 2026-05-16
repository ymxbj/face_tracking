import torch

import math
from face_tracking.tracking import util
from face_tracking.utils.utils_3d import rotation_6d_to_matrix, matrix_to_rotation_6d
from face_tracking.utils.device import get_device

I = torch.eye(3)[None].to(get_device()).detach()
I6D = matrix_to_rotation_6d(I)

class FlameForwardMixin:

    def cal_visibility_weights(self, y_rot_angle, num_landmarks, threshold_start = 30, threshold_end = 50):
        '''
        :param y_rot_angle: [B]
        return:
        weights [B, num_landmarks]
        '''
        batch_size = y_rot_angle.shape[0]

        weights = torch.ones(batch_size, num_landmarks, device = self.device)

        # 左半脸权重
        t_left = torch.clamp(
            (-y_rot_angle - threshold_start) / (threshold_end - threshold_start), 0.0, 1.0
        )  # [B]

        w_left = 0.5 * (1 + torch.cos(t_left * math.pi))  # [B]

        t_right = torch.clamp(
                (y_rot_angle - threshold_start) / (threshold_end - threshold_start), 0.0, 1.0
            )  # [B]
        
        w_right = 0.5 * (1 + torch.cos(t_right * math.pi))  # [B]

        weights[:, self.left_face_indices] *= w_left.unsqueeze(-1)  # [B, 1] 广播
        weights[:, self.right_face_indices] *= w_right.unsqueeze(-1)  # [B, 1] 广播

        return weights

    @torch.compiler.disable
    def get_vars(self, selected_frames):
        exp = self.exp(selected_frames)
        eyes = self.eyes(selected_frames)
        eyelids = self.eyelids(selected_frames)
        rotation = self.R(selected_frames)
        translation = self.t(selected_frames)
        jaw = self.jaw(selected_frames)
        neck = self.neck(selected_frames)
        focal_length = self.focal_length
        principal_point = self.principal_point

        return exp, eyes, eyelids, rotation, translation, jaw, neck, focal_length, principal_point

    def model_forward_compute(self, params, image_lmk, bs, num_views, iters, p):
        """
        内部核心计算函数，负责 FLAME 前向传播、投影和 Loss 计算
        """
        image_size = [self.config.size, self.config.size]
        losses = {}

        # 解包参数
        exp = params['exp']
        eyes = params['eyes']
        eyelids = params['eyelids']
        R = params['R']
        t = params['t']
        jaw = params['jaw']
        neck = params['neck']
        # principal_point = params['principal_point'] # 如果需要用到传参进来的pp

        matrix_R = rotation_6d_to_matrix(R)  # [B, 3, 3]

        vertices_can, pred_lmk, joint_transforms, vertices_can_can, vertices_noneck, y_rot_angle = self.flame(
            cameras=torch.inverse(self.R_base[0]).repeat(bs, 1, 1),
            shape_params=self.shape if self.shape.shape[0] == bs else self.shape.repeat(bs, 1).to(self.device),
            expression_params=exp.repeat_interleave(num_views, dim=0),
            eye_pose_params=eyes.repeat_interleave(num_views, dim=0),
            jaw_pose_params=jaw.repeat_interleave(num_views, dim=0),
            neck_pose_params=neck.repeat_interleave(num_views, dim=0),
            eyelid_params=eyelids.repeat_interleave(num_views, dim=0),
            rot_params_lmk_shift=(matrix_to_rotation_6d(torch.inverse(matrix_R))).repeat_interleave(num_views, dim=0),
        )

        verts_can_can_mirrored = vertices_can_can[:, self.mirror_order, :]
        vertices_can_can_mirrored = torch.zeros_like(verts_can_can_mirrored)
        vertices_can_can_mirrored[:, :, 0] = -verts_can_can_mirrored[:, :, 0]
        vertices_can_can_mirrored[:, :, 1:] = verts_can_can_mirrored[:, :, 1:]
        mirror_loss = (vertices_can_can_mirrored - vertices_can_can).square().sum(-1)
        mirror_loss = mirror_loss.mean()

        pred_lmk = torch.einsum('bny,bxy->bnx', pred_lmk,
                             matrix_R.repeat_interleave(num_views, dim=0)) + t.repeat_interleave(
            num_views, dim=0).unsqueeze(1)

        vertices = torch.einsum('bny,bxy->bnx', vertices_can,
                                matrix_R.repeat_interleave(num_views, dim=0)) + t.repeat_interleave(
            num_views, dim=0).unsqueeze(1)
        vertices_noneck = torch.einsum('bny,bxy->bnx', vertices_noneck,
                               matrix_R.repeat_interleave(num_views, dim=0)) + t.repeat_interleave(
            num_views, dim=0).unsqueeze(1)

        proj_lmk = self.project_points_screen_space(pred_lmk)
        proj_vertices = self.project_points_screen_space(vertices)

        right_eye, left_eye = eyes[:, :6], eyes[:, 6:]
        lmk_num = proj_lmk.shape[1]

        visibility_wights = self.cal_visibility_weights(y_rot_angle, lmk_num)
        visibility_wights = visibility_wights.unsqueeze(-1)

        # landmark loss
        scaled_proj_lmk, scaled_image_lmk = util.scale_lmks(proj_lmk[..., :2], image_lmk, image_size)

        losses['loss/lmk_lip'] = util.lmk_loss(scaled_proj_lmk[:, :self.lip_len], scaled_image_lmk[:, :self.lip_len], image_size, visibility_wights[:, :self.lip_len], scale = False) * self.config.w_lmks_mouth * 10
        losses['loss/lmk_eyebrow'] = util.lmk_loss(scaled_proj_lmk[:, self.lip_len: self.lip_len + self.eyebrow_len], scaled_image_lmk[:, self.lip_len: self.lip_len + self.eyebrow_len], image_size, visibility_wights[:, self.lip_len: self.lip_len + self.eyebrow_len], scale = False) * self.config.w_lmks_eyebrow * 10
        losses['loss/lmk_eyes'] = util.lmk_loss(scaled_proj_lmk[:, self.lip_len + self.eyebrow_len: self.lip_len + self.eyebrow_len + self.eye_len], scaled_image_lmk[:, self.lip_len + self.eyebrow_len: self.lip_len + self.eyebrow_len + self.eye_len], image_size, visibility_wights[:, self.lip_len + self.eyebrow_len: self.lip_len + self.eyebrow_len + self.eye_len], scale = False) * self.config.w_lmks_eyes * 10
        losses['loss/lmk_iris'] = util.lmk_loss(scaled_proj_lmk[:, self.lip_len + self.eyebrow_len + self.eye_len: self.lip_len + self.eyebrow_len + self.eye_len + self.iris_len], scaled_image_lmk[:, self.lip_len + self.eyebrow_len + self.eye_len: self.lip_len + self.eyebrow_len + self.eye_len + self.iris_len], image_size, visibility_wights[:, self.lip_len + self.eyebrow_len + self.eye_len: self.lip_len + self.eyebrow_len + self.eye_len + self.iris_len], scale = False) * self.config.w_lmks_iris * 10
        losses['loss/lmk_nose'] = util.lmk_loss(scaled_proj_lmk[:, self.lip_len + self.eyebrow_len + self.eye_len + self.iris_len: self.lip_len + self.eyebrow_len + self.eye_len + self.iris_len + self.nose_len], scaled_image_lmk[:, self.lip_len + self.eyebrow_len + self.eye_len + self.iris_len: self.lip_len + self.eyebrow_len + self.eye_len + self.iris_len + self.nose_len], image_size, visibility_wights[:, self.lip_len + self.eyebrow_len + self.eye_len + self.iris_len: self.lip_len + self.eyebrow_len + self.eye_len + self.iris_len + self.nose_len], scale = False) * self.config.w_lmks_nose * 10
        losses['loss/lmk_contour'] = util.lmk_loss(scaled_proj_lmk[:, self.lip_len + self.eyebrow_len + self.eye_len + self.iris_len + self.nose_len: self.lip_len + self.eyebrow_len + self.eye_len + self.iris_len + self.nose_len + self.contour_len], scaled_image_lmk[:, self.lip_len + self.eyebrow_len + self.eye_len + self.iris_len + self.nose_len: self.lip_len + self.eyebrow_len + self.eye_len + self.iris_len + self.nose_len + self.contour_len], image_size, visibility_wights[:, self.lip_len + self.eyebrow_len + self.eye_len + self.iris_len + self.nose_len: self.lip_len + self.eyebrow_len + self.eye_len + self.iris_len + self.nose_len + self.contour_len], scale = False) * self.config.w_lmks_contour * 10
        
        # Reguralizers
        losses['reg/exp'] = torch.sum(exp ** 2, dim=-1).mean() * self.config.w_exp
        losses['reg/sym'] = torch.sum((right_eye - left_eye) ** 2, dim=-1).mean() * 0.1
        losses['reg/jaw'] = torch.sum((I6D - jaw) ** 2, dim=-1).mean() * self.config.w_jaw
        losses['reg/neck'] = torch.sum((I6D - neck) ** 2, dim=-1).mean() * self.config.w_neck
        # losses['reg/eye_lids'] = torch.sum((eyelids[:, 0] - eyelids[:, 1]) ** 2, dim=-1).mean() * 0.1
        losses['reg/eye_left'] = torch.sum((I6D - left_eye) ** 2, dim=-1).mean() * 0.01
        losses['reg/eye_right'] = torch.sum((I6D - right_eye) ** 2, dim=-1).mean() * 0.01
        losses['reg/shape_general'] = torch.sum((self.shape) ** 2, dim=-1).mean() * self.config.w_shape_general
        losses['reg/mirror'] = mirror_loss * 5000
        
        if not (self.config.n_fine and p >= iters // 2):
            losses['reg/pp'] = torch.sum(self.principal_point ** 2, dim=-1).mean()

        return losses, vertices, vertices_noneck, proj_vertices, proj_lmk

    def flame_forward(self, iters, p, image_lmk):

        self.diff_renderer.reset()
        bs = 1
        num_views = 1
        
        params = {
            'exp': self.exp,
            'eyes': self.eyes,
            'eyelids': self.eyelids,
            'R': self.R,   # [1, 6]
            't': self.t,
            'jaw': self.jaw,
            'neck': self.neck,
            'principal_point': self.principal_point,
            'focal_length': self.focal_length,
        }

        losses, vertices, vertices_noneck, proj_vertices, proj_lmk = self.model_forward_compute(
            params, image_lmk, bs, num_views, iters, p
        )

        return losses, vertices, vertices_noneck, proj_vertices, proj_lmk, params, num_views
    
    def flame_forward_global(self, iters, p, selected_frames, bs, num_views, image_lmk):

        self.diff_renderer.reset()

        exp, eyes, eyelids, rotation, translation, jaw, neck, focal_length, principal_point = self.get_vars(selected_frames)

        params = {
            'exp': exp,
            'eyes': eyes,
            'eyelids': eyelids,
            'R': rotation,   # [1, 6]
            't': translation,
            'jaw': jaw,
            'neck': neck,
            'principal_point': principal_point,
            'focal_length': focal_length,
        }

        losses, vertices, vertices_noneck, proj_vertices, proj_lmk = self.model_forward_compute(
            params, image_lmk, bs, num_views, iters, p
        )

        return losses, vertices, vertices_noneck, proj_vertices, proj_lmk, params