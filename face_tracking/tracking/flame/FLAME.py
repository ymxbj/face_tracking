# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2023 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: mica@tue.mpg.de

import os
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.insert(0, '../../..')
from face_tracking import env_paths
from face_tracking.utils.device import get_device

from face_tracking.utils.utils_3d import rotation_6d_to_matrix, matrix_to_rotation_6d
from .vertices2joints import Vertices2JointsTriton

I = matrix_to_rotation_6d(torch.eye(3)[None].to(get_device()))


def to_tensor(array, dtype=torch.float32):
    if 'torch.tensor' not in str(type(array)):
        return torch.tensor(array, dtype=dtype)


def to_np(array, dtype=np.float32):
    if 'scipy.sparse' in str(type(array)):
        array = array.todense()
    return np.array(array, dtype=dtype)


class Struct(object):
    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)

def transform_mat(R, t):
    ''' Creates a batch of transformation matrices
        Args:
            - R: Bx3x3 array of a batch of rotation matrices
            - t: Bx3x1 array of a batch of translation vectors
        Returns:
            - T: Bx4x4 Transformation matrix
    '''
    # No padding left or right, only add an extra row
    return torch.cat([F.pad(R, [0, 0, 0, 1]),
            F.pad(t, [0, 0, 0, 1], value=1)], dim=2)


def rot_mat_to_euler(rot_mats):
    # Calculates rotation matrix to euler angles
    # Careful for extreme cases of eular angles like [0.0, pi, 0.0]

    sy = torch.sqrt(rot_mats[:, 0, 0] * rot_mats[:, 0, 0] +
                    rot_mats[:, 1, 0] * rot_mats[:, 1, 0])
    return torch.atan2(-rot_mats[:, 2, 0], sy)

def dense_to_sparse_with_threshold(tensor: torch.Tensor, threshold: float) -> torch.Tensor:
    """
    Args:
        tensor (torch.Tensor): 输入的稠密张量。
        threshold (float): 用于判断是否保留元素的绝对值阈值。

    Returns:
        torch.Tensor: 生成的稀疏张量。
    """
    mask = torch.abs(tensor) >= threshold
    indices = torch.nonzero(mask).t()
    values = tensor[mask]
    return torch.sparse_coo_tensor(indices, values, tensor.shape).coalesce()

class FLAME(nn.Module):
    """
    borrowed from https://github.com/soubhiksanyal/FLAME_PyTorch/blob/master/FLAME.py
    Given FLAME parameters for shape, pose, and expression, this class generates a differentiable FLAME function
    which outputs the a mesh and 2D/3D facial landmarks
    """

    def __init__(self, config, flame_assets, contour_side_num,lmk_loss_in_flame_list):
        super(FLAME, self).__init__()
        self.flame_assets = flame_assets
        with open(f'{self.flame_assets}/FLAME2020/generic_model.pkl', 'rb') as f:
            ss = pickle.load(f, encoding='latin1')
            flame_model = Struct(**ss)

        self.dtype = torch.float32

        self.contour_side_num = contour_side_num
        self.lmk_loss_in_flame_list = lmk_loss_in_flame_list
        
        [self.flame_lip_loss,
        self.flame_left_eyebrow_loss, self.flame_right_eyebrow_loss,
        self.flame_left_eye_loss, self.flame_right_eye_loss,
        self.flame_left_iris_loss, self.flame_right_iris_loss,
        self.flame_left_nose_loss, self.flame_right_nose_loss, self.flame_column_nose_loss] = self.lmk_loss_in_flame_list

        self.register_buffer('faces', to_tensor(to_np(flame_model.f, dtype=np.int64), dtype=torch.long))
        # The vertices of the template model
        self.register_buffer('v_template', to_tensor(to_np(flame_model.v_template), dtype=self.dtype))
        # The shape components and expression
        shapedirs = to_tensor(to_np(flame_model.shapedirs), dtype=self.dtype)
        shapedirs = torch.cat([shapedirs[:, :, :config.num_shape_params], shapedirs[:, :, 300:300 + config.num_exp_params]], 2)
        self.register_buffer('shapedirs', shapedirs)
        # The pose components
        num_pose_basis = flame_model.posedirs.shape[-1]
        posedirs = np.reshape(flame_model.posedirs, [-1, num_pose_basis]).T
        self.register_buffer('posedirs', to_tensor(to_np(posedirs), dtype=self.dtype))
        #
        self.register_buffer('J_regressor', to_tensor(to_np(flame_model.J_regressor), dtype=self.dtype))
        self.register_buffer('J_regressor_sparse', self.J_regressor.to_sparse_coo())
        self.register_buffer('J_regressor_sparse_joint_idxes', self.J_regressor_sparse.indices()[0])
        self.register_buffer('J_regressor_sparse_verts_idxes', self.J_regressor_sparse.indices()[1])
        self.register_buffer('J_regressor_sparse_weights', self.J_regressor_sparse.values())        

        parents = to_tensor(to_np(flame_model.kintree_table[0])).long();
        parents[0] = -1
        self.register_buffer('parents', parents)
        self.register_buffer('lbs_weights', to_tensor(to_np(flame_model.weights), dtype=self.dtype))
        self.register_buffer('lbs_weights_sparse', dense_to_sparse_with_threshold(self.lbs_weights, threshold = 1e-5))
        self.register_buffer('lbs_weights_sparse_verts_idxes', self.lbs_weights_sparse.indices()[0])
        self.register_buffer('lbs_weights_sparse_joints_idxes', self.lbs_weights_sparse.indices()[1])
        self.register_buffer('lbs_weights_sparse_weights', self.lbs_weights_sparse.values())

        self.register_buffer('l_eyelid', torch.from_numpy(np.load(f'{os.path.abspath(os.path.dirname(__file__))}/blendshapes/l_eyelid.npy')).to(self.dtype)[None])
        self.register_buffer('r_eyelid', torch.from_numpy(np.load(f'{os.path.abspath(os.path.dirname(__file__))}/blendshapes/r_eyelid.npy')).to(self.dtype)[None])

        # Register default parameters
        self._register_default_params('neck_pose_params', 6)
        self._register_default_params('jaw_pose_params', 6)
        self._register_default_params('eye_pose_params', 12)
        self._register_default_params('shape_params', config.num_shape_params)
        self._register_default_params('expression_params', config.num_exp_params)

        # Static and Dynamic Landmark embeddings for FLAME
        lmk_embedding_path = f'{self.flame_assets}/FLAME2020/landmark_embedding.npy'
        if not os.path.exists(lmk_embedding_path):
            # Fallback to the lightweight copy that ships with this repo.
            lmk_embedding_path = env_paths.FLAME_LANDMARK_EMBEDDING
        lmk_embeddings = np.load(lmk_embedding_path, allow_pickle=True, encoding='latin1')
        lmk_embeddings = lmk_embeddings[()]

        self.register_buffer('lmk_faces_idx', torch.from_numpy(lmk_embeddings['static_lmk_faces_idx'].astype(int)).to(torch.int64))
        self.register_buffer('lmk_bary_coords', torch.from_numpy(lmk_embeddings['static_lmk_bary_coords']).to(self.dtype).float())
        self.register_buffer('dynamic_lmk_faces_idx', torch.from_numpy(np.array(lmk_embeddings['dynamic_lmk_faces_idx']).astype(int)).to(torch.int64))
        self.register_buffer('dynamic_lmk_bary_coords', torch.from_numpy(np.array(lmk_embeddings['dynamic_lmk_bary_coords'])).to(self.dtype).float())

        neck_kin_chain = []
        NECK_IDX = 1
        curr_idx = torch.tensor(NECK_IDX, dtype=torch.long)
        while curr_idx != -1:
            neck_kin_chain.append(curr_idx)
            curr_idx = self.parents[curr_idx]
        self.register_buffer('neck_kin_chain', torch.stack(neck_kin_chain))

    def vertices2landmarks(self, vertices, dyn_lmk_faces_idx, dyn_lmk_bary_coords):
        """
        vertices: [batch_size, 5023, 3]
        return:
        landmarks: [batch_size, L, 3]
        """
        batch_size = vertices.shape[0]
        num_verts = vertices.shape[1]
        device = vertices.device

        # Static landmarks
        lip_landmarks = vertices[:, self.flame_lip_loss, :]  # [batch_size, 8, 3]
        
        left_eyebrow_landmarks = vertices[:, self.flame_left_eyebrow_loss, :]  # [batch_size, 5, 3]

        right_eyebrow_landmarks = vertices[:, self.flame_right_eyebrow_loss, :]  # [batch_size, 5, 3]

        left_eye_landmarks = vertices[:, self.flame_left_eye_loss, :]  # [batch_size, 1, 3]
        right_eye_landmarks = vertices[:, self.flame_right_eye_loss, :]  # [batch_size, 1, 3]

        left_iris_landmarks = vertices[:, self.flame_left_iris_loss, :]  # [batch_size, 1, 3]
        right_iris_landmarks = vertices[:, self.flame_right_iris_loss, :]  # [batch_size, 1, 3]

        left_nose_landmarks = vertices[:, self.flame_left_nose_loss, :]   # [batch_size, 6, 3]
        right_nose_landmarks = vertices[:, self.flame_right_nose_loss, :]   # [batch_size, 6, 3]
        column_nose_landmarks = vertices[:, self.flame_column_nose_loss, :]   # [batch_size, 4, 3]

        # Dynamic landmarks (contour)
        contour_lmk_faces = torch.index_select(self.faces, 0, dyn_lmk_faces_idx.view(-1).to(torch.long)).view(batch_size, -1, 3)
        contour_lmk_faces += torch.arange(batch_size, dtype=torch.long, device=device).view(-1, 1, 1) * num_verts
        contour_lmk_vertices = vertices.view(-1, 3)[contour_lmk_faces].view(batch_size, -1, 3, 3)
        contour_landmarks = torch.einsum('blfi,blf->bli', [contour_lmk_vertices, dyn_lmk_bary_coords])  # [batch_size, 17, 3]

        # 对于contour_landmarks，选取中间的那个点，以及左右两侧紧挨的side_num个点，总共2 * side_num + 1个点
        mid_idx = contour_landmarks.shape[1] // 2
        selected_indices = list(range(mid_idx - self.contour_side_num, mid_idx + self.contour_side_num + 1))
        contour_landmarks = contour_landmarks[:, selected_indices, :]

        pred_landmarks = torch.cat([lip_landmarks,
                                    left_eyebrow_landmarks, right_eyebrow_landmarks,
                                    left_eye_landmarks, right_eye_landmarks,
                                    left_iris_landmarks, right_iris_landmarks,
                                    left_nose_landmarks, right_nose_landmarks, column_nose_landmarks,
                                    contour_landmarks], dim=1)  # [batch_size, 41, 3]

        return pred_landmarks
    
    def _find_dynamic_lmk_idx_and_bcoords(self, vertices, pose, dynamic_lmk_faces_idx,
                                          dynamic_lmk_b_coords,
                                          neck_kin_chain, cameras, dtype=torch.float32):
        """
            Selects the face contour depending on the reletive position of the head
            Input:
                vertices: N X num_of_vertices X 3
                pose: N X full pose
                dynamic_lmk_faces_idx: The list of contour face indexes
                dynamic_lmk_b_coords: The list of contour barycentric weights
                neck_kin_chain: The tree to consider for the relative rotation
                dtype: Data type
            return:
                The contour face indexes and the corresponding barycentric weights
        """

        batch_size = vertices.shape[0]

        aa_pose = torch.index_select(pose.view(batch_size, -1, 6), 1, neck_kin_chain)
        rot_mats = rotation_6d_to_matrix(aa_pose.view(-1, 6)).view([batch_size, -1, 3, 3])

        rel_rot_mat = torch.eye(3, device=vertices.device, dtype=dtype).unsqueeze_(dim=0).expand(batch_size, -1, -1)

        for idx in range(len(neck_kin_chain)):
            rel_rot_mat = torch.bmm(rot_mats[:, idx], rel_rot_mat)

        if cameras is not None:
            rel_rot_mat = cameras @ rel_rot_mat  # Cameras flips z and x, plus multiview needs different lmk sliding per view
        
        y_rot_angle_return = torch.round(-rot_mat_to_euler(rel_rot_mat) * 180.0 / np.pi).to(dtype=torch.long)

        y_rot_angle = torch.round(torch.clamp(-rot_mat_to_euler(rel_rot_mat) * 180.0 / np.pi, max=39)).to(dtype=torch.long)
        neg_mask = y_rot_angle.lt(0).to(dtype=torch.long)
        mask = y_rot_angle.lt(-39).to(dtype=torch.long)
        neg_vals = mask * 78 + (1 - mask) * (39 - y_rot_angle)
        y_rot_angle = (neg_mask * neg_vals + (1 - neg_mask) * y_rot_angle)

        dyn_lmk_faces_idx = torch.index_select(dynamic_lmk_faces_idx, 0, y_rot_angle)
        dyn_lmk_b_coords = torch.index_select(dynamic_lmk_b_coords, 0, y_rot_angle)

        return y_rot_angle_return, dyn_lmk_faces_idx, dyn_lmk_b_coords

    def cal_y_rot_angle(self, vertices, pose, neck_kin_chain, cameras, dtype=torch.float32):
        """
            Calculates the y rot angle based on the head poses.
            Input:
                vertices: [batch_size, num_of_vertices, 3]
                pose: [batch_size, full pose]
                neck_kin_chain: The tree to consider for the relative rotation
                dtype: Data type
            return:
                The contour face indexes and the corresponding barycentric weights
        """

        batch_size = vertices.shape[0]

        aa_pose = torch.index_select(pose.view(batch_size, -1, 6), 1, neck_kin_chain)
        rot_mats = rotation_6d_to_matrix(aa_pose.view(-1, 6)).view([batch_size, -1, 3, 3])

        rel_rot_mat = torch.eye(3, device=vertices.device, dtype=dtype).unsqueeze_(dim=0).expand(batch_size, -1, -1)

        for idx in range(len(neck_kin_chain)):
            rel_rot_mat = torch.bmm(rot_mats[:, idx], rel_rot_mat)

        if cameras is not None:
            rel_rot_mat = cameras @ rel_rot_mat  # Cameras flips z and x, plus multiview needs different lmk sliding per view

        y_rot_angle = torch.round(-rot_mat_to_euler(rel_rot_mat) * 180.0 / np.pi).to(dtype=torch.long)

        return y_rot_angle

    def vertices2joints(self, vertices):
        """
        J_regressor是个稠密矩阵，大小为[5, 5023] 只有43个有效值，稀疏率为99.83%
        vertices就是表面蒙皮上的点，大小为[2, 5023, 3]
        该函数最终返回[2, 5, 3]
        """
        return torch.einsum('bik,ji->bjk', [vertices, self.J_regressor])

    def vertices2joints_flash(self, vertices, device):
        """
        J_regressor是个稠密矩阵，大小为[5, 5023] 只有43个有效值，稀疏率为99.83%
        vertices就是表面蒙皮上的点，大小为[2, 5023, 3]
        该函数最终返回[2, 5, 3]
        """
        batch_size = vertices.shape[0]
        num_joints = self.J_regressor.shape[0]
        
        vertices_expand = vertices[:, self.J_regressor_sparse_verts_idxes, :]   # [2, 43, 3]

        joint_expand = vertices_expand * self.J_regressor_sparse_weights[None, :, None]  # [2, 43, 3]

        result = torch.zeros(batch_size, num_joints, 3).to(device)  # [2, 5, 3]

        result.index_add_(1, self.J_regressor_sparse_joint_idxes, joint_expand)   # [2, 5, 3]

        return result

    def vertices2joints_triton(self, vertices):
        """
        J_regressor是个稠密矩阵，大小为[5, 5023] 只有43个有效值，稀疏率为99.83%
        vertices就是表面蒙皮上的点，大小为[2, 5023, 3]
        该函数最终返回[2, 5, 3]
        """
        num_joints = self.J_regressor.shape[0]

        return Vertices2JointsTriton.apply(
            vertices,
            self.J_regressor_sparse_weights,
            self.J_regressor_sparse_verts_idxes,
            self.J_regressor_sparse_joint_idxes,
            num_joints,
        )

    def skinning(self, v_posed, A, batch_size, device):
        '''
        v_posed: [batch_size, 5023, 3]
        A:  [batch_size, 5, 16]
        '''

        W = self.lbs_weights.unsqueeze(dim=0).expand([batch_size, -1, -1])
        # (N x V x (J + 1)) x (N x (J + 1) x 16)
        num_joints = self.J_regressor.shape[0]
        T = torch.matmul(W, A.view(batch_size, num_joints, 16)).view(batch_size, -1, 4, 4)   # [batch_size, 5023, 4, 4]

        homogen_coord = torch.ones([batch_size, v_posed.shape[1], 1],
                                dtype=self.dtype, device=device)
        v_posed_homo = torch.cat([v_posed, homogen_coord], dim=2)
        v_homo = torch.matmul(T, torch.unsqueeze(v_posed_homo, dim=-1))

        verts = v_homo[:, :, :3, 0]

        return verts
    
    def skinning_flash(self, v_posed, A, batch_size, device):
        '''
        v_posed: [batch_size, 5023, 3]
        A:  [batch_size, 5, 16]
        '''
        B = batch_size
        N = v_posed.shape[1]
        J = A.shape[1]
        K = A.shape[2]

        A_gathered = A[:, self.lbs_weights_sparse_joints_idxes,:]
        weighted_A = self.lbs_weights_sparse_weights.view(1, -1, 1) * A_gathered 

        result = torch.zeros(B, N, K, device=device, dtype=self.dtype)

        result.index_add_(1, self.lbs_weights_sparse_verts_idxes, weighted_A).view(batch_size, -1, 4, 4)

        T = result

        homogen_coord = torch.ones([batch_size, v_posed.shape[1], 1],
                                dtype=self.dtype, device=device)
        v_posed_homo = torch.cat([v_posed, homogen_coord], dim=2)
        v_homo = torch.matmul(T, torch.unsqueeze(v_posed_homo, dim=-1))

        verts = v_homo[:, :, :3, 0]

        return verts

    def batch_rigid_transform(self, rot_mats, joints):
        """
        Applies a batch of rigid transformations to the joints

        Parameters
        ----------
        rot_mats : torch.tensor BxNx3x3
            Tensor of rotation matrices
        joints : torch.tensor BxNx3
            Locations of joints
        parents : torch.tensor BxN
            The kinematic tree of each object
        dtype : torch.dtype, optional:
            The data type of the created tensors, the default is torch.float32

        Returns
        -------
        posed_joints : torch.tensor BxNx3
            The locations of the joints after applying the pose rotations
        rel_transforms : torch.tensor BxNx4x4
            The relative (with respect to the root joint) rigid transformations
            for all the joints
        """

        joints = torch.unsqueeze(joints, dim=-1)

        rel_joints = joints.clone()
        rel_joints[:, 1:] -= joints[:, self.parents[1:]]

        transforms_mat = transform_mat(
            rot_mats.view(-1, 3, 3),
            rel_joints.reshape(-1, 3, 1)).reshape(-1, joints.shape[1], 4, 4)

        transform_chain = [transforms_mat[:, 0]]   # [batch_size, 4, 4]

        curr_res = torch.matmul(transform_chain[0], transforms_mat[:, 1])   # [batch_size, 4, 4]
        transform_chain.append(curr_res)

        transform_chain_combined = torch.cat([transform_chain[1], transform_chain[1], transform_chain[1]], dim = 0)  # [3 * batch_size, 4, 4]
        transforms_mat_combined = torch.cat([transforms_mat[:, 2], transforms_mat[:, 3], transforms_mat[:, 4]], dim = 0)    # [3 * batch_size, 4, 4]
        curr_res_combined = torch.matmul(transform_chain_combined, transforms_mat_combined)  # [3 * batch_size, 4, 4]
        curr_res_1, curr_res_2, curr_res_3 = torch.chunk(curr_res_combined, chunks = 3, dim = 0)  #  3 * [batch_size, 4, 4]    

        transform_chain.append(curr_res_1)
        transform_chain.append(curr_res_2)
        transform_chain.append(curr_res_3)

        transforms = torch.stack(transform_chain, dim=1)

        # The last column of the transformations contains the posed joints
        posed_joints = transforms[:, :, :3, 3]

        # The last column of the transformations contains the posed joints
        posed_joints = transforms[:, :, :3, 3]

        joints_homogen = F.pad(joints, [0, 0, 0, 1])

        rel_transforms = transforms - F.pad(
            torch.matmul(transforms, joints_homogen), [3, 0, 0, 0, 0, 0, 0, 0])

        return posed_joints, rel_transforms
    
    def blend_shapes(self, betas):
        ''' Calculates the per vertex displacement due to the blend shapes
        Parameters
        ----------
        betas : torch.tensor Bx(num_betas)
            Blend shape coefficients
        shapedirs: torch.tensor Vx3x(num_betas)
            Blend shapes

        Returns
        -------
        torch.tensor BxVx3
            The per-vertex displacement due to shape deformation
        '''

        # Displacement[b, m, k] = sum_{l} betas[b, l] * shape_disps[m, k, l]
        # i.e. Multiply each shape displacement by its corresponding beta and
        # then sum them.
        blend_shape = torch.einsum('bl,mkl->bmk', [betas, self.shapedirs])
        return blend_shape

    def lbs(self, betas, pose, v_template, pose2rot = True):

        batch_size = max(betas.shape[0], pose.shape[0])
        device = betas.device

        # Add shape contribution
        v_shaped = v_template + self.blend_shapes(betas)

        # Get the joints
        # NxJx3 array
        # J_regressor是个稠密矩阵，大小为[5, 5023] 只有43个有效值，稀疏率为99.83%
        # v_shaped就是表面蒙皮上的点，大小为[2, 5023, 3]
        # lbs_weights也是个稠密矩阵，大小为[5023, 5] 有11716个有效值，稀疏率为53.35%

        # J = self.vertices2joints(v_shaped)  # 返回[2, 5, 3]

        # J = self.vertices2joints_flash(v_shaped, device) # 返回[2, 5, 3]

        J = self.vertices2joints_triton(v_shaped)  # 返回[2, 5, 3]

        # 3. Add pose blend shapes
        # N x J x 3 x 3
        ident = torch.eye(3, dtype=self.dtype, device=device)
        if pose2rot:            
            rot_mats = rotation_6d_to_matrix(pose.view(-1, 6)).view([batch_size, -1, 3, 3])

            pose_feature = (rot_mats[:, 1:, :, :] - ident).view([batch_size, -1])
            # (N x P) x (P, V * 3) -> N x V x 3
            pose_offsets = torch.matmul(pose_feature, self.posedirs) \
                .view(batch_size, -1, 3)
        else:
            pose_feature = pose[:, 1:].view(batch_size, -1, 3, 3) - ident
            rot_mats = pose.view(batch_size, -1, 3, 3)

            pose_offsets = torch.matmul(pose_feature.view(batch_size, -1),
                                        self.posedirs).view(batch_size, -1, 3)

        v_posed = pose_offsets + v_shaped

        # 4. Get the global joint location
        # rot_mats [batch_size, 5, 3, 3]
        # J [batch_size, 5, 3]
        # self.parents [5]
        J_transformed, A = self.batch_rigid_transform(rot_mats, J)   # 这一步是骨骼树的遍历，可以把4次循环优化为2次循环

        # 5. Do skinning:

        verts = self.skinning(v_posed, A, batch_size, device)

        return verts, A, v_shaped


    def forward(self, shape_params,
                cameras,
                trans_params=None,
                rot_params=None,
                neck_pose_params=None,
                jaw_pose_params=None,
                eye_pose_params=None,
                expression_params=None,
                eyelid_params=None,
                rot_params_lmk_shift = None,
                return_noneck = False,
                ):

        """
            Input:
                trans_params: N X 3 global translation
                rot_params: N X 3 global rotation around the root joint of the kinematic tree (rotation is NOT around the origin!)
                neck_pose_params (optional): N X 3 rotation of the head vertices around the neck joint
                jaw_pose_params (optional): N X 3 rotation of the jaw
                eye_pose_params (optional): N X 6 rotations of left (parameters [0:3]) and right eyeball (parameters [3:6])
                shape_params (optional): N X number of shape parameters
                expression_params (optional): N X number of expression parameters
            return:d
                vertices: N X V X 3
                landmarks: N X number of landmarks X 3
        """
        batch_size = shape_params.shape[0]

        I = matrix_to_rotation_6d(torch.cat([torch.eye(3)[None]] * batch_size, dim=0).to(get_device()))

        if trans_params is None:
            trans_params = torch.zeros(batch_size, 3).to(get_device())
        if rot_params is None:
            rot_params = I.clone()
        if rot_params_lmk_shift is None:
            rot_params_lmk_shift = rot_params
        if neck_pose_params is None:
            neck_pose_params = I.clone()
        if jaw_pose_params is None:
            jaw_pose_params = I.clone()
        if eye_pose_params is None:
            eye_pose_params = torch.cat([I.clone()] * 2, dim=1)
        if shape_params is None:
            shape_params = self.shape_params.expand(batch_size, -1)
        if expression_params is None:
            expression_params = self.expression_params.expand(batch_size, -1)

        # Concatenate identity shape and expression parameters
        betas = torch.cat([shape_params, expression_params], dim=1)

        # The pose vector contains global rotation, and neck, jaw, and eyeball rotations
        full_pose = torch.cat([rot_params, neck_pose_params, jaw_pose_params, eye_pose_params], dim=1)   # [1, 30]
        full_pose_no_neck = torch.cat([rot_params, I, jaw_pose_params, eye_pose_params], dim=1)    # [1, 30]
        full_pose_lmk = torch.cat([rot_params_lmk_shift, neck_pose_params, jaw_pose_params, eye_pose_params], dim=1)   # [1, 30]

        # FLAME models shape and expression deformations as vertex offset from the mean face in 'zero pose', called v_template
        template_vertices = self.v_template.unsqueeze(0).expand(batch_size, -1, -1)  # [1, 5023, 3]

        combined_full_pose = torch.cat([full_pose, full_pose_no_neck], dim=0)   # [2, 30]
        combined_betas = betas.repeat(2, 1)  # [2, num_betas]
        combined_template_vertices = template_vertices.repeat(2, 1, 1)  # [2, 5023, 3]

        # Use linear blendskinning to model pose roations
        combined_vertices, combined_joint_transforms, combined_v_can = \
        self.lbs(combined_betas, combined_full_pose, combined_template_vertices)

        # combined_vertices [2, 5023, 3]
        # combined_joint_transforms [2, 5, 4, 4]
        # combined_v_can [2, 5023, 3]

        vertices, vertices_noneck = torch.chunk(combined_vertices, 2, dim = 0)
        joint_transforms, joint_transforms_noneck = torch.chunk(combined_joint_transforms, 2, dim = 0)
        v_can, v_can_noneck = torch.chunk(combined_v_can, 2, dim=0)

        if eyelid_params is not None:
            vertices = vertices + self.r_eyelid.expand(batch_size, -1, -1) * eyelid_params[:, 1:2, None]
            vertices = vertices + self.l_eyelid.expand(batch_size, -1, -1) * eyelid_params[:, 0:1, None]
            
            vertices_noneck = vertices_noneck + self.r_eyelid.expand(batch_size, -1, -1) * eyelid_params[:, 1:2, None]
            vertices_noneck = vertices_noneck + self.l_eyelid.expand(batch_size, -1, -1) * eyelid_params[:, 0:1, None]

        y_rot_angle, dyn_lmk_faces_idx, dyn_lmk_bary_coords = self._find_dynamic_lmk_idx_and_bcoords(
                vertices, full_pose_lmk, self.dynamic_lmk_faces_idx,
                self.dynamic_lmk_bary_coords,
                self.neck_kin_chain, cameras, dtype=self.dtype)
    
        # dyn_lmk_faces_idx: [batch_size, 17]
        # dyn_lmk_bary_coords: [batch_size, 17, 3]
        
        pred_lmk = self.vertices2landmarks(vertices, dyn_lmk_faces_idx, dyn_lmk_bary_coords)  # [N, 28, 3]

        if return_noneck:
            return vertices, pred_lmk, joint_transforms, joint_transforms_noneck, v_can_noneck, vertices_noneck, y_rot_angle

        return vertices, pred_lmk, joint_transforms, v_can_noneck, vertices_noneck, y_rot_angle

    def _register_default_params(self, param_fname, dim):
        default_params = torch.zeros([1, dim], dtype=self.dtype, requires_grad=False)
        self.register_parameter(param_fname, nn.Parameter(default_params, requires_grad=False))