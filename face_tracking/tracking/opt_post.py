import torch
from torchvision.transforms.functional import gaussian_blur
from face_tracking.utils.utils_3d import rotation_6d_to_matrix, rotation_6d_to_matrix_triton

class ComputeLoss:

    def compute_loss(self, variables, ops, proj_vertices, batch, is_joint, is_first_step, losses, p, iters, num_views, normal_mask, normal_map):
        grabbed_depth = ops['actual_rendered_depth'][:, 0,
                                                     torch.clamp(proj_vertices[:, :, 1].long(), 0,
                                                                 self.config.size - 1),
                                                     torch.clamp(proj_vertices[:, :, 0].long(), 0,
                                                                 self.config.size - 1),
        ][:, 0, :]

        is_visible_verts_idx = grabbed_depth < (proj_vertices[:, :, 2] + 1e-2)
        if not self.config.occ_filter:
            is_visible_verts_idx = torch.ones_like(is_visible_verts_idx)

        valid_bg_classes = batch['valid_bg']  # bg-class or neck-class
        if self.config.sil_super > 0:
            if is_joint or (not is_first_step):  # and p > 50 and p < int(iters*0.85): # 100
                # losses['loss/sil'] =((1-upper_forehead[:, None, :, :]) * (batch['fg_mask'] - ops['fg_images'])).abs().mean() * self.config.sil_super#0
                losses['loss/sil'] = ((valid_bg_classes[:, None, :, :]) * (
                            batch['fg_mask'] - ops['fg_images'])).abs().mean() * self.config.sil_super  # 0
            else:
                losses['loss/sil'] = ((valid_bg_classes[:, None, :, :]) * (
                            batch['fg_mask'] - ops['fg_images'])).abs().mean() * self.config.sil_super / 10  # 0

        skip_normals = False
        if self.config.n_fine and p < iters // 2:
            skip_normals = True

        if (self.config.normal_super > 0.0 or self.config.normal_super_can > 0.0) and not skip_normals:
            # normal_loss_map = normal_loss_map * dilated_eye_mask[:, 0, ...] * (1 - ops['mask_images_eyes_region'][:, 0, ...])
            # use dilated eye mask only
            # maybe also applie eyemask in image not rendering
            dilated_eye_mask = 1 - (gaussian_blur(ops['mask_images_eyes'],
                                                  [self.config.normal_mask_ksize, self.config.normal_mask_ksize],
                                                  sigma=[self.config.normal_mask_ksize,
                                                         self.config.normal_mask_ksize]) > 0).float()
            pred_normals = ops['normal_images']  # 1 3 512 512 normals in world space
            rot_mat = rotation_6d_to_matrix(variables["R"].repeat_interleave(num_views, dim=0))  # 1 3 3

            pred_normals_flame_space = torch.einsum('bxy,bxhw->byhw', rot_mat, pred_normals)
            if normal_map is not None:
                l_map = (normal_map - pred_normals_flame_space)
                valid = ((l_map.abs().sum(dim=1) / 3) < self.config.delta_n).unsqueeze(1)
                normal_loss_map = l_map * valid.float() * normal_mask * dilated_eye_mask
                if self.config.normal_l2:
                    losses['loss/normal'] = normal_loss_map.square().mean() * self.config.normal_super
                else:
                    losses['loss/normal'] = normal_loss_map.abs().mean() * self.config.normal_super
            else:
                losses['loss/normal'] = 0.0


        # smoothness loss
        losses = self.add_smooth_loss(losses, is_joint, p, iters, variables)

        all_loss = self.reduce_loss(losses)

        return all_loss, losses