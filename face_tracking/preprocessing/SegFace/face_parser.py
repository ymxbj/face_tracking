
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
from .segface_celeb import SegFaceCeleb

class FaceParserInference:
    def __init__(self, model_path, swin_model_path, device='cuda'):

        # 使用swin base (v1)
        model_name = "swin_base"
        input_resolution = 512

        self.device = device
        self.input_resolution = input_resolution
        self.label_names = [
            'background', 'neck', 'skin', 'cloth', 'l_ear', 'r_ear', 
            'l_brow', 'r_brow', 'l_eye', 'r_eye', 'nose', 'mouth', 
            'l_lip', 'u_lip', 'hair', 'eye_g', 'hat', 'ear_r', 'neck_l'
        ]  # CelebAMaskHQ 19类
        
        # 构建模型
        self.model = SegFaceCeleb(input_resolution, model_name, swin_model_path).to(device)
        self.model.eval()
        
        # 加载权重
        checkpoint = torch.load(model_path, map_location=device)
        self.model.load_state_dict(checkpoint['state_dict_backbone'])

        # 创建标准化变换
        self.normalize_transform = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    
    def batch_predict(self, image_tensor):
        """
        批量图片预测
        image_tensor: [bs, 3, H, W]
        """
        image_tensor = self.normalize_transform(image_tensor).to(self.device)  # image_tensor现在是4维 [B, C, H, W]
        # print(image_tensor.max()) # tensor(2.5703, device='cuda:0')
        batch_size = image_tensor.shape[0]
        
        with torch.no_grad():
            # 创建虚拟的labels和dataset参数（来自validation函数的接口要求）
            dummy_labels = {
                'segmentation': torch.zeros(batch_size, self.input_resolution, self.input_resolution).to(self.device),
                'lnm_seg': torch.zeros(batch_size, 5, 2).to(self.device)
            }
            dummy_dataset = torch.zeros(batch_size).to(self.device).long()
            
            # 调用模型（与validation函数一致）
            seg_output = self.model(image_tensor, dummy_labels, dummy_dataset)

            # print(seg_output.shape) torch.Size([54, 19, 512, 512])
            
            mask = F.interpolate(seg_output, size=(self.input_resolution, self.input_resolution), 
                               mode='bilinear', align_corners=False)

            mask = mask.softmax(dim=1)
            preds = torch.argmax(mask, dim=1)  # [54, 512, 512]

            return seg_output, preds.cpu().numpy().astype(np.uint8)