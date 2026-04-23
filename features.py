import os
import logging
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from PIL import Image
from typing import List, Union, Dict

logger = logging.getLogger(__name__)

class LetterboxPad:
    """保持宽高比缩放，并在边缘填充颜色以达到目标尺寸，防止图像拉伸形变"""
    def __init__(self, target_size=518, fill=(0, 0, 0)):
        self.target_size = target_size
        self.fill = fill

    def __call__(self, img):
        w, h = img.size
        if w == 0 or h == 0:
            return img
        scale = self.target_size / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        # 兼容不同版本的 PIL
        resample_method = getattr(Image, 'Resampling', Image).BICUBIC
        img = img.resize((new_w, new_h), resample=resample_method)
        
        pad_left = (self.target_size - new_w) // 2
        pad_top = (self.target_size - new_h) // 2
        pad_right = self.target_size - new_w - pad_left
        pad_bottom = self.target_size - new_h - pad_top
        
        return F.pad(img, (pad_left, pad_top, pad_right, pad_bottom), fill=self.fill, padding_mode='constant')

class FeatureExtractor:
    """
    视觉特征提取器
    纯粹的视觉算法类：封装 PyTorch 模型，将图片转换为高维特征向量。
    支持 EfficientNet-B0 和 DINOv2 系列模型。
    """
    def __init__(self, model_type: str = "dinov2_vitb14"):
        """
        初始化特征提取器。
        
        :param model_type: 模型类型，可选 "efficientnet_b0", "dinov2_vits14", "dinov2_vitb14"
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_type = model_type
        
        logger.info(f"[*] 初始化视觉特征提取器 - Compute Device: {'GPU (CUDA)' if self.device.type == 'cuda' else 'CPU'}")
        
        try:
            # 1. 拦截 PyTorch 的默认缓存行为，防止任何模型被缓存在 C 盘 (~/.cache/torch)
            # 独立包模式：将模型缓存存放在本包所在目录的 models 文件夹下
            package_root = os.path.dirname(os.path.abspath(__file__))
            torch_cache_dir = os.path.join(package_root, 'models', 'torch_cache')
            os.environ['TORCH_HOME'] = torch_cache_dir
            os.makedirs(torch_cache_dir, exist_ok=True)
            
            logger.info(f"[*] 正在加载视觉模型: {model_type}")
            
            if model_type == "efficientnet_b0":
                self.feature_dim = 1280
                model_path = os.path.join(package_root, 'models', 'efficientnet_b0.pth')
                
                # 如果本地不存在指定名称的文件，则手动下载
                if not os.path.exists(model_path):
                    logger.info(f"[*] 本地未发现权重文件，开始从官方源拉取并存入 {model_path} ...")
                    url = "https://download.pytorch.org/models/efficientnet_b0_rwightman-7f5810bc.pth"
                    torch.hub.download_url_to_file(url, model_path)

                # 手动载入模型
                efficientnet = models.efficientnet_b0(weights=None)
                state_dict = torch.load(model_path, map_location=self.device)
                efficientnet.load_state_dict(state_dict)
                self.model = nn.Sequential(*(list(efficientnet.children())[:-1]), nn.Flatten())
                
                self.transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                
            elif model_type in ["dinov2_vits14", "dinov2_vitb14"]:
                self.feature_dim = 384 if model_type == "dinov2_vits14" else 768
                
                # 使用 torch.hub 加载 DINOv2
                # 模型会自动下载到 TORCH_HOME 指定的 models/torch_cache/hub 目录下
                self.model = torch.hub.load('facebookresearch/dinov2', model_type)
                
                # DINOv2 优化预处理：
                # 1. 移除 CenterCrop，防止偏离中心的物体（如空中的鞋子）被裁剪丢失。
                # 2. 提升分辨率至 DINOv2 原生的 518x518 (必须是 14 的倍数)，大幅增强细节捕捉能力。
                # 3. 使用 LetterboxPad 防止直接 Resize 导致的强行拉伸变形。
                self.transform = transforms.Compose([
                    LetterboxPad(518, fill=(255, 255, 255)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ])
                
            else:
                raise ValueError(f"不支持的模型类型: {model_type}")

            self.model = self.model.to(self.device)
            self.model.eval()
            logger.info(f"[*] 深度特征提取网络加载就绪 ({model_type})。")
            
        except Exception as e:
            logger.error(f"[!] 加载视觉模型失败引发异常: {e}")
            raise e

    def extract_features(self, image_paths: List[str]) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        批量解析给定文件路径列，并返回包含全局和局部特征的字典。
        """
        global_features: List[torch.Tensor] = []
        local_features: List[torch.Tensor] = []
        
        with torch.no_grad():
            for idx, path in enumerate(image_paths):
                try:
                    with Image.open(path) as img:
                        img_rgb = img.convert('RGB')
                    img_t = self.transform(img_rgb).unsqueeze(0).to(self.device)
                    
                    if self.model_type in ["dinov2_vits14", "dinov2_vitb14"]:
                        # 获取多尺度特征字典
                        ret = self.model.forward_features(img_t)
                        # cls token (全局特征), shape: (1, D) -> squeeze -> (D)
                        g_feat = ret['x_norm_clstoken'].squeeze(0)
                        # patch tokens (局部特征), shape: (1, N_patches, D) -> squeeze -> (N_patches, D)
                        l_feat = ret['x_norm_patchtokens'].squeeze(0)
                        global_features.append(g_feat)
                        local_features.append(l_feat)
                    else:
                        # EfficientNet 只有全局特征
                        g_feat = self.model(img_t).squeeze(0)
                        global_features.append(g_feat)
                        
                except Exception as e:
                    logger.warning(f"[!] 特征解析警告 => 处理图片失败 {os.path.basename(path)}: {e}")
                    global_features.append(torch.zeros(self.feature_dim).to(self.device))
                    if self.model_type in ["dinov2_vits14", "dinov2_vitb14"]:
                        # 对于 518x518，patch size = 14，所以是 37x37 = 1369 个 patch
                        local_features.append(torch.zeros((1369, self.feature_dim)).to(self.device))
        
        if len(global_features) == 0:
            g_empty = torch.empty((0, self.feature_dim)).to(self.device)
            if self.model_type in ["dinov2_vits14", "dinov2_vitb14"]:
                return {"global": g_empty, "local": torch.empty((0, 1369, self.feature_dim)).to(self.device)}
            return {"global": g_empty, "local": None}
            
        res = {"global": torch.stack(global_features)}
        if self.model_type in ["dinov2_vits14", "dinov2_vitb14"]:
            res["local"] = torch.stack(local_features)
        else:
            res["local"] = None
            
        return res

    def extract_features_tiled(self, image_paths: List[str], crop_size=518, overlap_ratio=0.25) -> List[Dict[str, torch.Tensor]]:
        """
        使用优化后的滑动窗口切片提取特征。
        提速策略：
        1. 适度预缩放 (限制最长边为 1920)。
        2. 降低重叠率至 0.25。
        3. 开启 FP16 半精度推理 (Autocast)。
        4. 增大推理 Batch Size。
        """
        results = []
        use_autocast = self.device.type == 'cuda'
        
        # 兼容旧版本 PyTorch
        if hasattr(torch, 'amp') and hasattr(torch.amp, 'autocast'):
            autocast_ctx = torch.amp.autocast(device_type='cuda', enabled=use_autocast)
        else:
            autocast_ctx = torch.cuda.amp.autocast(enabled=use_autocast)

        with torch.no_grad():
            for path in image_paths:
                try:
                    with Image.open(path) as img:
                        img_rgb = img.convert('RGB')
                        
                    # 提速策略 1：适度预缩放 (限制最长边为 1920)
                    w, h = img_rgb.size
                    max_side_limit = 1920
                    if max(w, h) > max_side_limit:
                        scale = max_side_limit / max(w, h)
                        new_w, new_h = int(w * scale), int(h * scale)
                        resample_method = getattr(Image, 'Resampling', Image).BICUBIC
                        img_rgb = img_rgb.resize((new_w, new_h), resample=resample_method)
                        w, h = new_w, new_h

                    # 获取切片
                    if w <= crop_size and h <= crop_size:
                        crops = [img_rgb]
                    else:
                        # 提速策略 2：更合理的重叠率
                        stride = int(crop_size * (1 - overlap_ratio))
                        crops = []
                        y_coords = list(range(0, h - crop_size + 1, stride))
                        if not y_coords or y_coords[-1] != h - crop_size:
                            y_coords.append(max(0, h - crop_size))
                            
                        x_coords = list(range(0, w - crop_size + 1, stride))
                        if not x_coords or x_coords[-1] != w - crop_size:
                            x_coords.append(max(0, w - crop_size))
                            
                        for y in y_coords:
                            for x in x_coords:
                                crops.append(img_rgb.crop((x, y, x + crop_size, y + crop_size)))
                                
                    # 提速策略 3 & 4：FP16 推理 + 增大 Batch Size (32)
                    batch_size = 32
                    img_global_feats = []
                    img_local_feats = []
                    
                    for i in range(0, len(crops), batch_size):
                        batch_crops = crops[i:i+batch_size]
                        tensors = [self.transform(c) for c in batch_crops]
                        batch_t = torch.stack(tensors).to(self.device)
                        
                        with autocast_ctx:
                            if self.model_type in ["dinov2_vits14", "dinov2_vitb14"]:
                                ret = self.model.forward_features(batch_t)
                                img_global_feats.append(ret['x_norm_clstoken'])
                                img_local_feats.append(ret['x_norm_patchtokens'])
                            else:
                                feat = self.model(batch_t)
                                img_global_feats.append(feat)
                            
                    if len(img_global_feats) > 0:
                        g_feat = torch.cat(img_global_feats, dim=0)
                        res = {"global": g_feat}
                        if self.model_type in ["dinov2_vits14", "dinov2_vitb14"]:
                            l_feat = torch.cat(img_local_feats, dim=0)
                            res["local"] = l_feat
                        else:
                            res["local"] = None
                        results.append(res)
                    else:
                        raise ValueError("No crops generated")
                        
                except Exception as e:
                    logger.warning(f"[!] 切片特征解析警告 => 处理图片失败 {os.path.basename(path)}: {e}")
                    g_empty = torch.zeros((1, self.feature_dim)).to(self.device)
                    res = {"global": g_empty}
                    if self.model_type in ["dinov2_vits14", "dinov2_vitb14"]:
                        res["local"] = torch.zeros((1, 1369, self.feature_dim)).to(self.device)
                    else:
                        res["local"] = None
                    results.append(res)
                    
        return results
