import os
import logging
from typing import List, Dict

# 导入本地同级模块
from .features import FeatureExtractor
from .matching import get_best_match_mapping_many_to_one

logger = logging.getLogger(__name__)

class SceneMatcherAPI:
    """
    场景素材匹配核心算法 API
    
    提供极为简洁的方法进行高精度的图片最佳匹配。
    主要用于在海量白底素材库中，寻找最匹配“特定场景图”透视角度和局部元素的素材。
    """
    
    def __init__(self, model_type: str = "dinov2_vitb14"):
        """
        初始化匹配引擎。首次运行时会自动下载权重并缓存到 models 目录下。
        
        :param model_type: 使用的视觉底层模型
                           推荐 "dinov2_vitb14" (高精度, 默认)
                           可选 "dinov2_vits14" (高精度, 较快)
                           可选 "efficientnet_b0" (轻量级, 极速)
        """
        self.model_type = model_type
        # 初始化底层特征提取器，处理所有的设备放置和缓存逻辑
        self.extractor = FeatureExtractor(model_type=model_type)

    def match_images(self, scene_paths: List[str], material_paths: List[str], top_k: int = 1) -> Dict[str, List[str]]:
        """
        核心匹配方法。输入两组图片路径，返回每个场景图对应的最佳素材路径。
        
        :param scene_paths: 场景图片(Target) 的绝对或相对路径列表
        :param material_paths: 白底素材图片(Template) 的绝对或相对路径列表
        :param top_k: 每个场景图返回最相似的 K 张素材
        :return: 字典结构。键为场景图路径，值为最佳匹配的素材路径列表
                 例如: {'scene1.jpg': ['material_top1.jpg', 'material_top2.jpg']}
        """
        if not scene_paths or not material_paths:
            logger.warning("场景图或素材图路径列表为空，无法进行匹配。")
            return {}

        logger.info(f"开始提取 {len(scene_paths)} 张场景图特征 (启用滑动窗口防分辨率丢失)...")
        # 场景图通常较大，使用切片模式提取
        scene_features = self.extractor.extract_features_tiled(scene_paths)

        logger.info(f"开始提取 {len(material_paths)} 张素材图特征...")
        # 素材图直接提取
        material_features = self.extractor.extract_features(material_paths)

        logger.info(f"进行高维特征矩阵比对 (计算 Top-{top_k})...")
        # 执行不对称局部相似度计算
        match_dict = get_best_match_mapping_many_to_one(scene_features, material_features, top_k)

        # 将内部基于索引的字典转换为基于路径的文件名字典
        result = {}
        for scene_idx, material_indices in match_dict.items():
            scene_path = scene_paths[scene_idx]
            matched_materials = [material_paths[m_idx] for m_idx in material_indices]
            result[scene_path] = matched_materials

        return result
        
    def match_folders(self, scene_dir: str, material_dir: str, top_k: int = 1) -> Dict[str, List[str]]:
        """
        便利用法：直接提供两个文件夹路径，自动读取内部图片并执行匹配计算。
        
        :param scene_dir: 存放场景图的文件夹路径
        :param material_dir: 存放白底素材图的文件夹路径
        :param top_k: 每个场景图返回最相似的 K 张素材
        :return: 字典结构。键为场景图路径，值为最佳匹配的素材路径列表
        """
        valid_exts = ('.jpg', '.jpeg', '.png', '.webp', '.bmp')
        
        def get_images_from_dir(directory):
            if not os.path.isdir(directory):
                return []
            return sorted([
                os.path.abspath(os.path.join(directory, f)) 
                for f in os.listdir(directory) 
                if f.lower().endswith(valid_exts)
            ])
            
        scene_paths = get_images_from_dir(scene_dir)
        material_paths = get_images_from_dir(material_dir)
        
        return self.match_images(scene_paths, material_paths, top_k)
