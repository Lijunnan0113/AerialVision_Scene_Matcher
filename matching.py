import torch
from scipy.optimize import linear_sum_assignment
from torch.nn.functional import cosine_similarity
from typing import Dict, Union, Optional, List
import logging

logger = logging.getLogger(__name__)

def _get_global_features(features: Union[torch.Tensor, Dict[str, torch.Tensor]]) -> torch.Tensor:
    if isinstance(features, dict):
        return features.get("global", torch.empty(0))
    return features

def _calculate_patch_similarity_matrix(t_local: torch.Tensor, tpl_local: torch.Tensor) -> torch.Tensor:
    """
    计算 Target 和 Template 的局部 Patch 相似度得分矩阵。
    基于不对称局部匹配 (Asymmetric Partial Matching)，解决模糊、严重局部遮挡导致的匹配失败。
    """
    M = t_local.size(0)
    N = tpl_local.size(0)
    P = tpl_local.size(1)
    
    # 动态提取 Template 的前景掩码
    grid_size = int(P**0.5)
    corner_indices = [0, grid_size-1, P-grid_size, P-1]
    bg_refs = tpl_local[:, corner_indices, :] # (N, 4, D)
    bg_ref = bg_refs.mean(dim=1, keepdim=True) # (N, 1, D)
    sim_to_bg = cosine_similarity(tpl_local, bg_ref, dim=-1) # (N, P)
    fg_mask = sim_to_bg < 0.90 # (N, P) boolean mask
    
    score_matrix = torch.zeros((M, N))
    
    for i in range(M):
        tp = t_local[i] # (P, D)
        tp_norm = torch.nn.functional.normalize(tp, p=2, dim=-1) # (P, D)
        tpl_norm = torch.nn.functional.normalize(tpl_local, p=2, dim=-1) # (N, P, D)
        
        for j in range(N):
            # 获取当前白底图的有效前景 Patch
            fg_indices = torch.nonzero(fg_mask[j]).squeeze(1)
            if len(fg_indices) < int(P * 0.05):
                # 前景太小容错，取所有 Patch
                tpl_valid = tpl_norm[j]
                valid_indices = torch.arange(P, device=tpl_local.device)
            else:
                tpl_valid = tpl_norm[j][fg_indices]
                valid_indices = fg_indices
                
            # 计算目标所有 Patch 与 白底图前景 Patch 的相似度矩阵
            # sim: (P_tgt, N_fg)
            sim = torch.matmul(tp_norm, tpl_valid.transpose(0, 1))
            
            # 不对称局部验证 (Asymmetric Partial Matching)
            # 1. 站在“白底图”的视角，问它身上的每一个前景特征块，在场景图中能找到的最相似的块得分是多少？
            # 这样做极其抗模糊：就算场景图很模糊，只要在整张图里找到最像的一个点，这个分数就是有效的。
            max_sim_per_tpl_patch, _ = torch.max(sim, dim=0) # (N_fg,)
            
            # 2. 因为场景图中可能只有鞋底（局部遮挡），我们只取白底图匹配度最高的 Top 25% 的特征块来算平均分。
            # 如果白底图是“单纯的鞋底”，它的 Top 25% 全是鞋底，匹配得分极高。
            # 如果白底图是“完整的鞋”，它的 Top 25% 可能是鞋底+部分鞋面，鞋面匹配不到场景图的鞋底，平均分就会被严重拉低。
            # 这样就能完美分离出“单纯鞋底”或“局部部位”的素材！
            k = max(1, int(len(max_sim_per_tpl_patch) * 0.25))
            topk_sims, _ = torch.topk(max_sim_per_tpl_patch, k)
            
            score_matrix[i, j] = topk_sims.mean().item()
            
    return score_matrix

def get_optimal_mapping(target_features: Union[torch.Tensor, Dict[str, torch.Tensor]], 
                        template_features: Union[torch.Tensor, Dict[str, torch.Tensor]]) -> Dict[int, int]:
    t_global = _get_global_features(target_features)
    tpl_global = _get_global_features(template_features)
    
    M: int = t_global.size(0)
    N: int = tpl_global.size(0)
    
    if M == 0 or N == 0:
        logger.warning("[!] 参与比对的特征张量为空，无法进行多维算法比较。")
        return {}
        
    cost_matrix = torch.zeros((M, N))
    
    t_local = target_features.get("local") if isinstance(target_features, dict) else None
    tpl_local = template_features.get("local") if isinstance(template_features, dict) else None
    use_two_stage = (t_local is not None and tpl_local is not None)
    
    if use_two_stage:
        logger.info("[*] 启用局部 Patch 精细匹配计算权重矩阵 (抗干扰增强)...")
        t_local = t_local.to(tpl_global.device)
        tpl_local = tpl_local.to(tpl_global.device)
        score_matrix = _calculate_patch_similarity_matrix(t_local, tpl_local)
        cost_matrix = 1.0 - score_matrix # 分数越高，cost 越小
    else:
        for i in range(M):
            for j in range(N):
                cost = 1.0 - cosine_similarity(t_global[i].unsqueeze(0), tpl_global[j].unsqueeze(0)).item()
                cost_matrix[i, j] = cost
                
    row_ind, col_ind = linear_sum_assignment(cost_matrix.cpu().numpy())
    match_dict = {int(r): int(c) for r, c in zip(row_ind, col_ind)}
    
    return match_dict

def get_best_match_mapping_many_to_one(target_features: Union[torch.Tensor, Dict[str, torch.Tensor], List[Dict[str, torch.Tensor]]], 
                                       template_features: Union[torch.Tensor, Dict[str, torch.Tensor]],
                                       top_k: int = 1) -> Dict[int, List[int]]:
    tpl_global = _get_global_features(template_features)
    N = tpl_global.size(0)
    
    t_is_tiled = isinstance(target_features, list)
    M = len(target_features) if t_is_tiled else _get_global_features(target_features).size(0)
    
    if M == 0 or N == 0:
        logger.warning("[!] 参与比对的特征张量为空，无法进行相似度计算。")
        return {}
    top_k = min(top_k, N)
        
    tpl_local = template_features.get("local") if isinstance(template_features, dict) else None
    match_dict = {}
    
    if t_is_tiled:
        sample_t_local = target_features[0].get("local")
    else:
        sample_t_local = target_features.get("local") if isinstance(target_features, dict) else None
        
    use_two_stage = (sample_t_local is not None and tpl_local is not None)
    
    if use_two_stage:
        logger.info("[*] 启用局部 Patch 不对称特征匹配 (支持滑动窗口切片, 解决4K模糊/尺度坍塌)...")
        tpl_local = tpl_local.to(tpl_global.device)
        
    for i in range(M):
        if t_is_tiled:
            t_feat_i = target_features[i]
            t_global_i = _get_global_features(t_feat_i).to(tpl_global.device)
            t_local_i = t_feat_i.get("local")
            if t_local_i is not None:
                t_local_i = t_local_i.to(tpl_global.device)
        else:
            t_global = _get_global_features(target_features)
            t_global_i = t_global[i].unsqueeze(0).to(tpl_global.device)
            t_local = target_features.get("local") if isinstance(target_features, dict) else None
            t_local_i = t_local[i].unsqueeze(0).to(tpl_global.device) if t_local is not None else None

        if use_two_stage and t_local_i is not None:
            # 扁平化所有切片特征：(K_tiles, P, D) -> (1, K_tiles * P, D)
            # 使得素材前景可以跨越切片在整个场景图中寻找最佳匹配点
            t_local_i_flat = t_local_i.view(1, -1, tpl_local.size(-1))
            score_matrix_i = _calculate_patch_similarity_matrix(t_local_i_flat, tpl_local) # (1, N)
            final_scores_i = score_matrix_i.squeeze(0) # (N,)
            _, top_indices = torch.topk(final_scores_i, top_k)
            match_dict[i] = top_indices.tolist()
        else:
            sims = cosine_similarity(t_global_i.unsqueeze(1), tpl_global.unsqueeze(0), dim=-1) # (K_tiles, N)
            final_scores_i, _ = torch.max(sims, dim=0) # (N,)
            _, top_indices = torch.topk(final_scores_i, top_k)
            match_dict[i] = top_indices.tolist()
            
    return match_dict