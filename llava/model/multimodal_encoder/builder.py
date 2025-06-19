import os
from .clip_encoder import CLIPVisionTower, CLIPVisionTowerS2
from .resnet_encoder import ResNetVisionTower


# def build_vision_tower(vision_tower_cfg, **kwargs):
#     vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))
#     is_absolute_path_exists = os.path.exists(vision_tower)
#     use_s2 = getattr(vision_tower_cfg, 's2', False)
#     if is_absolute_path_exists or vision_tower.startswith("openai") or vision_tower.startswith("laion") or "ShareGPT4V" in vision_tower:
#         if use_s2:
#             return CLIPVisionTowerS2(vision_tower, args=vision_tower_cfg, **kwargs)
#         else:
#             return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
#
#     raise ValueError(f'Unknown vision tower: {vision_tower}')


def build_vision_tower(vision_tower_cfg, **kwargs):
    # 从配置中获取vision_tower的名称或路径
    vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))

    # 将模型名称转换为小写以进行稳健的检查
    vision_tower_name_lower = vision_tower.lower()

    # 检查是否是已知的模型类型或本地路径
    is_absolute_path_exists = os.path.exists(vision_tower)
    is_hf_model = vision_tower.startswith("openai") or vision_tower.startswith("laion") or vision_tower.startswith(
        "microsoft")

    # --- 工厂逻辑 ---
    if is_absolute_path_exists or is_hf_model or "mamba" in vision_tower_name_lower:
        # 1. 新增：检查是否是 ResNet 模型
        if "resnet" in vision_tower_name_lower:
            print(f"Building ResNetVisionTower for {vision_tower}")
            return ResNetVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

        # 2. 检查是否是 Mamba 模型 (保留上次的逻辑)
        # if "mamba" in vision_tower_name_lower:
        #     print(f"Building MambaVisionTower for {vision_tower}")
        #     return MambaVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

        # 3. 检查是否是 CLIP S2 模型 (保持原有逻辑)
        if 's2' in vision_tower_name_lower:
            print(f"Building CLIPVisionTowerS2 for {vision_tower}")
            return CLIPVisionTowerS2(vision_tower, args=vision_tower_cfg, **kwargs)

        # 4. 默认为标准的 CLIP 模型 (保持原有逻辑)
        print(f"Building CLIPVisionTower for {vision_tower}")
        return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

    # 如果以上条件都不满足，则抛出错误
    raise ValueError(f'Unknown or unsupported vision tower: {vision_tower}')
