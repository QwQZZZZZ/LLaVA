import torch
import torch.nn as nn

from transformers import AutoModel, AutoImageProcessor, AutoConfig


class ResNetVisionTower(nn.Module):
    """
    一个高度解耦的ResNet视觉编码器类。
    它负责加载ResNet模型，并将最后一层的特征图转换为LLaVA期望的序列格式。
    """

    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False
        self.vision_tower_name = vision_tower
        # ResNet没有像ViT那样的明确分层，我们通常使用最后一个stage的输出
        # 此参数mm_vision_select_layer将被我们的实现忽略，但保留以保持接口一致性
        self.select_layer = getattr(args, 'mm_vision_select_layer', -1)
        # 此参数对于CNN特征图没有'cls' token的概念，因此我们将忽略
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')

        if not delay_load:
            self.load_model()
        else:
            self.cfg_only = AutoConfig.from_pretrained(self.vision_tower_name)

    def load_model(self):
        if self.is_loaded:
            print(f'{self.vision_tower_name} is already loaded, `load_model` called again, skipping.')
            return

        print(f"Loading ResNet Vision Tower from {self.vision_tower_name}")
        # 使用 AutoImageProcessor 和 AutoModel 加载
        self.image_processor = AutoImageProcessor.from_pretrained(self.vision_tower_name)
        # 我们需要模型输出中间层的特征图(hidden_states)
        self.vision_tower = AutoModel.from_pretrained(self.vision_tower_name,
                                                      output_hidden_states=True)
        self.vision_tower.requires_grad_(False)  # 默认冻结

        self.is_loaded = True
        print("ResNet Vision Tower loaded successfully.")

    def feature_select(self, image_forward_outs):
        # 对于HuggingFace的ResNet模型, `hidden_states` 是一个元组，包含了从嵌入层到每个stage输出的特征图
        # 最后一项 `hidden_states[-1]` 就是最后一个卷积stage的输出特征图
        return image_forward_outs.hidden_states[-1]

    @torch.no_grad()
    def forward(self, images):
        # 1. 提取特征图
        # 将图像传递给ResNet模型
        image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype))
        # 选择最后一个stage的输出特征图
        image_features = self.feature_select(image_forward_outs).to(images.dtype)

        # 2. 将特征图转换为序列格式
        # ResNet输出的特征图形状为 (batch_size, num_channels, height, width)
        # LLaVA的projector期望的输入是 (batch_size, sequence_length, hidden_size)
        # 因此，我们需要进行转换
        batch_size, num_channels, height, width = image_features.shape
        # 将 (B, C, H, W) -> (B, C, H*W)
        image_features = image_features.flatten(2)
        # 将 (B, C, H*W) -> (B, H*W, C) 以匹配序列格式
        image_features = image_features.permute(0, 2, 1).contiguous()

        return image_features

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        # 对于ResNet, hidden_size是最后一个stage输出的通道数
        # 例如，resnet-50 是 2048
        # 我们可以从config的 `hidden_sizes` 列表的最后一项获取
        return self.config.hidden_sizes[-1]

    @property
    def num_patches(self):
        # 对于CNN，我们可以将最后一个特征图的每个像素视为一个“patch”
        # 这个值取决于输入图像大小和模型的下采样率
        # 例如，对于224x224的输入，resnet-50的最后特征图是7x7，所以有49个"patches"
        # 这是一个动态计算的近似值，但对于LLaVA的拼接逻辑是足够的
        # 假设下采样率为32 (224 / 7 = 32)
        # image_size = self.image_processor.size['shortest_edge']
        # patch_size = 32 # ResNet-like models have a total stride of 32
        # return (image_size // patch_size) ** 2
        # 为了简单和稳定，可以直接返回一个典型值，比如 49 (7*7)，因为LLaVA主要关心序列长度
        return 49
