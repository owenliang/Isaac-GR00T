# 【中文】Gr00tN1d6 的 Processor 与 DataCollator 实现：负责将原始多模态数据
# 【中文】（图像、语言指令、状态、动作）转换为模型可直接消费的张量格式
import json
import os
from pathlib import Path
import re
from typing import Any, Dict, Literal
import warnings

import albumentations as A
from gr00t.configs.data.embodiment_configs import ModalityConfig
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.interfaces import BaseProcessor
from gr00t.data.state_action.state_action_processor import StateActionProcessor
from gr00t.data.utils import parse_modality_configs, to_json_serializable
import numpy as np
from PIL import Image
import torch
import torchvision.transforms.v2 as transforms
from transformers import AutoProcessor, ProcessorMixin
from transformers.feature_extraction_utils import BatchFeature
from transformers.utils import cached_file

from .image_augmentations import (
    apply_with_replay,
    build_image_transformations,
    build_image_transformations_albumentations,
)


# Suppress protobuf deprecation warnings
# 【中文】屏蔽 protobuf 的废弃警告，避免训练日志被无关信息刷屏
warnings.filterwarnings("ignore", category=DeprecationWarning, module="google.protobuf")

### Mapping from embodiment tag to projector index.
# 【中文】embodiment_tag 到 projector 索引的映射：
# 【中文】同一具身形态在 VLM projector 中使用固定的 index，用于选择对应的投影头
EMBODIMENT_TAG_TO_PROJECTOR_INDEX = {
    ##### Pretrain embodiment ids #####
    "robocasa_panda_omron": 13,
    "gr1": 20,
    "behavior_r1_pro": 24,
    ##### Pre-registered posttrain embodiment ids #####
    "unitree_g1": 8,
    "libero_panda": 2,
    "oxe_google": 0,
    "oxe_widowx": 1,
    "new_embodiment": 10,
}


def build_processor(model_name: str, transformers_loading_kwargs: dict) -> ProcessorMixin:
    """Build VLM processor for Eagle backbone.
    【中文】根据模型名称构建视觉-语言处理器，目前仅支持 Eagle-Block2A-2B-v2。
    """
    assert model_name == "nvidia/Eagle-Block2A-2B-v2", f"Processor for {model_name} not supported"
    eagle_path = os.path.join(
        os.path.dirname(__file__), "..", "modules", "nvidia", "Eagle-Block2A-2B-v2"
    )
    return AutoProcessor.from_pretrained(eagle_path, **transformers_loading_kwargs)


class Gr00tN1d6DataCollator:
    """Data collator for Gr00tN1d6.
    【中文】数据打包器：
    【中文】- 接收 processor 预处理后的多模态特征
    【中文】- 负责将同一 batch 内样本按 key 对齐并拼接成张量
    【中文】- 专门处理 `vlm_content`，调用 VLM 的 processor 做文本+图像编码
    
    【核心职责】
    DataCollator 是 DataLoader 和模型之间的桥梁：
    1. 将列表形式的样本（来自 Dataset）转换为批量张量
    2. 对不同模态使用不同的拼接策略：
       - VLM 内容：需要特殊处理文本 padding 和图像堆叠
       - 状态/动作：直接用 numpy.stack 堆叠即可
    3. 确保 batch 内所有样本的张量维度对齐
    """
    def __init__(
        self,
        model_name: str,
        model_type: Literal["eagle"] = "eagle",
        transformers_loading_kwargs: dict = {},
    ):
        # ==================== 初始化 VLM Processor ====================
        # 【必须使用相同 processor】与模型训练时保持一致，确保：
        # - tokenizer 的词表、special tokens 相同
        # - 图像预处理（resize、normalize）参数相同
        # - padding/truncation 策略一致
        self.processor = build_processor(model_name, transformers_loading_kwargs)
        
        # 【左侧 padding】Flash Attention 要求 padding 在左侧，因为：
        # - Flash Attention 从右向左处理序列
        # - 左侧 padding 可以让有效 token 在右侧连续排列，提高缓存命中率
        self.processor.tokenizer.padding_side = "left"
        
        self.model_type = model_type
        self.model_name = model_name

    def __call__(self, features: list[Dict[str, Any]]) -> BatchFeature:
        """Collate a list of feature dicts into a batched BatchFeature.
        【中文】将若干样本的特征字典对齐并打包成一个 batch：
        【中文】- 对 `vlm_content`：抽取文本+图像，调用 VLM processor 得到 input_ids / pixel_values 等
        【中文】- 对 state/action 等数值模态：用 numpy.stack → torch.from_numpy 堆叠
        
        【工作流程】
        输入：[{"vlm_content": ..., "state": ..., "action": ...}, ...] (batch_size 个样本)
        输出：{"input_ids": Tensor[B,L], "pixel_values": Tensor[B,N,C,H,W], 
               "state": Tensor[B,D_state], "action": Tensor[B,T,D_action], ...}
        """
        batch = {}
        # ==================== 步骤 1: 收集所有样本的 key ====================
        # 【处理可选字段】不同样本可能包含不同的 key（如训练有 action，推理没有）
        # 使用 set.union 找出所有样本中出现过的 key
        keys = list(set().union(*(elem.keys() for elem in features)))

        # ==================== 步骤 2: 逐 key 拼接 ====================
        for key in keys:
            # 【提取当前 key 的所有值】跳过不包含该 key 的样本
            values = [elem[key] for elem in features if key in elem]
            if key == "vlm_content":
                # ==================== VLM 内容特殊处理 ====================
                # 【为什么特殊】VLM 需要同时处理文本和图像，且需要 tokenization + padding
                # 【输入格式】每个样本的 vlm_content 包含：
                #   {"text": "pick up the cube", 
                #    "images": [PIL.Image, ...], 
                #    "conversation": [{"role": "user", "content": [...]}]}
                
                text_list = []      # 收集所有样本的文本
                image_inputs = []   # 收集所有样本的图像
                
                for v in values:
                    curr_text_list = [v["text"]]
                    text_list += curr_text_list
                    
                    curr_image_inputs = v["images"]
                    image_inputs += curr_image_inputs

                # 【Eagle VLM 特殊逻辑】
                # Eagle 需要从 conversation 中解析图像位置和特殊 token
                # 其他 VLM（如 LLaVA）可能直接使用 images 列表
                if self.model_type == "eagle":
                    image_inputs, _ = self.processor.process_vision_info(
                        [v["conversation"] for v in values]
                    )
                
                # 【VLM Processor 编码】
                # 输入：文本列表 + 图像列表
                # 输出：input_ids（文本token）、attention_mask、pixel_values（图像张量）等
                vlm_inputs = self.processor(
                    text=text_list, images=image_inputs, return_tensors="pt", padding=True
                )
                
                # 将 VLM processor 输出的所有字段加入 batch
                for k, v in vlm_inputs.items():
                    batch[k] = v
            elif key in ("pixel_values", "image_grid_thw", "attention_mask", "input_ids"):
                raise Exception("Not implemented")
            else:
                # ==================== 数值模态直接堆叠 ====================
                # 【适用于】state, state_mask, action, action_mask, embodiment_id 等
                # 【堆叠逻辑】
                # - 输入：[array[D], array[D], ...] (batch_size 个)
                # - np.stack：堆叠成 [B, D]
                # - torch.from_numpy：转为 PyTorch 张量
                # 
                # 【示例】如果 batch_size=4, state_dim=29：
                #   输入：4 个 shape=(29,) 的 numpy array
                #   输出：shape=(4, 29) 的 torch.Tensor
                batch[key] = torch.from_numpy(np.stack(values))
        return BatchFeature(data={"inputs": batch})

    def __str__(self):
        return f"Gr00tN1d6DataCollator(model_name={self.model_name}, model_type={self.model_type})"


class Gr00tN1d6Processor(BaseProcessor):
    """Processor for Gr00tN1d6 model.
    【中文】Gr00tN1d6 专用 Processor，负责：
    【中文】1）对状态/动作做归一化、相对动作转换等（借助 StateActionProcessor）
    【中文】2）对图像做裁剪、缩放、增强，并喂给 VLM 的 AutoProcessor
    【中文】3）在 `__call__` 中将一条轨迹打平成模型输入的张量字典
    """
    data_collator_class = Gr00tN1d6DataCollator

    def __init__(
        self,
        modality_configs: dict[str, dict[str, ModalityConfig]],
        statistics: dict[str, dict[str, dict[str, dict[str, list[float]]]]] | None = None,
        use_percentiles: bool = False,
        clip_outliers: bool = True,
        image_crop_size: list[int] = None,
        image_target_size: list[int] = None,
        shortest_image_edge: int = 512,
        crop_fraction: float = 0.95,
        random_rotation_angle: int | None = None,
        color_jitter_params: dict[str, float] | None = None,
        formalize_language: bool = True,
        model_name: str = "nvidia/Eagle-Block2A-2B-v2",
        model_type: Literal["eagle"] = "eagle",
        max_state_dim: int = 29,
        max_action_dim: int = 29,
        apply_sincos_state_encoding: bool = False,
        max_action_horizon: int = 40,
        use_albumentations: bool = False,
        use_relative_action: bool = False,
        embodiment_id_mapping: dict[str, int] | None = None,
        transformers_loading_kwargs: dict = {"trust_remote_code": True},
    ):
        self.modality_configs = parse_modality_configs(modality_configs)

        # Initialize StateActionProcessor for state/action normalization
        # 【中文】状态/动作处理器：负责归一化、百分位裁剪、相对动作 ↔ 绝对动作转换等
        self.state_action_processor = StateActionProcessor(
            modality_configs=modality_configs,
            statistics=statistics,
            use_percentiles=use_percentiles,
            clip_outliers=clip_outliers,
            apply_sincos_state_encoding=apply_sincos_state_encoding,
            use_relative_action=use_relative_action,
        )

        # Save state action processor settings
        self.use_percentiles = use_percentiles
        self.clip_outliers = clip_outliers
        self.apply_sincos_state_encoding = apply_sincos_state_encoding
        self.use_relative_action = use_relative_action

        # Save VLM settings
        # 【中文】保存与视觉-语言模型相关的配置（语言正规化、backbone 名称和类型）
        self.formalize_language = formalize_language
        self.model_name = model_name
        self.model_type = model_type

        self.max_state_dim = max_state_dim
        self.max_action_dim = max_action_dim
        self.max_action_horizon = max_action_horizon

        # Save image processing settings
        # 【中文】保存图像处理相关配置（裁剪/缩放/旋转/颜色抖动等）
        self.image_crop_size = image_crop_size
        self.image_target_size = image_target_size
        self.random_rotation_angle = random_rotation_angle
        self.color_jitter_params = color_jitter_params
        self.processor = build_processor(model_name, transformers_loading_kwargs)
        # Set padding side to 'left' for Flash Attention compatibility
        # 【中文】将 tokenizer 的 padding 设为左侧填充，以兼容 Flash Attention 的实现
        self.processor.tokenizer.padding_side = "left"
        self.embodiment_id_mapping = embodiment_id_mapping or EMBODIMENT_TAG_TO_PROJECTOR_INDEX
        # handle the case where the fine-tuning embodiment tag is not in the pre-trained embodiment tag mapping
        # 【中文】确保微调时的新具身形态也有 projector 索引：不存在的就沿用预训练中的默认映射
        for k, v in EMBODIMENT_TAG_TO_PROJECTOR_INDEX.items():
            if k not in self.embodiment_id_mapping:
                self.embodiment_id_mapping[k] = v
        self.shortest_image_edge = shortest_image_edge
        self.crop_fraction = crop_fraction

        # Choose between torchvision and albumentations transforms
        # 【中文】根据配置选择图像增强实现：albumentations（更灵活）或 torchvision.transforms
        self.use_albumentations = use_albumentations
        if use_albumentations:
            self.train_image_transform, self.eval_image_transform = (
                build_image_transformations_albumentations(
                    image_target_size,
                    image_crop_size,
                    random_rotation_angle,
                    color_jitter_params,
                    shortest_image_edge,
                    crop_fraction,
                )
            )
        else:
            self.train_image_transform, self.eval_image_transform = build_image_transformations(
                image_target_size, image_crop_size, random_rotation_angle, color_jitter_params
            )
        self._collator = self.data_collator_class(
            model_name=model_name,
            model_type=model_type,
            transformers_loading_kwargs=transformers_loading_kwargs,
        )
        self.train()

    @property
    def collator(self):
        return self._collator

    def train(self):
        super().train()
        self.state_action_processor.train()

    def eval(self):
        super().eval()
        self.state_action_processor.eval()

    def set_statistics(
        self,
        statistics: dict[str, dict[str, dict[str, dict[str, list[float]]]]],
        override: bool = False,
    ) -> None:
        """Set dataset statistics for normalization.
        【中文】设置数据集的统计信息（均值/方差/百分位等），用于状态/动作的归一化和反归一化。
        """
        self.state_action_processor.set_statistics(statistics, override=override)

        # Compute action dimensions for convenience
        self.action_dim = {}
        for embodiment_tag in self.state_action_processor.statistics:
            self.action_dim[embodiment_tag] = self.state_action_processor.get_action_dim(
                embodiment_tag
            )

    def decode_action(
        self,
        action: np.ndarray,
        embodiment_tag: EmbodimentTag,
        state: dict[str, np.ndarray] | None = None,
    ):
        """Undo action normalization and convert relative actions to absolute.
        【中文】对模型输出的动作进行反归一化，并在需要时将相对动作转换为绝对动作（关节角等）。
        """
        # Split concatenated action into joint groups
        out_dict = {}
        start_idx = 0
        joint_groups = self.modality_configs[embodiment_tag.value]["action"].modality_keys
        action_horizon = len(self.modality_configs[embodiment_tag.value]["action"].delta_indices)
        for key in joint_groups:
            joint_dim = self.state_action_processor.norm_params[embodiment_tag.value]["action"][
                key
            ]["dim"].item()
            out_dict[key] = action[..., :action_horizon, start_idx : start_idx + joint_dim]
            start_idx += joint_dim

        # Use StateActionProcessor to unnormalize and convert to absolute
        return self.state_action_processor.unapply_action(
            out_dict, embodiment_tag.value, state=state
        )

    def _apply_vlm_processing(self, images: np.ndarray, language: str) -> BatchFeature:
        """Prepare VLM inputs from video frames and language.
        【中文】将一段视频帧和语言指令封装成 VLM 期望的对话格式（conversation + text + images），
        【中文】并返回给 collator 统一处理。
        """
        """
        Args:
            batch:
                video: [T, C, H, W]
        Returns: vlm_content format for collation
        """
        # Convert images to PIL format
        pil_images = [Image.fromarray(np.transpose(v, (1, 2, 0))) for v in images]

        # Create conversation with images and text
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": language},
                    *[{"type": "image", "image": img} for img in pil_images],
                ],
            }
        ]

        # Apply chat template but don't process yet - let collator handle it
        text = self.processor.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=False
        )

        # Return vlm_content format for collation
        return {
            "vlm_content": {
                "text": text,
                "images": pil_images,
                "conversation": conversation,
            }
        }

    def __call__(
        self,
        messages: list[dict[str, Any]],
    ):
        """Main entry to process a single trajectory message.
        【中文】Processor 的主入口：将一条包含图像、文本、状态、动作的消息，
        【中文】转换为模型 forward 需要的 `state`/`action`/`action_mask`/VLM 输入等张量字典。
        
        【处理流程总览】
        输入：messages = [{
            "content": {
                "embodiment": EmbodimentTag,
                "states": {"joint_pos": [...], "gripper": [...]},
                "actions": {"joint_pos": [...], "gripper": [...]},
                "images": {"wrist_cam": [PIL.Image, ...], "front_cam": [...]},
                "text": "pick up the red cube"
            }
        }]
        
        输出：{
            "state": Tensor[T, max_state_dim],
            "action": Tensor[max_action_horizon, max_action_dim],
            "action_mask": Tensor[max_action_horizon, max_action_dim],
            "vlm_content": {"text": str, "images": [PIL.Image], "conversation": [...]},
            "embodiment_id": int
        }
        """
        assert len(messages) == 1
        content = messages[0]["content"]
        
        # ==================== 提取原始数据 ====================
        embodiment_tag = content.embodiment   # 具身形态标签（如 "gr1", "libero_panda"）
        action_data = content.actions         # 原始动作数据（未归一化）
        state_data = content.states           # 原始状态数据（未归一化）

        # ==================== 步骤 1: 状态/动作归一化与转换 ====================
        # 【StateActionProcessor 职责】
        # 1. 归一化：使用预先计算的均值/方差将状态和动作缩放到 [-1, 1] 或标准正态分布
        # 2. 相对动作转换（可选）：
        #    - 训练时：absolute_action → relative_action (delta = action - state)
        #    - 推理时：relative_action → absolute_action (action = state + delta)
        # 3. Sin/Cos 编码（可选）：对关节角度等周期性状态应用三角函数编码
        # 
        # 【输入】原始的 joint_pos, gripper_pos 等字典
        # 【输出】归一化后的字典，相同的 key 结构
        normalized_states, normalized_actions = self.state_action_processor.apply(
            state=state_data,
            action=action_data,
            embodiment_tag=embodiment_tag.value,
        )

        # ==================== 步骤 2: 动作拼接与 Padding ====================
        if normalized_actions:
            # 【拼接动作模态】
            # 不同具身形态的动作由多个模态组成，例如：
            # - gr1: ["joint_pos": 14-dim, "gripper": 2-dim] → 16-dim
            # - panda: ["joint_pos": 7-dim, "gripper": 1-dim] → 8-dim
            # 需要按配置的顺序拼接成单一向量
            action_keys = self.modality_configs[embodiment_tag.value]["action"].modality_keys
            normalized_actions = torch.cat(
                [torch.from_numpy(normalized_actions[key]) for key in action_keys], dim=-1
            )  # (action_horizon, action_dim)
            
            action_dim = normalized_actions.shape[1]      # 当前具身的实际动作维度
            action_horizon = normalized_actions.shape[0]  # 当前样本的实际时间步数
            
            # 【Padding 到 max_action_dim】
            # 【目的】不同具身的动作维度不同（如 gr1=16, panda=8），padding 到统一维度便于批处理
            # 【策略】右侧用 0 填充，mask 标记有效维度
            # 示例：如果 action_dim=8, max_action_dim=29
            #      原始 shape: (T, 8) → padding 后: (T, 29)，其中 [:, 8:] 全为 0
            normalized_actions = torch.cat(
                [
                    normalized_actions,
                    torch.zeros(
                        normalized_actions.shape[0],
                        self.max_action_dim - normalized_actions.shape[1],
                    ),
                ],
                dim=-1,
            )  # (action_horizon, max_action_dim)
            
            # 【Padding 到 max_action_horizon】
            # 【目的】不同样本的 action horizon 不同（如 10 步、20 步），统一到最大值
            # 【策略】下方用 0 填充时间维度
            # 示例：如果 action_horizon=10, max_action_horizon=40
            #      原始 shape: (10, 29) → padding 后: (40, 29)，其中 [10:, :] 全为 0
            normalized_actions = torch.cat(
                [
                    normalized_actions,
                    torch.zeros(
                        self.max_action_horizon - normalized_actions.shape[0],
                        self.max_action_dim,
                    ),
                ],
                dim=0,
            )  # (max_action_horizon, max_action_dim)
            
            # 【创建 action_mask】
            # mask[t, d] = 1 表示该位置是有效动作，= 0 表示是 padding
            # - mask[action_horizon:, :] = 0：时间维度超出的部分无效
            # - mask[:, action_dim:] = 0：动作维度超出的部分无效
            action_mask = torch.ones_like(normalized_actions)
            action_mask[action_horizon:] = 0    # 无效时间步
            action_mask[:, action_dim:] = 0     # 无效动作维度
        else:
            assert not self.training, "Action is required in training mode"
            normalized_actions = None
            action_mask = None

        # ==================== 步骤 3: 状态拼接与 Padding ====================
        # 【拼接状态模态】
        # 状态也由多个模态组成，例如：
        # - gr1: ["joint_pos": 14-dim, "joint_vel": 14-dim, "gripper": 2-dim] → 30-dim
        # - panda: ["joint_pos": 7-dim, "gripper": 1-dim] → 8-dim
        state_keys = self.modality_configs[embodiment_tag.value]["state"].modality_keys
        normalized_states = torch.cat(
            [torch.from_numpy(normalized_states[key]) for key in state_keys], dim=-1
        )  # (T, state_dim)
        
        # 【Padding 到 max_state_dim】
        # 同动作一样，右侧用 0 填充到统一维度
        # 示例：如果 state_dim=8, max_state_dim=29
        #      原始 shape: (T, 8) → padding 后: (T, 29)
        # 注意：状态不需要 time padding，因为每个 timestep 都需要状态输入
        normalized_states = torch.cat(
            [
                normalized_states,
                torch.zeros(
                    normalized_states.shape[0], self.max_state_dim - normalized_states.shape[1]
                ),
            ],
            dim=-1,
        )  # (T, max_state_dim)

        # ==================== 步骤 4: 图像预处理 ====================
        # 【训练 vs 评估的区别】
        # - 训练：应用数据增强（随机裁剪、旋转、颜色抖动等）提高鲁棒性
        # - 评估：只做中心裁剪和缩放，保持确定性
        if self.training:
            image_transform = self.train_image_transform
        else:
            image_transform = self.eval_image_transform
        
        # 【提取图像视角】
        # 不同具身形态有不同的相机配置，例如：
        # - gr1: ["wrist_cam", "front_cam"] (2个视角)
        # - panda: ["wrist_cam"] (1个视角)
        image_keys = self.modality_configs[embodiment_tag.value]["video"].modality_keys

        # ==================== 步骤 5: 语言指令预处理 ====================
        if self.formalize_language:
            # 【语言正规化】统一文本格式，提高泛化能力
            # 示例："Pick up the cube!" → "pick up the cube"
            #      "Grasp object." → "grasp object"
            language = content.text.lower()               # 转小写
            language = re.sub(r"[^\w\s]", "", language)  # 移除标点符号
        else:
            language = content.text

        # ==================== 步骤 6: 构建 VLM 输入 ====================
        # 【输入】图像列表 + 语言指令
        # 【输出】{"vlm_content": {"text": str, "images": [...], "conversation": [...]}}
        vlm_inputs = self._get_vlm_inputs(
            image_keys=image_keys,
            images=content.images,
            image_transform=image_transform,
            language=language,
        )

        # ==================== 步骤 7: 组装最终输出 ====================
        transformed_inputs = {
            "state": normalized_states.to(torch.get_default_dtype()),
        }
        
        # 【可选：动作】
        # 训练时必须有动作（用于计算损失），推理时可以没有（模型预测动作）
        if normalized_actions is not None:
            transformed_inputs["action"] = normalized_actions.to(torch.get_default_dtype())
        
        # 【添加 VLM 输入】包含 vlm_content 字典
        transformed_inputs.update(vlm_inputs)
        
        # 【可选：动作 mask】标记有效动作的位置
        if action_mask is not None:
            transformed_inputs["action_mask"] = action_mask
        
        # 【具身形态 ID】
        # 映射为整数索引，模型根据此 ID 选择对应的：
        # - 视觉 projector（不同具身可能需要不同的视觉特征投影）
        # - 动作头 MLP（不同具身的动作空间不同）
        # 示例："gr1" → 20, "libero_panda" → 2
        transformed_inputs["embodiment_id"] = self.embodiment_id_mapping[embodiment_tag.value]
        
        return transformed_inputs

    def _get_vlm_inputs(
        self,
        image_keys: list[str],
        images: list[Image.Image],
        image_transform: transforms.Compose | A.Compose,
        language: str,
    ):
        """处理多视角视频序列并构建 VLM 输入。
        
        【输入格式】
        images = {
            "wrist_cam": [PIL.Image(t=0), PIL.Image(t=1), ...],  # T 帧
            "front_cam": [PIL.Image(t=0), PIL.Image(t=1), ...],  # T 帧
        }
        
        【输出格式】
        temporal_stacked_images = {
            "wrist_cam": Tensor[T, 3, H, W],
            "front_cam": Tensor[T, 3, H, W],
        }
        最终拼接成：Tensor[T*V, 3, H, W] 送入 VLM
        """
        temporal_stacked_images = {}

        if self.use_albumentations:
            # ==================== Albumentations 增强（带 Replay） ====================
            # 【Replay 机制的必要性】
            # 当有多个视角时（如 wrist_cam + front_cam），需要确保：
            # 1. 同一时刻的不同视角应用相同的随机增强（如相同的裁剪位置、旋转角度）
            # 2. 不同时刻可以有不同的随机增强
            # 
            # 【实现】首次变换时记录随机参数（replay），后续视角复用这些参数
            replay = None
            for view in image_keys:
                assert view in images, f"{view} not in {images}"
                
                # apply_with_replay 会返回：(变换后的图像列表, 随机参数记录)
                # 第一个视角：replay=None，生成新的随机参数
                # 后续视角：replay!=None，复用之前的随机参数
                transformed_images, replay = apply_with_replay(
                    image_transform, images[view], replay
                )
                temporal_stacked_images[view] = torch.stack(transformed_images)  # (T, C, H, W)
        else:
            # ==================== Torchvision 增强（无 Replay） ====================
            # 【局限性】torchvision 不支持 replay，每个视角的随机增强是独立的
            # 【适用场景】单视角任务，或对多视角一致性要求不高的场景
            for view in image_keys:
                assert view in images, f"{view} not in {images}"
                temporal_stacked_images[view] = torch.stack(
                    [image_transform(img) for img in images[view]]
                )  # (T, C, H, W)

        # ==================== 数据验证 ====================
        # 【确保数据格式正确】防止后续处理出错
        for k, v in temporal_stacked_images.items():
            assert isinstance(k, str), f"{k} is not a string"
            assert isinstance(v, torch.Tensor), f"{v} is not a torch tensor"
            assert v.ndim == 4, f"{v} is not a 4D tensor"  # (T, C, H, W)
            assert v.dtype == torch.uint8, f"{v} is not a uint8 tensor"  # 图像像素值 [0, 255]
            assert v.shape[1] == 3, f"{v} is not a 3 channel tensor"  # RGB 3通道

        # ==================== 多视角时空拼接 ====================
        # 【拼接逻辑】
        # 输入：{"wrist": [T,3,H,W], "front": [T,3,H,W]}
        # 
        # 步骤 1：torch.stack([...], dim=1) → [T, V, 3, H, W]
        #   其中 V 是视角数量（如 2 个相机）
        # 
        # 步骤 2：.flatten(0, 1) → [T*V, 3, H, W]
        #   将时间和视角维度展平，方便 VLM 处理
        #   示例：T=10, V=2 → 20 张图像
        # 
        # 步骤 3：.numpy() → Eagle VLM 的 processor 期望 numpy array 输入
        stacked_images = (
            torch.stack([temporal_stacked_images[view] for view in image_keys], dim=1)
            .flatten(0, 1)
            .numpy()
        )  # (T*V, C, H, W), Eagle processor expects numpy array

        # ==================== 调用 VLM 预处理 ====================
        # 【输入】stacked_images + language
        # 【输出】{"vlm_content": {"text": str, "images": [...], "conversation": [...]}}
        vlm_inputs = self._apply_vlm_processing(stacked_images, language)
        return vlm_inputs

    def save_pretrained(self, save_directory: str | Path) -> list[Path]:
        """Save processor config, statistics and embodiment mapping to disk.
        【中文】将 Processor 的配置、统计信息和具身形态映射保存到磁盘，便于后续加载/推理复现。
        """
        # dump modality configs to dict using the recursive function
        save_directory.mkdir(parents=True, exist_ok=True)
        main_config_file = Path(save_directory) / "processor_config.json"
        statistics_file = Path(save_directory) / "statistics.json"
        embodiment_id_file = Path(save_directory) / "embodiment_id.json"

        config = {
            "processor_class": self.__class__.__name__,
            "processor_kwargs": {
                "modality_configs": to_json_serializable(self.modality_configs),
                # Image processing settings
                "image_crop_size": self.image_crop_size,
                "image_target_size": self.image_target_size,
                "use_albumentations": self.use_albumentations,
                "random_rotation_angle": self.random_rotation_angle,
                "color_jitter_params": self.color_jitter_params,
                "shortest_image_edge": self.shortest_image_edge,
                "crop_fraction": self.crop_fraction,
                # VLM settings
                "model_name": self.model_name,
                "model_type": self.model_type,
                "formalize_language": self.formalize_language,
                # State action dimensions
                "max_state_dim": self.max_state_dim,
                "max_action_dim": self.max_action_dim,
                "max_action_horizon": self.max_action_horizon,
                # StateActionProcessor settings
                "use_percentiles": self.use_percentiles,
                "clip_outliers": self.clip_outliers,
                "apply_sincos_state_encoding": self.apply_sincos_state_encoding,
                "use_relative_action": self.use_relative_action,
            },
        }
        with open(main_config_file, "w") as f:
            json.dump(config, f, indent=2)
        # Save statistics
        # 【中文】保存 StateActionProcessor 的统计信息，用于反归一化和 action 维度推断
        with open(statistics_file, "w") as f:
            json.dump(to_json_serializable(self.state_action_processor.statistics), f, indent=2)
        # Save embodiment id mapping
        # 【中文】保存具身形态到 projector 索引的映射，保证不同环境下具身编号一致
        with open(embodiment_id_file, "w") as f:
            json.dump(self.embodiment_id_mapping, f, indent=2)
        return [main_config_file, statistics_file, embodiment_id_file]

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str | Path, **kwargs):
        """Load processor from a saved directory or HF repo.
        【中文】从本地目录或远程仓库加载已保存的 Processor，并允许覆盖部分配置（如模态配置、数据增强）。
        """
        transformers_loading_kwargs = kwargs.pop(
            "transformers_loading_kwargs", {"trust_remote_code": True}
        )
        pretrained_model_name_or_path = Path(pretrained_model_name_or_path)
        config_file = pretrained_model_name_or_path / "processor_config.json"
        statistics_file = pretrained_model_name_or_path / "statistics.json"
        embodiment_id_file = pretrained_model_name_or_path / "embodiment_id.json"
        is_local = os.path.isdir(pretrained_model_name_or_path)
        if not is_local:
            config_file = Path(cached_file(pretrained_model_name_or_path, "processor_config.json"))
            statistics_file = Path(cached_file(pretrained_model_name_or_path, "statistics.json"))
            embodiment_id_file = Path(
                cached_file(pretrained_model_name_or_path, "embodiment_id.json")
            )

        with open(config_file, "r") as f:
            config = json.load(f)
        with open(statistics_file, "r") as f:
            statistics = json.load(f)
        if embodiment_id_file.exists():
            with open(embodiment_id_file, "r") as f:
                embodiment_id_mapping = json.load(f)
        else:
            embodiment_id_mapping = None
        processor_kwargs = config["processor_kwargs"]
        processor_kwargs["statistics"] = statistics
        processor_kwargs["embodiment_id_mapping"] = embodiment_id_mapping
        # Directly override other processor kwargs
        if kwargs:
            # Override modality configs while keeping pretrained embodiment configs
            # 【中文】允许在微调时覆盖部分 processor 参数：
            # 【中文】- modality_configs：替换/新增具身形态的模态配置
            # 【中文】- random_rotation_angle / color_jitter_params / use_relative_action：根据当前任务调整增强和动作表示
            modality_configs = kwargs.pop("modality_configs", {})
            for embodiment_tag, modality_config in modality_configs.items():
                processor_kwargs["modality_configs"][embodiment_tag] = modality_config
            override_keys = [
                "random_rotation_angle",
                "color_jitter_params",
                "use_relative_action",
            ]
            for key in override_keys:
                if key in kwargs:
                    override = kwargs.pop(key)
                    if override is not None:
                        processor_kwargs[key] = override
        return cls(**processor_kwargs, transformers_loading_kwargs=transformers_loading_kwargs)


# 【中文】将 Gr00tN1d6Processor 注册到 Transformers 的 AutoProcessor，方便通过名称自动构建
AutoProcessor.register("Gr00tN1d6", Gr00tN1d6Processor)
