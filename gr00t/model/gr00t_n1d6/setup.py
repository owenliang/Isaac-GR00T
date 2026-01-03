# Gr00tN1d6 模型的 Pipeline 实现
# 负责初始化模型、数据集、Processor 和 Collator

import json
import logging
from pathlib import Path

from gr00t.configs.base_config import Config
from gr00t.configs.model.gr00t_n1d6 import Gr00tN1d6Config
from gr00t.data.dataset.factory import DatasetFactory
from gr00t.experiment.dist_utils import get_rank
from gr00t.model.base.model_pipeline import ModelPipeline
from gr00t.model.gr00t_n1d6.gr00t_n1d6 import Gr00tN1d6
from gr00t.model.gr00t_n1d6.processing_gr00t_n1d6 import Gr00tN1d6Processor
from gr00t.model.registry import register_model
import numpy as np
from termcolor import colored
import torch
from transformers import AutoModel, AutoProcessor


# 工具函数：将 Tensor 转换为列表以便 JSON 序列化
def convert_tensors_to_lists(obj):
    """Recursively convert tensors to lists in nested dictionaries/lists."""
    if torch.is_tensor(obj) or isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_tensors_to_lists(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_tensors_to_lists(item) for item in obj]
    else:
        return obj


class Gr00tN1d6Pipeline(ModelPipeline):
    """
    Gr00t N1.6 模型的训练 Pipeline。
    关键职责：
    1. 从 checkpoint 加载预训练模型（或从头创建）
    2. 构建 Processor（处理图像/状态/动作等模态）
    3. 通过 DatasetFactory 构建数据集
    4. 返回 data collator 用于 batch 拼接
    """
    model_class = Gr00tN1d6
    processor_class = Gr00tN1d6Processor

    def __init__(self, config: Config, save_cfg_dir: Path):
        super().__init__(config)
        self.save_cfg_dir = save_cfg_dir

        # 构建 transformers 加载参数（用于 HuggingFace Hub 或本地加载）
        transformers_loading_kwargs = {
            "trust_remote_code": self.config.training.transformers_trust_remote_code,
            "local_files_only": self.config.training.transformers_local_files_only,
        }
        if self.model_config.model_revision is not None:
            transformers_loading_kwargs["revision"] = self.model_config.model_revision
        if self.config.training.transformers_cache_dir is not None:
            transformers_loading_kwargs["cache_dir"] = self.config.training.transformers_cache_dir
        if self.config.training.transformers_access_token is not None:
            transformers_loading_kwargs["token"] = self.config.training.transformers_access_token

        self.transformers_loading_kwargs = transformers_loading_kwargs

    @property
    def model_config(self):
        """返回模型配置（Gr00tN1d6Config）。"""
        return self.config.model

    def setup(self):
        """初始化 Pipeline 的核心组件：模型、数据集、collator。"""
        self.model = self._create_model()
        self.train_dataset, self.eval_dataset = self._create_dataset(self.save_cfg_dir)
        self.data_collator = self._create_collator()

    def _create_model(self):
        """
        构建 Gr00t N1.6 模型。
        关键逻辑：
        - 优先从 checkpoint 加载（fine-tuning 场景）
        - 根据 tune_* 参数决定哪些模块需要训练
        - 如缺失 mask_token，则随机初始化
        """

        # Build transformers loading kwargs from training config

        if self.config.training.start_from_checkpoint is not None:
            # 从 checkpoint 加载模型（fine-tuning 的典型路径）
            model, loading_info = AutoModel.from_pretrained(
                self.config.training.start_from_checkpoint,  # 【中文】预训练 checkpoint 路径或 HuggingFace 模型名称
                tune_llm=self.config.model.tune_llm,  # 【中文】是否微调语言模型（LLM）骨干
                tune_visual=self.config.model.tune_visual,  # 【中文】是否微调视觉编码器
                tune_projector=self.config.model.tune_projector,  # 【中文】是否微调多模态投影层
                tune_diffusion_model=self.config.model.tune_diffusion_model,  # 【中文】是否微调扩散动作头
                tune_vlln=self.config.model.tune_vlln,  # 【中文】是否微调视觉-语言-导航（VLLN）相关模块（如有）
                state_dropout_prob=self.config.model.state_dropout_prob,  # 【中文】训练时对状态输入施加的dropout比例
                backbone_trainable_params_fp32=self.config.model.backbone_trainable_params_fp32,  # 【中文】Backbone可训练参数是否使用FP32精度
                transformers_loading_kwargs=self.transformers_loading_kwargs,  # 【中文】传递给 transformers.from_pretrained 的通用加载参数
                output_loading_info=True,  # 【中文】返回 loading_info，包含missing_keys/unexpected_keys等
                **self.transformers_loading_kwargs,  # 【中文】解包其余 transformers 加载相关参数
            )

            # 如果 base checkpoint 中没有 mask_token，则需要初始化
            # 【中文】mask_token：与 state_dropout 搭配使用的“缺失状态”embedding，当某个样本的状态被drop时，用此向量整体替代真实状态
            # 【中文】旧版 checkpoint 可能不包含该参数，这里检测缺失并进行随机初始化以保持模型结构完整
            missing_keys = loading_info.get("missing_keys", [])
            mask_token_missing = any("mask_token" in key for key in missing_keys)

            if mask_token_missing and model.action_head.mask_token is not None:
                # Initialize mask_token
                # 【中文】使用小尺度高斯噪声随机初始化 mask_token 权重
                with torch.no_grad():
                    model.action_head.mask_token.data.copy_(
                        0.02 * torch.randn_like(model.action_head.mask_token)
                    )
                logging.info("mask_token not in checkpoint - initialized")

        else:
            # 从头创建模型（不常用）
            model = self.model_class(
                self.config.model, transformers_loading_kwargs=self.transformers_loading_kwargs
            )

        print(colored(f"Model Config: {model.config}", "yellow"))
        if get_rank() == 0:
            # 保存模型配置到文件
            with open(self.save_cfg_dir / "final_model_config.json", "w") as f:
                f.write(model.config.to_filtered_json())
        # 打印参数统计（总参数、可训练参数比例）
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logging.info(f"Total parameters: {total_params:,}")
        logging.info(
            f"Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)"
        )
        print("Model: ", model)

        return model

    def _get_statistics(self) -> dict[str, dict[str, dict[str, dict[str, list[float]]]]] | None:
        """返回预计算的统计信息（默认为 None，后续从数据集计算）。"""
        return None

    def _get_embodiment_id_mapping(self) -> dict[str, int]:
        """返回 embodiment ID 映射（默认为 None）。"""
        return None

    def _create_dataset(self, save_cfg_dir: Path):
        """
        构建数据集和 Processor。
        关键流程：
        1. 从 checkpoint 加载或创建新的 Processor
        2. Processor 封装了所有模态的预处理逻辑（图像、状态、动作）
        3. 通过 DatasetFactory 构建 shard-based 数据集
        4. 计算并保存数据集统计信息（用于推理时 denormalization）
        """

        if self.config.training.start_from_checkpoint is not None:
            # 从 checkpoint 加载 Processor（继承预训练模型的配置）
            processor = AutoProcessor.from_pretrained(
                self.config.training.start_from_checkpoint,
                # Overrides
                modality_configs=self.config.data.modality_configs,  # 【中文】各具身形态的模态配置（图像/状态/动作的key布局）
                image_crop_size=self.model_config.image_crop_size,  # 【中文】输入图像裁剪尺寸
                image_target_size=self.model_config.image_target_size,  # 【中文】裁剪后缩放到的目标分辨率
                random_rotation_angle=self.model_config.random_rotation_angle,  # 【中文】图像随机旋转角度上限（用于数据增强）
                color_jitter_params=self.model_config.color_jitter_params,  # 【中文】颜色抖动参数（亮度/对比度/饱和度/色调）
                model_name=self.model_config.model_name,  # 【中文】VLM backbone 名称（如 Eagle 模型名）
                model_type=self.model_config.backbone_model_type,  # 【中文】backbone 类型（如 "eagle"），影响特征对齐方式
                formalize_language=self.model_config.formalize_language,  # 【中文】是否对语言指令做“正规化”（如统一大小写/标点）
                apply_sincos_state_encoding=self.model_config.apply_sincos_state_encoding,  # 【中文】是否对连续状态加入sincos位置编码
                max_action_horizon=self.model_config.action_horizon,  # 【中文】最大动作时间跨度（预测多少步动作）
                use_albumentations=self.model_config.use_albumentations_transforms,  # 【中文】是否使用albumentations库做更强的数据增强
                shortest_image_edge=self.model_config.shortest_image_edge,  # 【中文】图像最短边缩放到的尺寸
                crop_fraction=self.model_config.crop_fraction,  # 【中文】中心裁剪比例
                transformers_loading_kwargs=self.transformers_loading_kwargs,  # 【中文】传递给 transformers 的通用加载参数
                use_alternate_vl_dit=self.model_config.use_alternate_vl_dit,  # 【中文】是否使用交替式 VL-DiT 结构（影响动作头结构）
                use_relative_action=self.model_config.use_relative_action,  # 【中文】是否使用相对动作表示（相对当前状态的增量动作）
                **self.transformers_loading_kwargs,
            )
        else:
            # 创建新的 Processor（从头训练时）
            processor = self.processor_class(
                modality_configs=self.config.data.modality_configs,  # 【中文】各具身形态的模态配置
                statistics=self._get_statistics(),  # By default is None, so this will be computed and set later.  # 【中文】预计算的数据统计信息（均值/方差等），默认None由数据集计算
                embodiment_id_mapping=self._get_embodiment_id_mapping(),  # By default is None, so this will be set later.  # 【中文】具身形态字符串 → 整数ID的映射，默认None稍后由数据集设置
                image_crop_size=self.model_config.image_crop_size,  # 【中文】图像裁剪尺寸
                image_target_size=self.model_config.image_target_size,  # 【中文】裁剪后目标分辨率
                random_rotation_angle=self.model_config.random_rotation_angle,  # 【中文】图像随机旋转角度
                color_jitter_params=self.model_config.color_jitter_params,  # 【中文】颜色抖动参数
                model_name=self.model_config.model_name,  # 【中文】VLM backbone 名称
                model_type=self.model_config.backbone_model_type,  # 【中文】backbone 类型（如 "eagle"）
                formalize_language=self.model_config.formalize_language,  # 【中文】语言正规化开关
                max_state_dim=self.model_config.max_state_dim,  # 【中文】状态向量最大维度（高出部分会被截断）
                max_action_dim=self.model_config.max_action_dim,  # 【中文】动作向量最大维度
                apply_sincos_state_encoding=self.model_config.apply_sincos_state_encoding,  # 【中文】是否对状态加sincos编码
                max_action_horizon=self.model_config.action_horizon,  # 【中文】最大动作时间跨度
                use_albumentations=self.model_config.use_albumentations_transforms,  # 【中文】是否使用albumentations增强
                shortest_image_edge=self.model_config.shortest_image_edge,  # 【中文】图像最短边缩放尺寸
                crop_fraction=self.model_config.crop_fraction,  # 【中文】裁剪比例
                use_relative_action=self.model_config.use_relative_action,  # 【中文】是否使用相对动作表示
                transformers_loading_kwargs=self.transformers_loading_kwargs,  # 【中文】transformers加载参数
            )

        # 打印并保存 Processor 配置（用于调试和复现）
        print(
            colored(
                f"These are all the processor configs for training: {json.dumps({k: str(v) for k, v in vars(processor).items()}, indent=2)}",
                "yellow",
            )
        )
        if get_rank() == 0:
            with open(self.save_cfg_dir / "final_processor_config.json", "w") as f:
                json.dump({k: str(v) for k, v in vars(processor).items()}, f, indent=2)

        self.processor = processor
        # 通过 DatasetFactory 构建数据集（支持多数据集混合、shard-based 加载）
        dataset_factory = DatasetFactory(config=self.config)
        train_dataset, eval_dataset = dataset_factory.build(processor=self.processor)

        # 保存数据集统计信息（均值、方差等，用于推理时反归一化）
        stats = train_dataset.get_dataset_statistics()
        stats_dict = convert_tensors_to_lists(stats)
        # Save statistics
        with open(save_cfg_dir / "dataset_statistics.json", "w") as f:
            json.dump(stats_dict, f, indent=2)
        logging.info("Saved dataset statistics for inference")

        return train_dataset, eval_dataset

    def _create_collator(self):
        """返回 data collator，用于将多个样本拼接成 batch。"""
        data_collator = self.processor.collator
        return data_collator


# 将 Gr00tN1d6Pipeline 注册到全局 MODEL_REGISTRY
register_model(Gr00tN1d6Config, Gr00tN1d6Pipeline)
