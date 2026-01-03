# 【中文】基础配置类，整合了模型、数据、训练三大配置模块
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import List, Optional

import yaml

from gr00t.data.types import ActionConfig, ActionFormat, ActionRepresentation, ActionType

from .data.data_config import DataConfig, SingleDatasetConfig
from .model import create_model_union_type
from .model.gr00t_n1d6 import Gr00tN1d6Config
from .training.training_config import TrainingConfig

# 【中文】创建模型配置的联合类型（支持不同模型架构）
ModelUnionType = create_model_union_type()


@dataclass
class Config:
    """Complete configuration.
    【中文】完整的训练配置类，包含模型、数据、训练三大模块的所有配置项。
    """

    load_config_path: Optional[str] = None  # 【中文】配置文件加载路径
    model: ModelUnionType = field(default_factory=lambda: Gr00tN1d6Config())  # 【中文】模型配置，默认为Gr00tN1d6
    data: DataConfig = field(default_factory=DataConfig)  # 【中文】数据配置
    training: TrainingConfig = field(default_factory=TrainingConfig)  # 【中文】训练配置

    def save(self, path: Path):
        """Save configuration to YAML file.
        【中文】将配置保存为YAML文件。
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            yaml.dump(self, f)

    def load(self, path: Path):
        """Load configuration from YAML file.
        【中文】从YAML文件加载配置。
        """
        data = yaml.load(path.read_text(), Loader=yaml.Loader)
        if isinstance(data, dict):  # for training
            self.load_dict(data)
        elif isinstance(data, self.__class__):
            self = data
        else:
            raise ValueError(f"Invalid config file: {path}")
        # config = cls(**config) # if yaml.dump(self.__dict__, ...) is used
        return self

    def load_dict(self, data: dict):
        """【中文】从字典加载配置，支持动态更新model、data、training三个模块。"""
        if "model" in data:
            self.model = self.model.__class__(**data["model"])
        if "data" in data:
            self.data = DataConfig(**data["data"])
            # Ensure nested datasets are converted to dataclass instances
            # 【中文】确保嵌套的数据集配置被转换为dataclass实例
            converted: List[SingleDatasetConfig] = []
            for ds in self.data.datasets:
                if isinstance(ds, dict):
                    converted.append(SingleDatasetConfig(**ds))
                else:
                    converted.append(ds)
            self.data.datasets = converted
        if "training" in data:
            self.training = TrainingConfig(**data["training"])
        return self

    @classmethod
    def from_pretrained(cls, path: Path) -> "Config":
        """Load configuration from YAML file.
        【中文】从预训练模型的YAML文件加载配置。
        """
        data = yaml.load(path.read_text(), Loader=yaml.Loader)
        return data

    def get_deepspeed_config(self) -> dict:
        """Generate DeepSpeed configuration.
        【中文】生成DeepSpeed配置，根据stage参数选择ZeRO-2或ZeRO-3优化策略。
        """
        stage = self.training.deepspeed_stage

        gr00t_dir = Path(__file__).parent.parent
        if stage == 2:
            config = json.load(open(gr00t_dir / "configs/deepspeed/zero2_config.json"))
        elif stage == 3:
            config = json.load(open(gr00t_dir / "configs/deepspeed/zero3_config.json"))
        else:
            raise ValueError(f"Invalid DeepSpeed stage: {stage}")

        return config

    def validate(self):
        """Validate configuration.
        【中文】验证配置的有效性，包括数据集路径、具身形态标签、混合比例、动作配置等。
        """
        # Check dataset path(s)
        # 【中文】检查数据集路径和具身形态标签
        embodiment_tags = set()
        for d_cfg in self.data.datasets:
            # (Disable missing data check because we now support caching PDX data sources.)
            # if not Path(d_cfg.dataset_path).exists():
            #     raise ValueError(f"Dataset path does not exist: {d_cfg.dataset_path}")
            if d_cfg.dataset_type == "physical_embodiment" and not d_cfg.embodiment_tag:
                raise ValueError(f"Embodiment tag is empty for dataset {d_cfg.dataset_path}")
            if d_cfg.embodiment_tag is not None:
                embodiment_tags.add(d_cfg.embodiment_tag)

        # 【中文】提取并保留实际使用的具身形态配置
        stripped_modality_configs = {}
        for embodiment_tag in embodiment_tags:
            stripped_modality_configs[embodiment_tag] = self.data.modality_configs[embodiment_tag]
        self.data.modality_configs = stripped_modality_configs

        # ensure mix ratios are valid
        # 【中文】确保混合比例有效（总和必须大于0）
        total_ratio = sum(d.mix_ratio for d in self.data.datasets)
        if total_ratio <= 0:
            raise ValueError("Sum of mix_ratio must be greater than zero")

        # Fill in default values for action configs
        # 【中文】为动作配置填充默认值（绝对表示、非末端执行器类型、默认格式）
        for embodiment_tag in self.data.modality_configs:
            # Fill in default values for action representation, type and format
            if self.data.modality_configs[embodiment_tag]["action"].action_configs is None:
                self.data.modality_configs[embodiment_tag]["action"].action_configs = [
                    ActionConfig(
                        rep=ActionRepresentation.ABSOLUTE,
                        type=ActionType.NON_EEF,
                        format=ActionFormat.DEFAULT,
                    )
                ] * len(self.data.modality_configs[embodiment_tag]["action"].modality_keys)

        # 【中文】如果是Gr00tN1d6模型，验证backbone类型
        if isinstance(self.model, Gr00tN1d6Config):
            import warnings

            if self.model.eagle_collator:
                warnings.warn(
                    'eagle_collator is deprecated. Please use backbone_model_type "eagle" in the future.',
                    DeprecationWarning,
                )
                self.model.backbone_model_type = "eagle"
            assert self.model.backbone_model_type in [
                "eagle",
            ], f"Invalid backbone model type: {self.model.backbone_model_type}"

        # Validate precision settings
        # 【中文】验证精度设置（不能同时使用fp16和bf16）
        if self.training.fp16 and self.training.bf16:
            raise ValueError("Cannot use both fp16 and bf16")


def get_default_config() -> Config:
    """Get default configuration.
    【中文】获取默认配置实例。
    """
    return Config()
