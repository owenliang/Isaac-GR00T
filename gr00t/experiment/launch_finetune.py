# Launch finetuning for N1.6 on "single node".
# This script tries to provide a similar user experience as current OSS.
# 【中文】这是单节点微调的入口脚本，负责将 FinetuneConfig 转换为完整的训练配置并启动训练流程

import os
from pathlib import Path

import tyro

from gr00t.configs.base_config import get_default_config
from gr00t.configs.finetune_config import FinetuneConfig
from gr00t.experiment.experiment import run


# Make sure the user provided modality config is registered.
# 【中文】动态加载用户提供的模态配置文件（如果有的话），通过导入 Python 模块的方式注册自定义的 modality config
def load_modality_config(modality_config_path: str):
    import importlib
    import sys

    path = Path(modality_config_path)
    if path.exists() and path.suffix == ".py":
        sys.path.append(str(path.parent))
        importlib.import_module(path.stem)
        print(f"Loaded modality config: {path}")
    else:
        raise FileNotFoundError(f"Modality config path does not exist: {modality_config_path}")


if __name__ == "__main__":
    # Set LOGURU_LEVEL environment variable if not already set (default: INFO)
    # 【中文】设置日志级别
    if "LOGURU_LEVEL" not in os.environ:
        os.environ["LOGURU_LEVEL"] = "INFO"
    
    # Use tyro for clean CLI
    # 【中文】使用 tyro 解析命令行参数，生成 FinetuneConfig 对象
    ft_config = tyro.cli(FinetuneConfig, description=__doc__)
    embodiment_tag = ft_config.embodiment_tag.value

    # all rank workers should register for the modality config
    # 【中文】所有进程都需要注册 modality config（分布式训练时保持一致）
    if ft_config.modality_config_path is not None:
        load_modality_config(ft_config.modality_config_path)

    # 【中文】获取默认配置并注入数据集信息
    config = get_default_config().load_dict(
        {
            "data": {
                "download_cache": False,
                "datasets": [
                    {
                        "dataset_paths": [ft_config.dataset_path],
                        "mix_ratio": 1.0,
                        "embodiment_tag": embodiment_tag,
                    }
                ],
            }
        }
    )
    config.load_config_path = None

    # overwrite with finetune config supplied by the user
    # 【中文】用用户提供的 finetune config 覆盖默认配置，包括：模型微调开关、数据增强参数
    config.model.tune_llm = ft_config.tune_llm
    config.model.tune_visual = ft_config.tune_visual
    config.model.tune_projector = ft_config.tune_projector
    config.model.tune_diffusion_model = ft_config.tune_diffusion_model
    config.model.state_dropout_prob = ft_config.state_dropout_prob
    config.model.random_rotation_angle = ft_config.random_rotation_angle
    config.model.color_jitter_params = ft_config.color_jitter_params

    # 【中文】N1.6 模型的固定配置，使用 Eagle-Block2A-2B-v2 作为 VLM backbone
    config.model.load_bf16 = False
    config.model.reproject_vision = False
    config.model.eagle_collator = True
    config.model.model_name = "nvidia/Eagle-Block2A-2B-v2"
    config.model.backbone_trainable_params_fp32 = True  # 【中文】backbone 可训练参数使用 FP32 精度
    config.model.use_relative_action = True  # 【中文】使用相对动作表示（关键设计）

    # 【中文】训练配置：从 base model 开始微调
    config.training.start_from_checkpoint = ft_config.base_model_path
    config.training.optim = "adamw_torch"
    config.training.global_batch_size = ft_config.global_batch_size
    config.training.dataloader_num_workers = ft_config.dataloader_num_workers
    config.training.learning_rate = ft_config.learning_rate
    config.training.gradient_accumulation_steps = ft_config.gradient_accumulation_steps
    config.training.output_dir = ft_config.output_dir
    config.training.save_steps = ft_config.save_steps
    config.training.save_total_limit = ft_config.save_total_limit
    config.training.num_gpus = ft_config.num_gpus
    config.training.use_wandb = ft_config.use_wandb
    config.training.max_steps = ft_config.max_steps
    config.training.weight_decay = ft_config.weight_decay
    config.training.warmup_ratio = ft_config.warmup_ratio
    config.training.wandb_project = "finetune-gr00t-n1d6"

    # 【中文】数据加载配置：基于 shard 的高效数据流
    config.data.shard_size = ft_config.shard_size
    config.data.episode_sampling_rate = ft_config.episode_sampling_rate
    config.data.num_shards_per_epoch = ft_config.num_shards_per_epoch

    # 【中文】启动训练主流程
    run(config)
