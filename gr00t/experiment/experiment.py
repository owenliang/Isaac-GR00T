#!/usr/bin/env python
import json
import logging
import os
from pathlib import Path
import warnings

from omegaconf import OmegaConf
import torch
import torch.distributed as dist
from transformers import TrainingArguments, set_seed
import wandb

from gr00t.configs.base_config import Config

# Use custom trainer that profiles data loading & forward times
from gr00t.experiment.trainer import Gr00tTrainer, ProfCallback
from gr00t.experiment.utils import BestMetricCheckpointCallback, CheckpointFormatCallback
from gr00t.model import MODEL_REGISTRY
from gr00t.utils.initial_actions import INITIAL_ACTIONS_FILENAME, save_initial_actions


def setup_logging(debug: bool = False):
    """配置日志系统，减少 transformers 和 datasets 库的冗余输出。"""
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
    )
    # Reduce verbosity of some libraries
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("datasets").setLevel(logging.WARNING)


def warn_configs(config: Config):
    """校验配置并发出弃用警告，避免常见配置错误。"""
    # updates to batch size
    assert config.training.global_batch_size % config.training.num_gpus == 0, (
        "global_batch_size must be divisible by num_gpus"
    )

    if config.data.video_backend != "torchcodec":
        warnings.warn(
            "video_backend is not torchcodec. Only torchcodec will be supported in the future."
        )

    if config.training.batch_size is not None:
        warnings.warn(
            "batch_size will be deprecated in the future, please use global_batch_size instead. For now, this will override global_batch_size."
        )

    if config.training.warmup_steps > 0:
        warnings.warn(
            "warmup_steps will be deprecated in the future, please use warmup_ratio instead. For now, this will override warmup_ratio."
        )

    if (
        hasattr(config.model, "backbone_trainable_params_fp32")
        and not config.model.backbone_trainable_params_fp32
    ):
        warnings.warn(
            "backbone_trainable_params_fp32 is not True. This will be deprecated in the future."
        )

    if (
        hasattr(config.model, "use_albumentations_transforms")
        and not config.model.use_albumentations_transforms
    ):
        warnings.warn(
            "use_albumentations_transforms is not True. This will be deprecated in the future."
        )

    if (
        hasattr(config.model, "image_crop_size")
        and hasattr(config.model, "image_target_size")
        and (config.model.image_crop_size is not None or config.model.image_target_size is not None)
    ):
        assert (
            config.model.image_crop_size is not None and config.model.image_target_size is not None
        ), "image_crop_size and image_target_size must be set together"
        warnings.warn(
            "image_crop_size and image_target_size will be deprecated in the future. Please use shortest_image_edge and crop_fraction instead."
        )
        if hasattr(config.model, "shortest_image_edge") and hasattr(config.model, "crop_fraction"):
            assert (
                config.model.shortest_image_edge is None and config.model.crop_fraction is None
            ), (
                "Do not set shortest_image_edge and crop_fraction together with image_crop_size and image_target_size"
            )

    if (
        hasattr(config.model, "shortest_image_edge")
        and hasattr(config.model, "crop_fraction")
        and (config.model.shortest_image_edge is not None or config.model.crop_fraction is not None)
    ):
        assert config.model.use_albumentations_transforms, (
            "use_albumentations_transforms must be True when shortest_image_edge and crop_fraction are set"
        )


def run(config: Config):
    """训练主流程：初始化分布式环境 → 构建模型/数据 → 启动 Trainer。"""
    warn_configs(config)

    # 初始化分布式训练环境（如果需要）
    # 支持 torchrun 和普通单机模式
    if dist.is_initialized():
        global_rank = dist.get_rank()
    elif "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1:
        # 多进程模式：使用 NCCL 进行分布式通信
        dist.init_process_group(backend="nccl")
        # only meaningful for torchrun, for ray it is always 0
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        global_rank = dist.get_rank()
    else:
        local_rank = 0
        global_rank = 0

    # 基础设置：日志、随机种子、配置校验
    setup_logging()
    set_seed(config.data.seed)

    # 校验配置（embodiment tag、mix_ratio、action config 等）
    config.validate()

    # 创建输出目录（保存 checkpoints、日志等）
    if config.training.experiment_name is None:
        output_dir = Path(config.training.output_dir)
        experiment_name = output_dir.name
    else:
        output_dir = Path(config.training.output_dir) / config.training.experiment_name
        experiment_name = config.training.experiment_name

    output_dir.mkdir(parents=True, exist_ok=True)

    # 保存配置文件（用于复现和推理）
    save_cfg_dir = output_dir / "experiment_cfg"
    processor_dir = output_dir / "processor"
    config.save(save_cfg_dir / "config.yaml")
    omegaconf_config = OmegaConf.create(config.__dict__)
    omegaconf_config["max_steps"] = config.training.max_steps
    omegaconf_config["save_steps"] = config.training.save_steps
    OmegaConf.save(omegaconf_config, save_cfg_dir / "conf.yaml", resolve=True)
    wandb_config_file = output_dir / "wandb_config.json"
    with open(wandb_config_file, "w") as f:
        json.dump(
            {
                "project": config.training.wandb_project,
                "run_id": experiment_name,
            },
            f,
        )

    logging.info(f"Saved config to {save_cfg_dir}")

    # 初始化 WandB（仅主进程）
    if config.training.use_wandb and global_rank == 0:
        # Add git commit hash and version info to config
        config_dict = {
            **config.__dict__,
            "git_commit_hash": os.environ.get("GROOT_COMMIT_HASH", "unknown"),
        }

        wandb.init(
            project=config.training.wandb_project,
            name=experiment_name,
            config=config_dict,
            tags=[config.data.mode],
        )

    # 核心：通过 MODEL_REGISTRY 获取模型 Pipeline
    # Pipeline 负责构建模型、数据集、Processor、Collator
    # 【中文】注册表模式：
    # 1. type(config.model) 获取配置类型（如 Gr00tN1d6Config）
    # 2. MODEL_REGISTRY.get(...) 查找对应的 Pipeline 类（如 Gr00tN1d6Pipeline）
    # 3. (config, save_cfg_dir) 实例化 Pipeline 对象
    pipeline = MODEL_REGISTRY.get(type(config.model))(config, save_cfg_dir) # 由gr00t/model/__init__.py初始化注册了gr00t模型
    pipeline.setup()  # 初始化所有组件，主要用途是创建模型、创建训练数据集、dataloader数据处理方法
    model = pipeline.return_model() # 获取模型
    train_dataset, eval_dataset = pipeline.return_dataset() # 获取训练/评估数据集
    data_collator = pipeline.return_collator() # 获取数据_collator
    processor = pipeline.return_processor()
    processor.save_pretrained(processor_dir)

    # DeepSpeed 配置（多 GPU 且不使用 DDP 时）
    if config.training.num_gpus > 1 and not config.training.use_ddp:
        deepspeed_config = config.get_deepspeed_config()
    else:
        deepspeed_config = None

    # 计算每个设备的 batch size
    # for now we will let batch_size override global_batch_size, in future we will deprecate batch_size
    if config.training.batch_size is None:
        per_device_train_batch_size = config.training.global_batch_size // config.training.num_gpus
    else:
        per_device_train_batch_size = config.training.batch_size

    # 创建 HuggingFace TrainingArguments（封装所有训练超参数）
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        max_steps=config.training.max_steps,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=config.training.eval_batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        learning_rate=config.training.learning_rate,
        lr_scheduler_type=config.training.lr_scheduler_type,
        weight_decay=config.training.weight_decay,
        warmup_ratio=config.training.warmup_ratio,
        max_grad_norm=config.training.max_grad_norm,
        logging_steps=config.training.logging_steps,
        save_steps=config.training.save_steps,
        save_total_limit=config.training.save_total_limit,
        fp16=config.training.fp16,
        bf16=config.training.bf16,
        tf32=config.training.tf32,
        gradient_checkpointing=config.training.gradient_checkpointing,
        optim=config.training.optim,
        dataloader_num_workers=config.training.dataloader_num_workers,
        report_to="wandb" if config.training.use_wandb else "none",
        seed=config.data.seed,
        deepspeed=deepspeed_config,
        ddp_find_unused_parameters=False,
        ddp_bucket_cap_mb=config.training.ddp_bucket_cap_mb,
        eval_strategy=config.training.eval_strategy,
        eval_steps=config.training.eval_steps,
        batch_eval_metrics=True,
        remove_unused_columns=config.training.remove_unused_columns,
        ignore_data_skip=True,
    )

    # 创建自定义 Trainer（支持 profiling、自定义 dataloader）
    # 【中文】multiprocessing_context 指定多进程启动方式：
    # - "fork": 复制父进程内存（Linux默认，快速但可能有CUDA问题）
    # - "spawn": 启动全新进程（Windows默认，安全但慢）
    # - "forkserver": 使用服务器进程fork（平衡性能和安全性）
    trainer = Gr00tTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        multiprocessing_context=config.data.multiprocessing_context,
    )

    # 添加 checkpoint 格式化回调（保存时自动整理配置文件）
    trainer.add_callback(
        CheckpointFormatCallback(
            run_name=experiment_name,
            exp_cfg_dir=save_cfg_dir,
            processor_dir=processor_dir,
        )
    )

    # 可选：保存最佳评估指标的 checkpoint
    if config.training.save_best_eval_metric_name != "":
        trainer.add_callback(
            BestMetricCheckpointCallback(
                metric_name=config.training.save_best_eval_metric_name,
                greater_is_better=config.training.save_best_eval_metric_greater_is_better,
                exp_cfg_dir=save_cfg_dir,
            )
        )

    # 如果数据集支持，保存初始动作（用于某些任务的初始化）
    if hasattr(train_dataset, "get_initial_actions"):
        initial_actions = train_dataset.get_initial_actions()
        if initial_actions:
            initial_actions_path = save_cfg_dir / INITIAL_ACTIONS_FILENAME
            save_initial_actions(initial_actions, initial_actions_path)
            logging.info(f"Saved {len(initial_actions)} initial actions to {initial_actions_path}")

    # 开始训练
    logging.info("🚀 Starting training...")
    if config.training.enable_profiling:
        # 性能分析模式：使用 torch.profiler 记录 CPU/CUDA 执行轨迹
        from functools import partial

        logging.info(f"{global_rank} Starting training with profiling...")

        def on_trace_ready_handler(trainer, profile_dir, prof):
            output_path = (
                profile_dir / f"trace_rank_{global_rank}_iter_{trainer.state.global_step}.json"
            )
            prof.export_chrome_trace(str(output_path))
            logging.info(f"Trace saved to {output_path}")

        profile_dir = output_dir / "profiling"
        profile_dir.mkdir(parents=True, exist_ok=True)

        # 【中文】torch.profiler.profile 是一个上下文管理器，自动管理 profiler 的生命周期
        # 【中文】schedule 参数定义了性能分析的阶段：
        # - skip_first=10: 跳过前10个step（避免启动开销干扰）
        # - wait=1: 等待1个step
        # - warmup=1: 预热1个step
        # - active=3: 活跃记录3个step
        # - repeat=1: 重复1次周期
        # 【中文】ProfCallback.on_step_end() 会调用 prof.step()，通知 profiler 进入下一个阶段
        # 【中文】profiler 根据 schedule 自动切换状态（wait → warmup → active → wait...）
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(skip_first=10, wait=1, warmup=1, active=3, repeat=1),
            # profile_memory=True,
            with_stack=True,
            # record_shapes=True,
            on_trace_ready=partial(on_trace_ready_handler, trainer, profile_dir),
        ) as prof:
            trainer.add_callback(ProfCallback(prof=prof))
            trainer.train(resume_from_checkpoint=True)
    else:
        # 正常训练模式
        # 【中文】resume_from_checkpoint=True 的处理逻辑：
        # 1. Gr00tTrainer.train() 接收到 True
        # 2. 调用 get_last_checkpoint(self.args.output_dir) 查找最新的 checkpoint
        # 3. output_dir 来自 TrainingArguments(output_dir=str(output_dir))
        # 4. output_dir 来自 experiment.py 的 config.training.output_dir
        # 【中文】查找逻辑：在 output_dir 中找到所有以 'checkpoint-' 开头的目录，返回最新的一个
        # 【中文】示例：output_dir/checkpoint-1000, checkpoint-2000 → 返回 checkpoint-2000
        trainer.train(resume_from_checkpoint=True)

    # 保存最终模型
    trainer.save_model()
    logging.info(f"Model saved to {output_dir}")

    # 可选：断言最终 loss 小于某个阈值（用于测试）
    if config.training.assert_loss_less_than is not None:
        final_loss = trainer.loss
        if final_loss.item() > config.training.assert_loss_less_than:
            raise AssertionError(
                f"Loss too high: {final_loss.item()} vs {config.training.assert_loss_less_than})"
            )

    # 清理资源
    if hasattr(train_dataset, "close"):
        train_dataset.close()
    if eval_dataset is not None and hasattr(eval_dataset, "close"):
        eval_dataset.close()
    logging.info("Training completed!")
