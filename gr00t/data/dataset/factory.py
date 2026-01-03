# DatasetFactory: 数据集构建工厂类
# 负责创建 shard-based 的混合数据集，支持多数据集按比例混合

import numpy as np
import torch
from tqdm import tqdm

from gr00t.configs.base_config import Config
from gr00t.data.dataset.sharded_mixture_dataset import ShardedMixtureDataset
from gr00t.data.dataset.sharded_single_step_dataset import ShardedSingleStepDataset
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.interfaces import BaseProcessor
from gr00t.data.stats import generate_rel_stats, generate_stats


class DatasetFactory:
    """
    数据集构建工厂。与模型无关。
    关键特性：
    - 支持多数据集按 mix_ratio 混合
    - 使用 shard-based 加载，内存高效
    - 自动生成统计信息（绝对值和相对值）
    """

    def __init__(self, config: Config):
        self.config = config

    def build(
        self, processor: BaseProcessor
    ) -> tuple[ShardedMixtureDataset, ShardedMixtureDataset | None]:
        """
        构建训练数据集。
        返回：(train_dataset, eval_dataset)
        注意：Shard-based 数据集不支持评估集，eval_dataset 总是 None
        """
        assert self.config.training.eval_strategy == "no", (
            "Sharded dataset does not support evaluation sets"
        )

        all_datasets = []
        all_weights = []
        # 遍历所有数据集配置，构建单个数据集
        for dataset_spec in tqdm(
            self.config.data.datasets,
            total=len(self.config.data.datasets),
            desc="Initializing datasets",
        ):
            datasets = []
            for dataset_path in dataset_spec.dataset_paths:
                embodiment_tag = dataset_spec.embodiment_tag
                assert embodiment_tag is not None, "Embodiment tag is required"
                assert self.config.data.mode == "single_turn", "Only single turn mode is supported"
                # 生成统计信息（仅主进程，分布式时同步）
                if torch.distributed.is_initialized():
                    if torch.distributed.get_rank() == 0:
                        generate_stats(dataset_path)
                        generate_rel_stats(dataset_path, EmbodimentTag(embodiment_tag))
                else:
                    generate_stats(dataset_path)
                    generate_rel_stats(dataset_path, EmbodimentTag(embodiment_tag))
                torch.distributed.barrier()  # 等待所有进程生成完成
                # 创建 shard-based 单步数据集
                dataset = ShardedSingleStepDataset(
                    dataset_path=dataset_path,
                    embodiment_tag=EmbodimentTag(embodiment_tag),
                    modality_configs=self.config.data.modality_configs[embodiment_tag],
                    video_backend=self.config.data.video_backend,
                    shard_size=self.config.data.shard_size,
                    episode_sampling_rate=self.config.data.episode_sampling_rate,
                    seed=self.config.data.seed,
                    allow_padding=self.config.data.allow_padding,
                )
                datasets.append(dataset)
            # 根据数据集长度和 mix_ratio 计算权重
            dataset_lengths = np.array([len(dataset) for dataset in datasets])
            dataset_relative_lengths = dataset_lengths / dataset_lengths.sum()
            for dataset, relative_length in zip(datasets, dataset_relative_lengths):
                weight = relative_length * dataset_spec.mix_ratio
                all_datasets.append(dataset)
                all_weights.append(weight)

        # 创建混合数据集（按权重采样）
        return (
            ShardedMixtureDataset(
                datasets=all_datasets,
                weights=all_weights,
                processor=processor,
                seed=self.config.data.seed,
                training=True,
                num_shards_per_epoch=self.config.data.num_shards_per_epoch,
                override_pretraining_statistics=self.config.data.override_pretraining_statistics,
            ),
            None,
        )
