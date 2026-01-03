from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gr00t.data.interfaces import ShardedDataset
from gr00t.data.types import EmbodimentTag, MessageType, ModalityConfig, VLAStepData

from .lerobot_episode_loader import LeRobotEpisodeLoader



def extract_step_data(
    episode_data: pd.DataFrame,
    step_index: int,
    modality_configs: dict[str, ModalityConfig],
    embodiment_tag: EmbodimentTag,
    allow_padding: bool = False,
) -> VLAStepData:
    """
    从回合数据中提取单个时间步的多模态数据。
    
    该函数根据模态配置从 DataFrame 格式的回合数据中提取特定时间步的数据，
    支持多个模态（视频、状态、动作、语言等），并处理时间序列采样。
    
    参数：
        episode_data: 包含完整回合数据的 DataFrame，列名格式为 "modality.key"
        step_index: 要提取的目标时间步索引
        modality_configs: 每个模态的配置字典，指定采样的时间偏移和数据键
        embodiment_tag: 体现标识符，用于标记数据来源
        allow_padding: 是否允许索引填充。如果为 True，超出范围的索引会被限制到有效范围内
        
    返回：
        VLAStepData: 包含提取的多模态数据的结构化对象，包括图像、状态、动作和文本
        
    异常：
        KeyError: 当配置的模态键在回合数据中不存在时抛出
        AssertionError: 当语言数据键数量不为 1 时抛出
        
    示例：
        >>> config = {
        ...     "video": ModalityConfig(delta_indices=[0, -1], modality_keys=["front_cam"]),
        ...     "action": ModalityConfig(delta_indices=[0, 1, 2], modality_keys=["joint_vel"])
        ... }
        >>> step_data = extract_step_data(episode_df, 10, config, EmbodimentTag.FRANKA)
    """
    step_data = {}

    # 遍历每个配置的模态，提取对应的数据
    for modality, config in modality_configs.items():
        step_data[modality] = {}
        # 根据 delta_indices 配置计算需要采样的时间步索引
        # delta_indices 表示相对于当前 step_index 的时间偏移
        indices_to_load = [step_index + delta_index for delta_index in config.delta_indices]
        # 如果允许填充，将超出范围的索引限制到 [0, len-1] 范围内
        if allow_padding:
            indices_to_load = [max(0, min(idx, len(episode_data) - 1)) for idx in indices_to_load]
        # 遍历该模态配置的所有数据键
        for key in config.modality_keys:
            # 检查数据键是否存在于 DataFrame 中（列名格式为 "modality.key"）
            if f"{modality}.{key}" in episode_data.columns:
                # 根据索引列表提取对应的数据
                modality_data = episode_data[f"{modality}.{key}"].iloc[indices_to_load]
            else:
                # 如果键不存在，抛出错误并显示可用的列名
                raise KeyError(
                    f"{modality}.{key} not found in episode data, available keys: {episode_data.columns}"
                )
            # 对于数值型模态（状态、动作），将数据堆叠成数组格式
            if modality in ["state", "action"]:
                step_data[modality][key] = np.vstack(
                    [
                        np.array(modality_data.iloc[i]).astype(np.float32)
                        for i in range(len(modality_data))
                    ]
                )
            else:
                # 对于其他模态（视频、语言），保持列表格式
                step_data[modality][key] = modality_data.tolist()

    # 将提取的数据解析为 VLAStepData 结构
    video_data = step_data.get("video", {})
    state_data = step_data.get("state", {})
    action_data = step_data.get("action", {})
    language_data = step_data.get("language", {})
    # 验证语言数据只有一个键（通常是 "instruction" 或 "task"）
    assert len(language_data) == 1, f"Expected 1 language, got {len(language_data)}"
    # 提取文本指令（取第一个键的第一个元素）
    text = language_data[list(language_data.keys())[0]][0]

    # 构造并返回 VLAStepData 对象
    vla_step_data = VLAStepData(
        images=video_data,
        states=state_data,
        actions=action_data,
        text=text,
        embodiment=embodiment_tag,
    )
    return vla_step_data




class ShardedSingleStepDataset(ShardedDataset):
    """
    单步数据集，从各个回合中创建包含单个时间步的分片。

    该数据集实现通过以下方式为 VLA 训练提供步级数据访问：
    1. 使用 LeRobotEpisodeLoader 加载回合数据
    2. 将回合拆分为单个时间步
    3. 将时间步组织成平衡的分片以实现高效加载
    4. 支持回合子采样以提高数据效率

    分片策略确保分片大小平衡，同时在回合和回合内时间步之间保持随机化。
    每个分片包含来自不同回合的时间步混合，以提高训练多样性。

    主要特性：
    - 步级数据访问（相对于回合级）
    - 平衡分片以实现一致的批次大小
    - 通过采样率进行回合子采样
    - 与 LeRobot 数据格式集成
    - 支持多模态数据（视频、状态、动作、语言）

    参数：
        dataset_path: LeRobot 格式数据集目录路径
        embodiment_tag: 用于跨体现训练的体现标识符
        modality_configs: 每个模态的配置（采样、键）
        video_backend: 视频解码后端（'torchcodec'、'decord' 等）
        video_backend_kwargs: 视频后端的额外参数
        shard_size: 每个分片的目标时间步数
        episode_sampling_rate: 要使用的回合时间步比例（用于效率）
        seed: 用于可重现分片和采样的随机种子
        allow_padding: 是否允许将索引填充到有效范围 [0, max_length - 1]

    示例：
        >>> dataset = ShardedSingleStepDataset(
        ...     dataset_path="/path/to/lerobot_dataset",
        ...     embodiment_tag=EmbodimentTag.FRANKA,
        ...     modality_configs={
        ...         "video": ModalityConfig(delta_indices=[0], modality_keys=["front_cam"]),
        ...         "state": ModalityConfig(delta_indices=[0], modality_keys=["joint_positions"]),
        ...         "action": ModalityConfig(
        ...             delta_indices=list(range(8)), modality_keys=["joint_velocities"]
        ...         ),
        ...     },
        ...     shard_size=1024,
        ...     episode_sampling_rate=0.1,
        ... )
        >>> shard_data = dataset.get_shard(0)  # 获取处理过的时间步的第一个分片
    """

    def __init__(
        self,
        dataset_path: str | Path,
        embodiment_tag: EmbodimentTag,
        modality_configs: dict[str, ModalityConfig],
        video_backend: str = "torchcodec",
        video_backend_kwargs: dict[str, Any] | None = None,
        shard_size: int = 2**10,  # 1024 步
        episode_sampling_rate: float = 0.1,
        seed: int = 42,
        allow_padding: bool = False,
    ):
        """初始化具有分片配置的单步数据集。"""
        super().__init__(dataset_path)
        # 体现标识符
        self.embodiment_tag = embodiment_tag
        # 模态配置字典
        self.modality_configs = modality_configs
        # 视频后端
        self.video_backend = video_backend
        # 视频后端额外参数
        self.video_backend_kwargs = video_backend_kwargs
        # 分片大小
        self.shard_size = shard_size
        # 回合采样率
        self.episode_sampling_rate = episode_sampling_rate
        # 随机种子
        self.seed = seed
        # 是否允许填充
        self.allow_padding = allow_padding
        # 处理器（稍后设置）
        self.processor = None
        # 随机数生成器
        self.rng = np.random.default_rng(seed)
        # 计算动作范围
        action_delta_indices = modality_configs["action"].delta_indices
        self.action_horizon = max(action_delta_indices) - min(action_delta_indices) + 1

        # 创建回合加载器
        self.episode_loader = LeRobotEpisodeLoader(
            dataset_path=dataset_path,
            modality_configs=modality_configs,
            video_backend=video_backend,
            video_backend_kwargs=video_backend_kwargs,
        )

        # 从回合时间步创建平衡分片
        self.shard_dataset()

    def shard_dataset(self):
        """
        通过跨分片分配回合时间步来创建平衡的分片。

        分片过程：
        1. 打乱回合顺序以实现随机化
        2. 根据采样率将每个回合拆分为多个子序列
        3. 跨分片分配子序列以平衡分片大小
        4. 使用贪婪分配来最小化分片大小方差

        这种方法确保：
        - 平衡的分片大小，以实现一致的训练批次
        - 分片内的多样性（回合和时间步的混合）
        - 基于种子的可重现分片
        """
        # 打乱回合索引
        shuffled_episode_indices = self.rng.permutation(len(self.episode_loader.episode_lengths))
        # 计算分割数量
        num_splits = int(1 / self.episode_sampling_rate)

        # 验证至少有一个有效回合
        assert len(shuffled_episode_indices) > 0, (
            f"No valid trajectories found for dataset {self.dataset_path}"
        )

        # 计算总时间步数和所需分片数量
        total_steps = np.sum(
            [self.get_effective_episode_length(idx) for idx in shuffled_episode_indices]
        ).astype(int)
        num_shards = np.ceil(total_steps / self.shard_size).astype(int)

        # 初始化分片容器
        sharded_episodes = [[] for _ in range(num_shards)]
        shard_lengths = np.zeros(num_shards, dtype=int)

        # 跨分片分配回合子序列
        for ep_idx in shuffled_episode_indices:
            # 将回合时间步拆分为多个子序列
            step_indices = np.arange(0, self.get_effective_episode_length(ep_idx))
            self.rng.shuffle(step_indices)
            for i in range(num_splits):
                # 获取第 i 个子序列（间隔采样）
                split_step_indices = step_indices[i::num_splits]
                # 分配到当前长度最小的分片（贪婪平衡）
                shard_index = np.argmin(shard_lengths)
                sharded_episodes[shard_index].append((ep_idx, split_step_indices))
                shard_lengths[shard_index] += len(split_step_indices)

        # 验证分片创建
        assert all(shard_lengths[i] > 0 for i in range(num_shards)), (
            "All shards must have length greater than 0"
        )

        # 打印分片统计信息
        print(f"Generated {num_shards} shards for dataset {self.dataset_path}")
        print(
            f"Total steps: {total_steps}, average shard length: {total_steps / num_shards}, shard length std: {np.std(shard_lengths)}"
        )
        # 保存分片信息
        self.sharded_episodes = sharded_episodes
        self.shard_lengths = shard_lengths

    def get_effective_episode_length(self, episode_index: int) -> int:
        """
        获取考虑动作范围的有效回合长度。
        
        参数：
            episode_index: 回合索引
            
        返回：
            有效回合长度
        """
        original_length = self.episode_loader.get_episode_length(episode_index)
        return max(0, original_length - self.action_horizon + 1)

    def __len__(self):
        """返回数据集中的分片数量。"""
        return len(self.shard_lengths)

    def get_datapoint(self, episode_data: pd.DataFrame, step_index: int) -> dict:
        """
        从回合数据中提取并处理单个时间步。

        将原始回合数据转换为 VLAStepData 结构，并应用
        配置的处理器来创建模型就绪的输入。

        参数：
            episode_data: 来自 LeRobotEpisodeLoader 的完整回合 DataFrame
            step_index: 要提取的回合内时间步索引

        返回：
            准备好进行模型训练的处理后数据点

        异常：
            AssertionError: 如果在调用此方法之前未设置处理器
        """
        assert self.processor is not None, "Processor must be set before getting datapoints"
        # 提取步骤数据
        vla_step_data = extract_step_data(
            episode_data, step_index, self.modality_configs, self.embodiment_tag, self.allow_padding
        )
        # 应用处理器转换为模型输入
        messages = [{"type": MessageType.EPISODE_STEP.value, "content": vla_step_data}]
        return self.processor(messages)

    def get_shard_length(self, idx: int) -> int:
        """
        获取特定分片中的时间步数量。
        
        参数：
            idx: 分片索引
            
        返回：
            分片中的时间步数量
        """
        return self.shard_lengths[idx]

    def get_shard(self, idx: int) -> list:
        """
        加载并处理特定分片中的所有时间步。

        加载所需的回合并提取分配给该分片的所有时间步，
        对每个时间步应用配置的处理器。

        参数：
            idx: 要加载的分片索引

        返回：
            准备好进行模型训练的处理后时间步列表
        """
        # 获取该分片的所有回合和步骤索引
        episodes = self.sharded_episodes[idx]
        datapoints = []
        for ep_idx, step_indices in episodes:
            # 每个回合只加载一次数据
            episode_data = self.episode_loader[ep_idx]
            # 提取该回合的所有步骤
            for step_index in step_indices:
                datapoints.append(self.get_datapoint(episode_data, step_index))
        return datapoints

    def get_dataset_statistics(self) -> dict:
        """
        获取底层回合加载器的数据集统计信息。
        
        返回：
            数据集统计信息字典
        """
        return self.episode_loader.get_dataset_statistics()

    def get_initial_actions(self):
        """
        获取底层回合加载器的初始动作。
        
        返回：
            初始动作数据
        """
        return self.episode_loader.get_initial_actions()
