#!/usr/bin/env python
"""
LeRobot Dataset Loader

A simplified, clean implementation for loading LeRobot datasets with video support.
This module provides the core functionality for loading episodes from LeRobot format datasets,
handling metadata parsing, video decoding, and data preprocessing for VLA training.

The LeRobotEpisodeLoader serves as the foundation for higher-level dataset classes,
providing episode-level data access with support for multi-modal data including:
- Video frames from multiple camera views
- Proprioceptive state information
- Action sequences
- Language instructions/annotations

Returns messages with VLAStepData as defined in types.py.

【中文】LeRobot 数据集加载器，提供从 LeRobot 格式数据集中按 episode 加载多模态数据的核心能力。
【中文】负责：解析元数据（info/episodes/tasks/modality/stats）、解码多路相机视频、按配置切出 state/action 关节组，
【中文】并以与 VLA 训练流水线兼容的形式（DataFrame + VLAStepData）暴露给上层数据集与 Processor 使用。
"""

from collections import defaultdict
import json
from pathlib import Path
import random
from typing import Any

import numpy as np
import pandas as pd

from gr00t.data.types import ModalityConfig
from gr00t.utils.initial_actions import INITIAL_ACTIONS_FILENAME, load_initial_actions
from gr00t.utils.video_utils import get_frames_by_indices


# LeRobot standard metadata filenames
LEROBOT_META_DIR_NAME = "meta"
LEROBOT_INFO_FILENAME = "info.json"
LEROBOT_EPISODES_FILENAME = "episodes.jsonl"
LEROBOT_TASKS_FILENAME = "tasks.jsonl"
LEROBOT_MODALITY_FILENAME = "modality.json"
LEROBOT_STATS_FILE_NAME = "stats.json"
LEROBOT_RELATIVE_STATS_FILE_NAME = "relative_stats.json"

ALLOWED_MODALITIES = ["video", "state", "action", "language"]
DEFAULT_COLUMN_NAMES = {
    "state": "observation.state",
    "action": "action",
}

LANG_KEYS = ["task", "sub_task"]


def _rec_defaultdict() -> defaultdict:
    """Factory that creates an infinitely nestable defaultdict."""
    return defaultdict(_rec_defaultdict)


def _to_plain_dict(tree):
    """Recursively turn a (nested) defaultdict into a regular dict."""
    if isinstance(tree, defaultdict):
        return {k: _to_plain_dict(v) for k, v in tree.items()}
    return tree


class LeRobotEpisodeLoader:
    """
    Episode-level data loader for LeRobot format datasets.

    This class handles the loading and preprocessing of individual episodes from LeRobot datasets.
    It manages metadata parsing, video decoding, and data extraction across multiple modalities
    (video, state, action, language) while maintaining compatibility with the VLA training pipeline.

    Key responsibilities:
    - Parse LeRobot metadata files (info.json, episodes.jsonl, etc.)
    - Load and decode video data using configurable backends
    - Extract and process multi-modal data according to modality configurations
    - Provide dataset statistics for normalization
    - Handle initial action loading for policy initialization

    Args:
        dataset_path: Path to dataset root directory containing meta/ and data files
        modality_configs: Dictionary mapping modality names to ModalityConfig objects
                         that specify temporal sampling and data keys to load
        video_backend: Video decoding backend ('torchcodec', 'decord', etc.)
        video_backend_kwargs: Additional arguments for the video backend

    Example:
        >>> loader = LeRobotEpisodeLoader(
        ...     dataset_path="/path/to/lerobot_dataset",
        ...     modality_configs={
        ...         "video": ModalityConfig(delta_indices=[0], modality_keys=["front_cam"]),
        ...         "state": ModalityConfig(delta_indices=[0], modality_keys=["joint_positions"]),
        ...         "action": ModalityConfig(
        ...             delta_indices=list(range(16)), modality_keys=["joint_velocities"]
        ...         ),
        ...     },
        ... )
        >>> episode_data = loader[0]  # Load first episode as DataFrame

    【中文】面向 LeRobot 格式数据集的“episode 级”加载器，是上层所有数据集类的基础。
    【中文】通过解析 meta 目录中的 info/episodes/tasks/modality/stats 等文件，
    【中文】再按给定的 ModalityConfig 从 parquet + 视频文件中抽取出 video/state/action/language，
    【中文】最终返回一个按时间步排列的 DataFrame，每列对应一个模态下的关节组或视角（如 state.left_arm, action.left_arm, video.ego_view）。
    """

    def __init__(
        self,
        dataset_path: str | Path,
        modality_configs: dict[str, ModalityConfig],
        video_backend: str = "torchcodec",
        video_backend_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """
        Initialize LeRobot episode loader with dataset path and modality configurations.

        The initialization process involves:
        1. Loading all metadata files from the dataset
        2. Parsing and validating modality configurations
        3. Computing effective episode lengths based on action horizon

        【中文】初始化 episode 级加载器：
        【中文】1）加载并解析 meta 目录中的所有元数据文件（info/episodes/tasks/modality/stats 等）；
        【中文】2）根据传入的 modality_configs 校验/补全各模态配置；
        【中文】3）预先计算每个 episode 的长度，为后续分片/采样提供基础信息。
        """
        self.dataset_path = Path(dataset_path)
        self.video_backend = video_backend
        self.video_backend_kwargs = video_backend_kwargs

        if not self.dataset_path.is_dir():
            raise FileNotFoundError(f"Dataset path does not exist: {self.dataset_path}")

        # Load metadata files and parse dataset structure
        self._load_metadata()

        # Set up modality configs after metadata is loaded
        self.modality_configs = self._parse_and_validate_modality_configs(modality_configs)

        # Compute effective episode lengths accounting for action horizon
        self.episode_lengths = self.get_episode_lengths()

    def _load_metadata(self) -> None:
        """
        Load all metadata files including dataset statistics.

        Parses the standard LeRobot metadata structure:
        - info.json: Dataset configuration and file patterns
        - episodes.jsonl: Per-episode metadata (length, timestamps, etc.)
        - tasks.jsonl: Task descriptions and mappings
        - modality.json: Modality structure and data layout
        - stats.json: Dataset statistics for normalization

        【中文】从 meta 目录加载并解析 LeRobot 标准元数据：
        【中文】- info.json：数据集整体配置（路径模板、chunk 大小、fps 等）；
        【中文】- episodes.jsonl：每条 episode 的长度、索引等信息；
        【中文】- tasks.jsonl：任务 id → 文本描述的映射；
        【中文】- modality.json：各模态键在原始数组中的切片范围（start/end）；
        【中文】- stats.json / relative_stats.json：数值模态的统计量（用于归一化），并挂到 self.stats 上。
        """
        meta_dir = self.dataset_path / LEROBOT_META_DIR_NAME

        # Load dataset configuration
        info_path = meta_dir / LEROBOT_INFO_FILENAME
        with open(info_path, "r") as f:
            self.info_meta = json.load(f)

        # Load episode metadata (one episode per line)
        episodes_path = meta_dir / LEROBOT_EPISODES_FILENAME
        with open(episodes_path, "r") as f:
            self.episodes_metadata = [json.loads(line) for line in f]

        # Load task descriptions and create mapping
        tasks_path = meta_dir / LEROBOT_TASKS_FILENAME
        with open(tasks_path, "r") as f:
            tasks_data = [json.loads(line) for line in f]
            self.tasks_map = {task["task_index"]: task["task"] for task in tasks_data}

        # Load modality structure information
        modality_path = meta_dir / LEROBOT_MODALITY_FILENAME
        with open(modality_path, "r") as f:
            self.modality_meta = json.load(f)

        # Load dataset statistics for normalization
        stats_path = meta_dir / LEROBOT_STATS_FILE_NAME
        assert stats_path.exists(), (
            f"{stats_path} does not exist for {self.dataset_path}, please use gr00t/data/stats.py to generate it"
        )
        with open(stats_path, "r") as f:
            self.stats = json.load(f)

        relative_stats_path = meta_dir / LEROBOT_RELATIVE_STATS_FILE_NAME
        if relative_stats_path.exists():
            with open(relative_stats_path, "r") as f:
                self.stats["relative_action"] = json.load(f)

        # Extract key configuration parameters
        self.feature_config = self.info_meta.get("features", {})
        self.data_path_pattern = self.info_meta["data_path"]
        self.video_path_pattern = self.info_meta.get("video_path")
        self.chunk_size = self.info_meta["chunks_size"]
        self.fps = self.info_meta.get("fps", 30)

    def get_episode_lengths(self):
        """
        Compute original episode lengths.

        Returns:
            List of original episode lengths
        """
        episode_lengths = []
        for ep_meta in self.episodes_metadata:
            episode_lengths.append(int(ep_meta["length"]))
        return episode_lengths

    def get_episode_length(self, idx: int) -> int:
        """Get the length of a specific episode."""
        return self.episode_lengths[idx]

    def _parse_and_validate_modality_configs(
        self,
        modality_configs: dict[str, ModalityConfig],
    ) -> dict[str, ModalityConfig]:
        """
        Parse and validate modality configurations, filling in defaults where needed.

        For missing modality configs, creates default configurations:
        - video: All available camera views with single timestep
        - state: All available state keys with single timestep
        - action: All available action keys with 16-step horizon
        - language: Must be explicitly configured if needed

        Args:
            modality_configs: User-provided modality configurations

        Returns:
            Complete and validated modality configurations

        Raises:
            ValueError: If invalid modalities are specified
            AssertionError: If language modality configuration is invalid

        【中文】校验并标准化传入的各模态配置：
        【中文】- 检查模态名是否合法（限制在 video/state/action/language）；
        【中文】- 对 language 模态施加额外约束：必须只有一个 key，且只能采样单步（delta_indices=[0]）。
        【中文】这里不主动生成默认配置，只是做合法性检查并返回整理后的配置。
        """
        # Validate all modality configurations
        for modality in modality_configs:
            if modality not in ALLOWED_MODALITIES:
                raise ValueError(f"Invalid modality: {modality}")
            if modality == "language":
                # Language modality has special constraints
                assert len(modality_configs[modality].modality_keys) == 1, (
                    "Language modality must have exactly one key"
                )
                assert modality_configs[modality].delta_indices == [0], (
                    "Only single timestep is supported for language modality"
                )

        return modality_configs

    def __len__(self) -> int:
        """Return number of episodes in dataset."""
        return len(self.episodes_metadata)

    def _extract_joint_groups(
        self,
        df: pd.DataFrame,
        joint_groups: list[str],
        modality_type: str = "state",
    ) -> pd.DataFrame:
        """
        Extract specific joint groups from data arrays based on modality metadata.

        Uses the modality metadata to slice the appropriate indices from the raw data arrays,
        allowing for flexible joint group extraction (e.g., arm joints, gripper state).

        Args:
            df: DataFrame containing the raw episode data
            joint_groups: List of joint group names to extract (e.g., ["arm", "gripper"])
            modality_type: Type of modality ("state" or "action")

        Returns:
            DataFrame with columns for each requested joint group containing sliced arrays

        【中文】根据 modality.json 中记录的 start/end 信息，从原始数组列中切出指定的关节组。
        【中文】例如：从 observation.state 中切出 left_arm/right_arm 等子向量，并以列名为 group_name 写入新的 DataFrame。
        """
        modality_info = self.modality_meta.get(modality_type, {})
        joint_data = pd.DataFrame()

        for group_name in joint_groups:
            if group_name in modality_info:
                group_info = modality_info[group_name]
                start_idx = group_info["start"]
                end_idx = group_info["end"]
                original_key = group_info.get("original_key", DEFAULT_COLUMN_NAMES[modality_type])
                # Slice the array data for this joint group
                if isinstance(df[original_key].iloc[0], np.ndarray):
                    joint_data[group_name] = df[original_key].map(lambda x: x[start_idx:end_idx])
                else:
                    joint_data[group_name] = df[original_key]  # for strings and scalars
            else:
                print(
                    f"Warning: Joint group '{group_name}' not found in {modality_type} modality. Available groups: {list(modality_info.keys())}"
                )

        return joint_data

    def _load_parquet_data(self, episode_index: int) -> pd.DataFrame:
        """
        Load and process parquet data for a specific episode.

        Handles the complete data loading pipeline:
        1. Load raw parquet file based on chunking structure
        2. Process language annotations (convert task indices to strings)
        3. Extract state and action joint groups

        Args:
            episode_index: Index of the episode to load

        Returns:
            Processed DataFrame with all modality data

        【中文】加载单个 episode 对应的 parquet 文件，并完成数值/语言模态的预处理：
        【中文】1）按 chunk_size 计算所在 parquet 文件并读取原始 DataFrame；
        【中文】2）若配置了 language，则将任务 id 映射为可读文本，生成 language.xxx 列；
        【中文】3）调用 _extract_joint_groups 将 state/action 的大数组拆成多个关节组列，例如 state.left_arm、action.left_arm。
        """
        # Load raw parquet data using chunking pattern
        chunk_idx = episode_index // self.chunk_size
        parquet_filename = self.data_path_pattern.format(
            episode_chunk=chunk_idx, episode_index=episode_index
        )
        parquet_path = self.dataset_path / parquet_filename
        original_df = pd.read_parquet(parquet_path)
        loaded_df = pd.DataFrame()


        # 处理语言标注（将任务索引转换为任务字符串）
        # Process language annotations (convert task indices to task strings)
        if "language" in self.modality_configs:
            for key in self.modality_configs["language"].modality_keys:
                # 这些键将从 episodes.jsonl 中单独加载
                # these keys will be loaded separately from episodes.jsonl
                if key in LANG_KEYS:
                    continue
                # 确保键以 "annotation." 开头
                assert key.startswith("annotation.")
                # 提取子键（去除 "annotation." 前缀）
                subkey = key.replace("annotation.", "")
                # 验证子键存在于模态元数据中
                assert subkey in self.modality_meta["annotation"], (
                    f"Key {subkey} not found in language modality"
                )
                # 获取原始键名，如果未指定则使用当前键
                original_key = self.modality_meta["annotation"][subkey].get("original_key", key)
                # 将任务索引映射为对应的任务描述文本
                loaded_df[f"language.{key}"] = original_df[original_key].apply(
                    lambda x: self.tasks_map[x]
                )


        # 针对 state 和 action 模态，提取对应的关节组数据
        # Extract joint groups for state and action modalities
        for modality_type in ["state", "action"]:
            # 如果当前模态类型未在配置中，跳过处理
            # Skip if current modality type is not configured
            if modality_type not in self.modality_configs:
                continue
            # 调用 _extract_joint_groups 方法，根据配置的关节组键从原始 DataFrame 中提取数据
            # Call _extract_joint_groups to extract data from original DataFrame based on configured joint group keys
            joint_groups_df = self._extract_joint_groups(
                original_df, self.modality_configs[modality_type].modality_keys, modality_type
            )
            # 遍历提取出的关节组，将每个关节组数据作为新列添加到 loaded_df 中
            # 列名格式为: {modality_type}.{joint_group}，例如 "state.left_arm" 或 "action.gripper"
            # Iterate through extracted joint groups, add each as a new column to loaded_df
            # Column naming format: {modality_type}.{joint_group}, e.g., "state.left_arm" or "action.gripper"
            for joint_group in joint_groups_df.columns:
                loaded_df[f"{modality_type}.{joint_group}"] = joint_groups_df[joint_group]

        return loaded_df


    def _load_video_data(self, episode_index: int, indices: np.ndarray) -> dict[str, np.ndarray]:
        """
        Load video data for all configured camera views at specified indices.

        Uses the configured video backend to decode video frames at the exact indices
        needed for the episode, supporting multiple camera views simultaneously.

        Args:
            episode_index: Index of the episode to load videos for
            indices: Array of indices to extract frames at

        Returns:
            Dictionary mapping camera view names to arrays of decoded frames

        【中文】按给定的时间步索引，从对应视频文件中解码各相机视角的帧：
        【中文】- 使用 modality.json 中的 original_key 与 info.json 中的 data_path/video_path 模板定位视频文件；
        【中文】- 调用 get_frames_by_indices 按 indices 抽帧，返回一个 {camera_name: np.ndarray[...] } 的字典。
        """
        # 初始化空字典用于存储视频数据
        # Initialize empty dictionary to store video data
        video_data = {}

        # 如果没有配置视频路径模式或视频模态，直接返回空字典
        # Return empty dict if no video path pattern or video modality configured
        if not self.video_path_pattern or "video" not in self.modality_configs:
            return video_data

        # 计算当前 episode 所在的 chunk 索引
        # Calculate chunk index for current episode
        chunk_idx = episode_index // self.chunk_size
        # 获取配置的所有图像键（相机视角）
        # Get all configured image keys (camera views)
        image_keys = self.modality_configs["video"].modality_keys

        # 遍历每个图像键，加载对应的视频数据
        # Iterate through each image key to load corresponding video data
        for image_key in image_keys:
            # 解析视频文件命名中使用的原始键名
            # Resolve the original key used in video file naming
            original_key = self.modality_meta["video"][image_key].get(
                "original_key", f"observation.images.{image_key}"
            )
            # 确保原始键存在于特征配置中
            # Ensure original key exists in feature config
            assert original_key in self.feature_config, (
                f"Original key {original_key} not found in feature config"
            )

            # 使用路径模板构造视频文件路径
            # Construct video file path using pattern
            video_filename = self.video_path_pattern.format(
                episode_chunk=chunk_idx, video_key=original_key, episode_index=episode_index
            )
            video_path = self.dataset_path / video_filename

            # 在指定的时间戳索引处解码视频帧
            # Decode video frames at specified timestamps
            video_data[image_key] = get_frames_by_indices(
                str(video_path),
                indices,
                video_backend=self.video_backend,
                video_backend_kwargs=self.video_backend_kwargs or {},
            )

        return video_data

    def get_dataset_statistics(self) -> dict[str, Any]:
        """
        Extract dataset statistics for normalization from loaded metadata.

        Constructs a nested dictionary containing statistics (mean, std, min, max, q01, q99)
        for each joint group in state and action modalities. These statistics are used
        by processors for data normalization during training.

        Returns:
            Nested dictionary: {modality: {joint_group: {stat_type: values}}}

        【中文】从 self.stats 中抽取指定模态/关节组对应的统计量（mean/std/min/max/q01/q99），
        【中文】根据 modality.json 中记录的 start/end 切片到各关节组维度，并组织成嵌套字典返回，
        【中文】供上层 Processor/StateActionProcessor 做归一化和裁剪使用；同时若存在 relative_action 统计，也一并挂到返回值中。
        """
        mapping = {"state": "observation.state", "action": "action"}
        dataset_statistics = _rec_defaultdict()

        for modality in mapping.keys():  # state, action
            for joint_key in self.modality_configs[modality].modality_keys:
                # Determine which statistics key to use
                if self.modality_meta[modality][joint_key].get("original_key", None) is not None:
                    stats_key = self.modality_meta[modality][joint_key]["original_key"]
                else:
                    stats_key = mapping[modality]

                # Extract the relevant slice of statistics
                start_idx, end_idx = (
                    self.modality_meta[modality][joint_key]["start"],
                    self.modality_meta[modality][joint_key]["end"],
                )
                for stat_type in self.stats[stats_key].keys():  # mean, std, min, max, q01, q99
                    dataset_statistics[modality][joint_key][stat_type] = self.stats[stats_key][
                        stat_type
                    ][start_idx:end_idx]
        stats = _to_plain_dict(dataset_statistics)
        # Directly add relative action stats
        if "relative_action" in self.stats:
            stats["relative_action"] = self.stats["relative_action"]
        return stats

    def create_language_from_meta(
        self, episode_meta: dict, nframes: int, lang_key: str
    ) -> list[str]:
        if lang_key == "task":
            meta_language = random.choice(episode_meta["tasks"])
            new_languages = [meta_language] * nframes
        elif lang_key == "sub_task":
            action_delta_indices = self.modality_configs["action"].delta_indices
            action_horizon = max(action_delta_indices) - min(action_delta_indices) + 1
            new_languages = [[] for _ in range(nframes)]
            sub_tasks = episode_meta["sub_tasks"]
            for sub_task in sub_tasks:
                start_idx, end_idx, sub_text = sub_task["start"], sub_task["end"], sub_task["text"]
                horizon = action_horizon // 2
                for i in range(start_idx - horizon, end_idx):
                    if i < 0:
                        continue
                    new_languages[i].append(sub_text)
            new_languages = [i if len(i) > 0 else [""] for i in new_languages]
            new_languages = [random.choice(i) for i in new_languages]
        else:
            raise ValueError(f"Language key {lang_key} not supported")
        return new_languages

    def __getitem__(self, idx: int) -> pd.DataFrame:
        """
        Load complete episode data as a processed DataFrame.

        Combines parquet data loading and video decoding to create a unified DataFrame
        containing all modality data for the episode. Video frames are converted to
        PIL Images and stored in the DataFrame.

        Args:
            idx: Episode index to load

        Returns:
            DataFrame with columns for all modalities and timestamps, with video frames
            as PIL Images ready for further processing

        Raises:
            IndexError: If episode index is out of bounds

        【中文】按 episode 索引加载完整一条轨迹，并整合数值模态 + 语言 + 视频：
        【中文】1）根据 episodes_metadata 中记录的 episode_index 找到对应 parquet 和视频文件；
        【中文】2）调用 _load_parquet_data 构造 state.xxx / action.xxx / language.xxx 列；
        【中文】3）如需语言，通过 create_language_from_meta 基于 meta 生成逐帧文本；
        【中文】4）调用 _load_video_data 解码同步的视频帧，填入 video.xxx 列；
        【中文】最终返回一个按时间步排列的 DataFrame，每行对应一个时间步，列包含所有已配置模态。
        """
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Episode index {idx} out of bounds")

        episode_meta = self.episodes_metadata[idx]
        episode_id = episode_meta["episode_index"]
        nominal_length = episode_meta["length"]

        # Load and parse the parquet data
        df = self._load_parquet_data(episode_id)

        if "language" in self.modality_configs:
            lang_key = self.modality_configs["language"].modality_keys[0]
            if lang_key in LANG_KEYS:
                new_languages = self.create_language_from_meta(episode_meta, len(df), lang_key)
                df["language." + lang_key] = new_languages

        # Use actual dataframe length (might be less than nominal)
        actual_length = min(len(df), nominal_length)
        df = df.iloc[:actual_length]

        # Load synchronized video data
        video_data = self._load_video_data(episode_id, np.arange(actual_length))

        # Add video frames to dataframe as PIL Images
        for key in video_data.keys():
            assert len(video_data[key]) == len(df), (
                f"Video data for {key} has length {len(video_data[key])} but dataframe has length {len(df)}"
            )
            df[f"video.{key}"] = [frame for frame in video_data[key]]

        return df

    def get_initial_actions(self):
        """
        Load initial actions for policy initialization if available.

        Returns:
            List containing initial action dictionaries, or empty list if not available

        【中文】可选地从 meta/initial_actions.json 中加载“初始动作”配置，
        【中文】用于部署或策略初始化阶段为机器人提供一个安全/合理的起始动作（例如站姿或初始关节配置）。
        """
        meta_dirpath = self.dataset_path / LEROBOT_META_DIR_NAME
        initial_actions_path = meta_dirpath / INITIAL_ACTIONS_FILENAME
        if initial_actions_path.exists():
            initial_actions = load_initial_actions(initial_actions_path)
            return initial_actions  # a single-element list of dict[str, dict[str, np.ndarray]]
        else:
            return []
