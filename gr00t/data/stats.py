#!/usr/bin/env python
"""
Calculate dataset statistics for LeRobot datasets.
Note: Please update the `gr00t/configs/data/embodiment_configs.py` file with the correct modality configurations for the dataset you are using before running this script.

Usage:
    python gr00t/data/stats.py <dataset_path> <embodiment_tag>

Args:
    dataset_path: Path to the dataset.
    embodiment_tag: Embodiment tag to use to load modality configurations from `gr00t/configs/data/embodiment_configs.py`.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS
from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
from gr00t.data.state_action.action_chunking import EndEffectorActionChunk, JointActionChunk
from gr00t.data.state_action.pose import EndEffectorPose, JointPose
from gr00t.data.types import ActionRepresentation, ActionType, EmbodimentTag, ModalityConfig
from gr00t.data.utils import to_json_serializable


LE_ROBOT_DATA_FILENAME = "data/*/*.parquet"
LE_ROBOT_INFO_FILENAME = "meta/info.json"
LE_ROBOT_STATS_FILENAME = "meta/stats.json"
LE_ROBOT_REL_STATS_FILENAME = "meta/relative_stats.json"


def calculate_dataset_statistics(
    parquet_paths: list[Path], features: list[str] | None = None
) -> dict[str, dict[str, float]]:
    """Calculate the dataset statistics of all columns for a list of parquet files.

    Args:
        parquet_paths (list[Path]): List of paths to parquet files to process.
        features (list[str] | None): List of feature names to compute statistics for.
            If None, computes statistics for all columns in the data.

    Returns:
        dict[str, DatasetStatisticalValues]: Dictionary mapping feature names to their
            statistical values (mean, std, min, max, q01, q99).
    """
    # Dataset statistics
    all_low_dim_data_list = []
    # Collect all the data
    for parquet_path in tqdm(
        sorted(list(parquet_paths)),
        desc="Collecting all parquet files...",
    ):
        # Load the parquet file
        parquet_data = pd.read_parquet(parquet_path)
        parquet_data = parquet_data
        all_low_dim_data_list.append(parquet_data)
    all_low_dim_data = pd.concat(all_low_dim_data_list, axis=0)
    # Compute dataset statistics
    dataset_statistics = {}
    if features is None:
        features = list(all_low_dim_data.columns)
    for le_modality in features:
        print(f"Computing statistics for {le_modality}...")
        np_data = np.vstack(
            [np.asarray(x, dtype=np.float32) for x in all_low_dim_data[le_modality]]
        )
        dataset_statistics[le_modality] = dict(
            mean=np.mean(np_data, axis=0).tolist(),
            std=np.std(np_data, axis=0).tolist(),
            min=np.min(np_data, axis=0).tolist(),
            max=np.max(np_data, axis=0).tolist(),
            q01=np.quantile(np_data, 0.01, axis=0).tolist(),
            q99=np.quantile(np_data, 0.99, axis=0).tolist(),
        )
    return dataset_statistics


def check_stats_validity(dataset_path: Path | str, features: list[str]):
    stats_path = Path(dataset_path) / LE_ROBOT_STATS_FILENAME
    if not stats_path.exists():
        return False
    with open(stats_path, "r") as f:
        stats = json.load(f)
    for feature in features:
        if feature not in stats:
            return False
        if not isinstance(stats[feature], dict):
            return False
        for stat in ["mean", "std", "min", "max", "q01", "q99"]:
            if stat not in stats[feature]:
                return False
    return True


def generate_stats(dataset_path: Path | str):
    dataset_path = Path(dataset_path)
    print(f"Generating stats for {str(dataset_path)}")
    lowdim_features = []
    with open(dataset_path / LE_ROBOT_INFO_FILENAME, "r") as f:
        le_features = json.load(f)["features"]
    for feature in le_features:
        if "float" in le_features[feature]["dtype"]:
            lowdim_features.append(feature)
    if check_stats_validity(dataset_path, lowdim_features):
        return

    parquet_files = list(dataset_path.glob(LE_ROBOT_DATA_FILENAME))
    stats = calculate_dataset_statistics(parquet_files, lowdim_features)
    stats_path = dataset_path / LE_ROBOT_STATS_FILENAME
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=4)


class RelativeActionLoader:
    """Loader for computing relative action trajectories for a single joint group.
    【中文】从 LeRobot episode 数据中抽取指定 action 关节组的相对动作轨迹，用于后续统计/归一化。
    """

    def __init__(self, dataset_path: Path | str, embodiment_tag: EmbodimentTag, action_key: str):
        self.dataset_path = Path(dataset_path)
        self.modality_configs: dict[str, ModalityConfig] = {}
        self.action_key = action_key
        # Check action config
        # 【中文】根据具身形态和 action_key 找到对应的 ActionConfig（表示方式、类型、格式等）
        assert action_key in MODALITY_CONFIGS[embodiment_tag.value]["action"].modality_keys
        idx = MODALITY_CONFIGS[embodiment_tag.value]["action"].modality_keys.index(action_key)
        action_configs = MODALITY_CONFIGS[embodiment_tag.value]["action"].action_configs
        assert action_configs is not None, MODALITY_CONFIGS[embodiment_tag.value]["action"]
        self.action_config = action_configs[idx]
        # 【中文】只保留当前 action_key 的 ModalityConfig，形成一个简化版的 action 配置
        self.modality_configs["action"] = ModalityConfig(
            delta_indices=MODALITY_CONFIGS[embodiment_tag.value]["action"].delta_indices,
            modality_keys=[action_key],
        )
        # Check state config
        # 【中文】某些动作可以指定单独的 state_key，否则默认与 action_key 同名
        state_key = self.action_config.state_key or action_key
        assert state_key in MODALITY_CONFIGS[embodiment_tag.value]["state"].modality_keys
        # 【中文】只保留对应 state_key 的 ModalityConfig，用于取参考状态
        self.modality_configs["state"] = ModalityConfig(
            delta_indices=MODALITY_CONFIGS[embodiment_tag.value]["state"].delta_indices,
            modality_keys=[state_key],
        )
        # Check state-action consistency
        # 【中文】保证 state 的最后一个时间索引 == action 序列的第一个时间索引，方便滑动窗口对齐
        assert (
            self.modality_configs["state"].delta_indices[-1]
            == self.modality_configs["action"].delta_indices[0]
        )
        # 【中文】底层使用 LeRobotEpisodeLoader 逐 episode 加载 DataFrame（state/action/video 等列）
        self.loader = LeRobotEpisodeLoader(dataset_path, self.modality_configs)

    def load_relative_actions(self, trajectory_id: int) -> list[np.ndarray]:
        """Load relative actions for a single episode.

        【中文】对单个 episode：
        1. 读取 state 与 action 序列
        2. 以某一时刻的 state 作为参考
        3. 取一段 action horizon，并在关节/EEF 空间内减去参考，得到相对动作轨迹
        """
        df = self.loader[trajectory_id]

        # OPTIMIZATION: Extract columns once and convert to numpy arrays
        # This eliminates repeated DataFrame.__getitem__ and Series.__getitem__ calls
        # 【中文】构造用于访问 DataFrame 的列名：state.xxx / action.xxx
        if self.action_config.state_key is not None:
            state_key = f"state.{self.action_config.state_key}"
        else:
            state_key = f"state.{self.action_key}"
        action_key = f"action.{self.action_key}"

        # Convert to numpy arrays once - this is much faster than repeated pandas access
        # 【中文】DataFrame 中每行是一帧，单元格是一个关节向量；这里取出整条 episode 的数组表示
        state_data = df[state_key].values  # Shape: (episode_length, joint_dim)
        action_data = df[action_key].values  # Shape: (episode_length, joint_dim)
        trajectories = []
        # 【中文】可用起点数 = 总步数 - action horizon 最大偏移，后面会做滑动窗口遍历
        usable_length = len(df) - self.modality_configs["action"].delta_indices[-1]
        action_delta_indices = np.array(self.modality_configs["action"].delta_indices)
        for i in range(usable_length):
            # 【中文】state_ind：当前窗口参考状态所在的时间步索引
            state_ind = self.modality_configs["state"].delta_indices[-1] + i
            # 【中文】action_inds：当前窗口内 action 的时间步索引（如 i..i+29）
            action_inds = action_delta_indices + i
            last_state = state_data[state_ind]
            actions = action_data[action_inds]
            if self.action_config.type == ActionType.EEF:
                # raise NotImplementedError("EEF action is not yet supported")
                assert len(last_state) == 9  # xyz + rot6d
                assert actions.shape[1] == 9  # xyz + rot6d

                # 【中文】EEF 类型：在末端执行器 (xyz+rotation) 空间中计算相对轨迹
                reference_frame = EndEffectorPose(
                    translation=last_state[:3],
                    rotation=last_state[3:],
                    rotation_type="rot6d",
                )

                traj = EndEffectorActionChunk(
                    [
                        EndEffectorPose(translation=m[:3], rotation=m[3:], rotation_type="rot6d")
                        for m in actions
                    ]
                ).relative_chunking(reference_frame=reference_frame)

                raise NotImplementedError(
                    "EEF action is not yet supported, need to handle rotation transformation based on action format"
                )
            elif self.action_config.type == ActionType.NON_EEF:
                # 【中文】关节空间类型：把 state / action 都看作关节姿态，用减法得到相对关节位移轨迹
                reference_frame = JointPose(last_state)
                traj = JointActionChunk([JointPose(m) for m in actions]).relative_chunking(
                    reference_frame=reference_frame
                )
                # 【中文】将 JointPose 序列堆成 (horizon, joint_dim) 的数组，并加入结果列表
                trajectories.append(np.stack([p.joints for p in traj.poses], dtype=np.float32))
            else:
                raise ValueError(f"Unknown ActionType: {self.action_config.type}")
        return trajectories

    def __len__(self) -> int:
        # 【中文】返回 episode 数量，使 loader 可以被 len() 调用
        return len(self.loader)


def calculate_stats_for_key(
    dataset_path: Path | str,
    embodiment_tag: EmbodimentTag,
    group_key: str,
    max_episodes: int = -1,
) -> dict:
    loader = RelativeActionLoader(dataset_path, embodiment_tag, group_key)
    trajectories = []
    for episode_id in tqdm(range(len(loader)), desc=f"Loading trajectories for key {group_key}"):
        if max_episodes != -1 and episode_id >= max_episodes:
            break
        trajectories.extend(loader.load_relative_actions(episode_id))
    return {
        "max": np.max(trajectories, axis=0),
        "min": np.min(trajectories, axis=0),
        "q01": np.quantile(trajectories, 0.01, axis=0),
        "q99": np.quantile(trajectories, 0.99, axis=0),
        "mean": np.mean(trajectories, axis=0),
        "std": np.std(trajectories, axis=0),
    }



def generate_rel_stats(dataset_path: Path | str, embodiment_tag: EmbodimentTag) -> None:
    """生成相对动作统计信息。
    
    【中文】为指定的数据集和具身形态生成相对动作的统计信息（min, max, mean, std, q01, q99）。
    仅处理配置为 RELATIVE 表示的动作关节组，统计结果保存到 meta/relative_stats.json。
    
    Args:
        dataset_path: 数据集路径
        embodiment_tag: 具身形态标签，用于从配置中加载对应的模态设置
    """
    dataset_path = Path(dataset_path)
    # 【中文】获取当前具身形态的 action 配置
    action_config = MODALITY_CONFIGS[embodiment_tag.value]["action"]
    # 【中文】如果没有配置 action_configs，直接返回（无需计算相对统计）
    if action_config.action_configs is None:
        return
    # 【中文】筛选出需要计算相对统计的 action_key（rep == RELATIVE）
    action_keys = [
        key
        for key, action_config in zip(action_config.modality_keys, action_config.action_configs)
        if action_config.rep == ActionRepresentation.RELATIVE
    ]
    # 【中文】相对统计信息文件路径
    stats_path = Path(dataset_path) / LE_ROBOT_REL_STATS_FILENAME
    # 【中文】如果统计文件已存在，则加载已有的统计信息
    if stats_path.exists():
        with open(stats_path, "r") as f:
            stats = json.load(f)
    else:
        stats = {}
    # 【中文】按字母顺序遍历需要计算的 action_key
    for action_key in sorted(action_keys):
        # 【中文】如果该 key 的统计已存在，跳过计算
        if action_key in stats:
            continue
        print(f"Generating relative stats for {dataset_path} {embodiment_tag} {action_key}")
        # 【中文】计算该 key 的相对动作统计信息（min, max, mean, std, q01, q99）
        stats[action_key] = calculate_stats_for_key(dataset_path, embodiment_tag, action_key)
    # 【中文】将统计信息转换为 JSON 可序列化格式并保存到文件
    with open(stats_path, "w") as f:
        json.dump(to_json_serializable(dict(stats)), f, indent=4)


def main(dataset_path: Path | str, embodiment_tag: EmbodimentTag):
    generate_stats(dataset_path)
    generate_rel_stats(dataset_path, embodiment_tag)


if __name__ == "__main__":
    import tyro

    tyro.cli(main)
