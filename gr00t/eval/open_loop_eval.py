from copy import deepcopy
from dataclasses import dataclass, field
import logging
from pathlib import Path
import re
from typing import Any
import warnings

from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.policy import BasePolicy
from gr00t.policy.gr00t_policy import Gr00tPolicy
from gr00t.policy.server_client import PolicyClient
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import tyro


warnings.simplefilter("ignore", category=FutureWarning)

"""
Example commands:

NOTE: provide --model_path to load up the model checkpoint in this script,
        else it will use the default host and port via RobotInferenceClient

【中文】脚本用途概览：
- 离线 open-loop 推理与评估：在已有 LeRobot 轨迹数据上复现模型的预测动作，不与环境交互；
- 支持两种 policy 来源：
  - 本地加载 `Gr00tPolicy` 模型 (`--model_path` 不为空)；
  - 通过 `PolicyClient` 连接远程推理服务器 (未提供 `--model_path` 时)；
- 主要流程：
  1. 从 checkpoint 或远程服务构建 policy；
  2. 用 policy 的 modality_config 构造 `LeRobotEpisodeLoader` 数据集；
  3. 对每条 episode，每隔 `action_horizon` 步抽取观测，调用 `policy.get_action` 得到一段动作轨迹；
  4. 将预测动作展开到时间轴上，与 GT 动作拼接在一起，对齐后计算 MSE/MAE 并画图。
"""


def plot_trajectory_results(
    state_joints_across_time: np.ndarray,
    gt_action_across_time: np.ndarray,
    pred_action_across_time: np.ndarray,
    traj_id: int,
    state_keys: list[str],
    action_keys: list[str],
    action_horizon: int,
    save_plot_path: str,
) -> None:
    """绘制并保存一条轨迹上 GT 动作与预测动作的对比曲线。

    目标：给定一条 episode 上按时间展开的关节状态 / GT 动作 / 预测动作，
    对每一维 action 画出随时间变化的曲线，并用红点标出每次推理发生的时间步。

    Args:
        state_joints_across_time: 状态关节随时间的拼接数组，形状 (T, D_state)
        gt_action_across_time:    GT 动作随时间的拼接数组，形状 (T, D_action)
        pred_action_across_time:  预测动作随时间的拼接数组，形状 (T, D_action)
        traj_id:                  当前绘制的 episode 编号
        state_keys:               state 中参与拼接的 key 列表
        action_keys:              action 中参与拼接的 key 列表（决定 D_action 的结构）
        action_horizon:           推理 horizon，主要用于在时间轴上标出推理点（每隔 horizon 一次）
        save_plot_path:           图片保存路径
    """
    actual_steps = len(gt_action_across_time)
    action_dim = gt_action_across_time.shape[1]

    indices_to_plot = list(range(action_dim))

    num_plots = len(indices_to_plot)
    if num_plots == 0:
        logging.warning("No valid indices to plot")
        return

    # Always plot and save
    fig, axes = plt.subplots(nrows=num_plots, ncols=1, figsize=(8, 4 * num_plots))

    # Handle case where there's only one subplot
    if num_plots == 1:
        axes = [axes]

    # Add a global title showing the modality keys
    fig.suptitle(
        f"Trajectory {traj_id} - State: {', '.join(state_keys)} | Action: {', '.join(action_keys)}",
        fontsize=16,
        color="blue",
    )

    for plot_idx, action_idx in enumerate(indices_to_plot):
        ax = axes[plot_idx]

        # The dimensions of state_joints and action are the same
        # only when the robot uses actions directly as joint commands.
        # Therefore, do not plot them if this is not the case.
        if state_joints_across_time.shape == gt_action_across_time.shape:
            ax.plot(state_joints_across_time[:, action_idx], label="state joints")
        ax.plot(gt_action_across_time[:, action_idx], label="gt action")
        ax.plot(pred_action_across_time[:, action_idx], label="pred action")

        # put a dot every ACTION_HORIZON
        for j in range(0, actual_steps, action_horizon):
            if j == 0:
                ax.plot(j, gt_action_across_time[j, action_idx], "ro", label="inference point")
            else:
                ax.plot(j, gt_action_across_time[j, action_idx], "ro")

        ax.set_title(f"Action {action_idx}")
        ax.legend()

    plt.tight_layout()

    # Create filename with trajectory ID
    Path(save_plot_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_plot_path)

    plt.close()  # Close the figure to free memory


def parse_observation_gr00t(
    obs: dict[str, Any], modality_configs: dict[str, Any]
) -> dict[str, Any]:
    """【中文】将原始观测字典整理为 Gr00tPolicy 期望的三模态结构。

    - 输入 obs: 形如 {"state.x": array[(T,D)], "video.front": array[(T,H,W,C)], "annotation.human...": str} 的扁平 key；
    - 输入 modality_configs: 来自 policy 的 modality 配置, 指定每个模态下应该有哪些 key 以及各自的时间长度；
    - 输出: {
        "video": {key: np.ndarray[(B=1, T_video, H, W, C)]},
        "state": {key: np.ndarray[(B=1, T_state, D)]},
        "language": {key: list[list[str]] 形如 [["instruction"]]},
      } 可直接喂给 `policy.get_action`。
    """
    new_obs = {}
    for modality in ["video", "state", "language"]:
        new_obs[modality] = {}
        for key in modality_configs[modality].modality_keys:
            if modality == "language":
                parsed_key = key
            else:
                parsed_key = f"{modality}.{key}"
            arr = obs[parsed_key]
            # Add batch dimension
            if isinstance(arr, str):
                new_obs[modality][key] = [[arr]]
            else:
                new_obs[modality][key] = arr[None, :]
    return new_obs


def parse_action_gr00t(action: dict[str, Any]) -> dict[str, Any]:
    """【中文】将 policy 输出的 batched 动作解包, 并恢复 DataFrame 使用的前缀格式。

    - 输入 action: 形如 {"left_arm": np.ndarray[(B,T,D)], "gripper": ...}；
    - 这里假设 B=1, 因此取第 0 个 batch 维度, 得到 (T,D)；
    - 输出: {"action.left_arm": (T,D), "action.gripper": ...}, 便于与 LeRobot 的 episode DataFrame 对齐。
    """
    # Unbatch and add prefix
    return {f"action.{key}": action[key][0] for key in action}


def evaluate_single_trajectory(
    policy: BasePolicy,
    loader: LeRobotEpisodeLoader,
    traj_id: int,
    embodiment_tag: EmbodimentTag,
    modality_keys: list[str] | None = None,
    steps=300,
    action_horizon=16,
    save_plot_path=None,
):
    """【中文】在单条 episode 上做 open-loop 推理评估。

    目标：不干预环境，只在已有 LeRobot 轨迹上重放观测并跑模型，比较“模型预测的动作序列”和“数据集中的 GT 动作序列”。

    - 每隔 `action_horizon` 步抽一帧观测, 调用一次 `policy.get_action`, 得到一段长度为 horizon 的动作序列;
    - 将所有预测的 horizon 片段沿时间展开成一条长序列 `pred_action_across_time` (形状约为 (T, D_all));
    - 用同样的方式从 DataFrame 中拼出 GT 动作 `gt_action_across_time`, 与预测对齐后计算 MSE / MAE, 并画 GT vs 预测曲线。
    """
    # Ensure steps doesn't exceed trajectory length
    traj = loader[traj_id]
    traj_length = len(traj)
    actual_steps = min(steps, traj_length)
    logging.info(
        f"Using {actual_steps} steps (requested: {steps}, trajectory length: {traj_length})"
    )

    pred_action_across_time = []

    # Extract state and action keys separately and sort for consistent order
    # 所有state和action的名字,left_arm,....
    state_keys = loader.modality_configs["state"].modality_keys
    action_keys = (
        loader.modality_configs["action"].modality_keys if modality_keys is None else modality_keys
    )

    modality_configs = deepcopy(loader.modality_configs)
    modality_configs.pop("action") # 没有action的modality，这样extract_step_data就不会处理要推理的action
    # horizon间隔取帧
    for step_count in range(0, actual_steps, action_horizon):
        # 获取第 step_count 帧的输入数据, 这里modality_configs是：
        '''
            ##### Pre-registered posttrain configurations #####
            "unitree_g1": {
                "video": ModalityConfig(
                    delta_indices=[0],
                    modality_keys=["ego_view"],
                ),
                "state": ModalityConfig(
                    delta_indices=[0],
                    modality_keys=[
                        "left_leg",
                        "right_leg",
                        "waist",
                        "left_arm",
                        "right_arm",
                        "left_hand",
                        "right_hand",
                    ],
                ),
                "action": ModalityConfig(
                    delta_indices=list(range(30)),
                    modality_keys=[
                        "left_arm",
                        "right_arm",
                        "left_hand",
                        "right_hand",
                        "waist",
                        "base_height_command",
                        "navigate_command",
                    ],
        '''
        # 然后lerobot数据的column里面是state.left_arm, action.left_arm这样，这样来提取VLAStepData
        data_point = extract_step_data(traj, step_count, modality_configs, embodiment_tag)
        '''
           !!!!!!! VLAStepData是SingleDataSet的原始返回，没有经过Processor进一步归一化、相对角度等处理。

                # 构造并返回 VLAStepData 对象
                vla_step_data = VLAStepData(
                    images=video_data,
                    states=state_data,
                    actions=action_data,
                    text=text,
                    embodiment=embodiment_tag,
                )
        '''
        logging.info(f"inferencing at step: {step_count}")

        # 把data_point搞成observation的格式要求，用于传给policy
        obs = {}
        for k, v in data_point.states.items():
            obs[f"state.{k}"] = v  # (T, D)
        for k, v in data_point.images.items():
            obs[f"video.{k}"] = np.array(v)  # (T, H, W, C)
        for language_key in loader.modality_configs["language"].modality_keys:
            obs[language_key] = data_point.text
        parsed_obs = parse_observation_gr00t(obs, loader.modality_configs)

        # 执行VLA推理
        _action_chunk, _ = policy.get_action(parsed_obs)

        # _action_chunk 格式样例: {"left_arm": np.ndarray[(B=1, T=horizon, D=7)], "gripper": ...}
        # action_chunk 整理后的格式样例: {"action.left_arm": np.ndarray[(T=horizon, D=7)], "action.gripper": ...}
        # pred_action_across_time 最终结果样例: [np.ndarray(D_total), np.ndarray(D_total), ...] (列表长度为 actual_steps)
        action_chunk = parse_action_gr00t(_action_chunk)
        for j in range(action_horizon):
            # 遍历 action_keys 中的每一个 key (如 'left_arm', 'gripper')，取其第 j 步的动作值并拼接成一个长向量
            # the np.atleast_1d is to ensure the action is a 1D array, handle where single value is returned
            concat_pred_action = np.concatenate(
                [
                    np.atleast_1d(np.atleast_1d(action_chunk[f"action.{key}"])[j])
                    for key in action_keys
                ],
                axis=0,
            )
            pred_action_across_time.append(concat_pred_action)


    # traj 是一个 episode 的 DataFrame，包含该轨迹中的若干个时间步 (steps)
    def extract_state_joints(traj: pd.DataFrame, columns: list[str]):
        """
        从 DataFrame 中提取指定的列（如 state.* 或 action.*）并将它们拼接成一个大的二维数组。
        
        数据样例说明：
        假设 columns = ["state.left_arm", "state.gripper"]
        - traj["state.left_arm"] 是一个 Series，其长度 T 代表轨迹的时间步总数 (steps)，
          每个元素是该时间步下的动作/状态，例如 shape=(7,) 的 ndarray。
        - traj["state.gripper"]  同样是一个长度为 T (steps) 的 Series，每个元素 shape=(1,)。
        
        执行过程：
        1. np.vstack([arr for arr in traj["state.left_arm"]]) -> shape=(T, 7)，即 (steps, dim_left_arm)
        2. np.vstack([arr for arr in traj["state.gripper"]])  -> shape=(T, 1)，即 (steps, dim_gripper)
        3. np.concatenate([ (T,7), (T,1) ], axis=-1)         -> shape=(T, 8)，即 (steps, dim_total)
        
        返回：
            (T, D_total) 的 ndarray，其中 T 是轨迹的时间步数 (steps)，D_total 是所有所选列维度之和。
        """
        np_dict = {}
        for column in columns:
            # traj[column] 每一行通常是 shape=(D,) 的向量; vstack 后得到 (T, D_col)
            np_dict[column] = np.vstack([arr for arr in traj[column]])
        # 按列顺序在最后一维（特征维）拼接, 得到 (T, sum(D_col)) 的大向量
        return np.concatenate([np_dict[column] for column in columns], axis=-1)

    # 提取状态数据用于绘图 (T, D_state)
    state_joints_across_time = extract_state_joints(traj, [f"state.{key}" for key in state_keys]) # traj是一个episode，里面若干steps
    
    # 提取数据集中的真值动作 (GT)，并截断到评估步数 (actual_steps, D_action)
    gt_action_across_time = extract_state_joints(traj, [f"action.{key}" for key in action_keys])[
        :actual_steps
    ]
    
    # 将模型预测的动作列表转换为 array，同样截断到评估步数 (actual_steps, D_action)
    # pred_action_across_time 之前是通过 append 得到的列表，每个元素是 concat 后的 1D 动作
    pred_action_across_time = np.array(pred_action_across_time)[:actual_steps]
    
    # 确保 GT 和预测动作的形状完全一致，否则无法计算 MSE/MAE
    assert gt_action_across_time.shape == pred_action_across_time.shape, (
        f"gt_action: {gt_action_across_time.shape}, pred_action: {pred_action_across_time.shape}"
    )

    # calc MSE and MAE across time
    mse = np.mean((gt_action_across_time - pred_action_across_time) ** 2)
    mae = np.mean(np.abs(gt_action_across_time - pred_action_across_time))
    logging.info(f"Unnormalized Action MSE across single traj: {mse}")
    logging.info(f"Unnormalized Action MAE across single traj: {mae}")

    logging.info(f"state_joints vs time {state_joints_across_time.shape}")
    logging.info(f"gt_action_joints vs time {gt_action_across_time.shape}")
    logging.info(f"pred_action_joints vs time {pred_action_across_time.shape}")

    # Plot trajectory results
    plot_trajectory_results(
        state_joints_across_time=state_joints_across_time,
        gt_action_across_time=gt_action_across_time,
        pred_action_across_time=pred_action_across_time,
        traj_id=traj_id,
        state_keys=state_keys,
        action_keys=action_keys,
        action_horizon=action_horizon,
        save_plot_path=save_plot_path or f"/tmp/open_loop_eval/traj_{traj_id}.jpeg",
    )

    return mse, mae


@dataclass
class ArgsConfig:
    """Configuration for evaluating a policy."""

    host: str = "127.0.0.1"
    """Host to connect to."""

    port: int = 5555
    """Port to connect to."""

    steps: int = 200
    """Maximum number of steps to evaluate (will be capped by trajectory length)."""

    traj_ids: list[int] = field(default_factory=lambda: [0])
    """List of trajectory IDs to evaluate."""

    action_horizon: int = 16
    """Action horizon to evaluate."""

    dataset_path: str = "demo_data/cube_to_bowl_5/"
    """Path to the dataset."""

    embodiment_tag: EmbodimentTag = EmbodimentTag.NEW_EMBODIMENT
    """Embodiment tag to use."""

    model_path: str | None = None
    """Path to the model checkpoint."""

    denoising_steps: int = 4
    """Number of denoising steps to use."""

    save_plot_path: str | None = None
    """Path to save the plot to."""

    modality_keys: list[str] | None = None
    """List of modality keys to plot. If None, plot all keys."""


def main(args: ArgsConfig):
    """【中文】脚本入口: 构建 policy 和数据集, 遍历若干条轨迹做推理评估。

    - 若提供 `args.model_path`, 本地加载 Gr00tPolicy 模型; 否则通过 PolicyClient 连远程推理服务；
    - 使用 policy 的 modality_config 构造 LeRobotEpisodeLoader, 保证数据字段与模型配置一致；
    - 对 `args.traj_ids` 中的每个 traj 调用 `evaluate_single_trajectory`, 统计并打印整体 MSE/MAE。
    """
    # Set up logging
    logging.basicConfig(level=logging.INFO)

    # Download model checkpoint if it's an S3 path
    local_model_path = args.model_path

    # Extract global_step and checkpoint directory name from checkpoint path
    global_step = None
    if local_model_path:
        # Search for pattern "checkpoint-{number}" anywhere in the path
        match = re.search(r"checkpoint-(\d+)", local_model_path)
        if match:
            try:
                global_step = int(match.group(1))
                logging.info(f"Extracted global_step {global_step} from checkpoint path")
            except ValueError:
                logging.warning(
                    f"Could not parse step number from checkpoint path: {local_model_path}"
                )
        else:
            logging.warning(f"Could not find checkpoint-<step> pattern in path: {local_model_path}")

    if local_model_path is not None:
        import torch

        # 推理的时候的包装类，封装了“观测 → Processor → 模型 → 反归一化动作”的完整推理链
        policy = Gr00tPolicy(
            embodiment_tag=args.embodiment_tag,
            model_path=local_model_path,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
    else:
        policy = PolicyClient(host=args.host, port=args.port)

    # Get the supported modalities for the policy
    modality = policy.get_modality_config()
    logging.info(f"Current modality config: \n{modality}")

    # Create the dataset
    dataset = LeRobotEpisodeLoader(
        dataset_path=args.dataset_path,
        modality_configs=modality,
        video_backend="torchcodec",
        video_backend_kwargs=None,
    )

    logging.info(f"Dataset length: {len(dataset)}")
    logging.info(f"Running evaluation on trajectories: {args.traj_ids}")

    all_mse = []
    all_mae = []

    for traj_id in args.traj_ids:
        if traj_id >= len(dataset):
            logging.warning(f"Trajectory ID {traj_id} is out of range. Skipping.")
            continue

        logging.info(f"Running trajectory: {traj_id}")
        mse, mae = evaluate_single_trajectory(
            policy,
            dataset,
            traj_id,
            args.embodiment_tag,
            args.modality_keys,
            steps=args.steps,
            action_horizon=args.action_horizon,
            save_plot_path=args.save_plot_path,
        )
        logging.info(f"MSE for trajectory {traj_id}: {mse}, MAE: {mae}")
        all_mse.append(mse)
        all_mae.append(mae)

    if all_mse:
        avg_mse = np.mean(np.array(all_mse))
        avg_mae = np.mean(np.array(all_mae))
        logging.info(f"Average MSE across all trajs: {avg_mse}")
        logging.info(f"Average MAE across all trajs: {avg_mae}")
    else:
        logging.info("No valid trajectories were evaluated.")
    logging.info("Done")


if __name__ == "__main__":
    # Parse arguments using tyro
    config = tyro.cli(ArgsConfig)
    main(config)
