"""
SO100 Real-Robot Gr00T Policy Evaluation Script

This script runs closed-loop policy evaluation on the SO100 / SO101 robots
using the GR00T Policy API.

Major responsibilities:
    • Initialize robot hardware from a RobotConfig (LeRobot)
    • Convert robot observations into GR00T VLA inputs
    • Query the GR00T policy server (PolicyClient)
    • Decode multi-step (temporal) model actions back into robot motor commands
    • Stream actions to the real robot in real time

This file is meant to be a simple, readable reference
for real-world policy debugging and demos.
"""

# =============================================================================
# Imports
# =============================================================================

from dataclasses import asdict, dataclass
import logging
from pprint import pformat
import time
from typing import Any, Dict, List

import draccus
from gr00t.policy.server_client import PolicyClient

# Importing various robot configs ensures CLI autocompletion works
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    koch_follower,
    make_robot_from_config,
    so100_follower,
    so101_follower,
)
from lerobot.utils.utils import init_logging, log_say
import numpy as np


def recursive_add_extra_dim(obs: Dict) -> Dict:
    """递归地给观测加额外维度, 对齐 GR00T Policy Server 的输入约定。

    目标：把原始形如
        obs["video"][key]  ~  (H, W, C)
        obs["state"][key]  ~  (D,)
        obs["language"][key] ~ 标量字符串
    转换成 GR00T 期望的
        obs["video"][key]    ~ (B=1, T=1, H, W, C)
        obs["state"][key]    ~ (B=1, T=1, D)
        obs["language"][key] ~ [["instr"]]

    本函数每调用一次, 对数组或标量外面包一层 batch/time 维;
    这里调用两次, 最终形成 (1, 1, ...) 的形状, 与 `Gr00tPolicy.check_observation` 完全兼容。
    """
    for key, val in obs.items():
        if isinstance(val, np.ndarray):
            obs[key] = val[np.newaxis, ...]
        elif isinstance(val, dict):
            obs[key] = recursive_add_extra_dim(val)
        else:
            obs[key] = [val]  # scalar → [scalar]
    return obs


class So100Adapter:
    """
    Adapter between:
        • Raw robot observation dictionary
        • GR00T VLA input format
        • GR00T action chunk → robot joint commands

    Responsible for:
        • Packaging camera frames as obs["video"]
        • Building obs["state"] for arm + gripper
        • Adding language instruction
        • Adding batch/time dimensions
        • Decoding model action chunks into real robot actions
    """

    def __init__(self, policy_client: PolicyClient):
        self.policy = policy_client

        # SO100 joint ordering used for BOTH training + robot execution
        self.robot_state_keys = [
            "shoulder_pan.pos",
            "shoulder_lift.pos",
            "elbow_flex.pos",
            "wrist_flex.pos",
            "wrist_roll.pos",
            "gripper.pos",
        ]

        self.camera_keys = ["front", "wrist"]

    # -------------------------------------------------------------------------
    # Observation → Model Input
    # -------------------------------------------------------------------------
    def obs_to_policy_inputs(self, obs: Dict[str, Any]) -> Dict:
        """将原始机器人观测转换为 GR00T VLA 输入格式。

        目标：从 SO100 的原始字典
            obs = {
                "front": np.ndarray[(H, W, 3)],
                "wrist": np.ndarray[(H, W, 3)],
                "shoulder_pan.pos": float,
                ...
                "gripper.pos": float,
                "lang": str,
            }
        构造出:
            model_obs = {
                "video": {"front": (1,1,H,W,3), "wrist": (1,1,H,W,3)},
                "state": {"single_arm": (1,1,5), "gripper": (1,1,1)},
                "language": {"annotation.human.task_description": [[lang_str]]},
            }
        其中形状与 `embodiment_configs.py` 中 SO100 的配置以及 `Gr00tPolicy` 的输入契约一致。
        """
        model_obs = {}

        # (1) Cameras
        model_obs["video"] = {k: obs[k] for k in self.camera_keys}

        # (2) Arm + gripper state
        state = np.array([obs[k] for k in self.robot_state_keys], dtype=np.float32)
        model_obs["state"] = {
            "single_arm": state[:5],  # (5,)
            "gripper": state[5:6],  # (1,)
        }

        # (3) Language
        model_obs["language"] = {"annotation.human.task_description": obs["lang"]}

        # (4) Add (B=1, T=1) dims
        model_obs = recursive_add_extra_dim(model_obs)
        model_obs = recursive_add_extra_dim(model_obs)
        return model_obs

    # -------------------------------------------------------------------------
    # Model Action Chunk → Robot Motor Commands
    # -------------------------------------------------------------------------
    def decode_action_chunk(self, chunk: Dict, t: int) -> Dict[str, float]:
        """从模型动作 chunk 中取出第 t 个时间步, 还原为机器人每个关节的目标值。

        输入 chunk 示例 (来自 `policy.get_action` 的返回):
            chunk["single_arm"] ~ np.ndarray[(B=1, T, 5)]
            chunk["gripper"]    ~ np.ndarray[(B=1, T, 1)]

        输出 action_dict 示例 (供 `robot.send_action` 使用):
            {
                "shoulder_pan.pos":  float,
                "shoulder_lift.pos": float,
                "elbow_flex.pos":    float,
                "wrist_flex.pos":    float,
                "wrist_roll.pos":    float,
                "gripper.pos":       float,
            }
        """
        single_arm = chunk["single_arm"][0][t]  # (5,)
        gripper = chunk["gripper"][0][t]  # (1,)

        full = np.concatenate([single_arm, gripper], axis=0)  # (6,)

        return {joint_name: float(full[i]) for i, joint_name in enumerate(self.robot_state_keys)}

    def get_action(self, obs: Dict) -> List[Dict[str, float]]:
        """根据当前观测, 预测一段动作序列, 返回每一时间步的关节命令列表。

        - 输入 obs 形如 `robot.get_observation()` 的字典 (见 eval() 中注释示例);
        - 内部调用 `obs_to_policy_inputs` 转为 (B=1, T=1, ...) 形状后, 通过 `PolicyClient.get_action` 请求远程 Gr00tPolicy；
        - 返回值是长度为 horizon 的列表, 每个元素都是一份完整的关节目标字典, 可直接传给 `robot.send_action`。
        """
        model_input = self.obs_to_policy_inputs(obs)
        action_chunk, info = self.policy.get_action(model_input)

        # Determine horizon
        any_key = next(iter(action_chunk.keys()))
        horizon = action_chunk[any_key].shape[1]  # (B, T, D) → T

        return [self.decode_action_chunk(action_chunk, t) for t in range(horizon)]


# =============================================================================
# Evaluation Config
# =============================================================================


@dataclass
class EvalConfig:
    """
    Command-line configuration for real-robot policy evaluation.
    """

    robot: RobotConfig
    policy_host: str = "localhost"
    policy_port: int = 5555
    action_horizon: int = 8
    lang_instruction: str = "Grab markers and place into pen holder."
    play_sounds: bool = False
    timeout: int = 60


# =============================================================================
# Main Eval Loop
# =============================================================================


@draccus.wrap()
def eval(cfg: EvalConfig):
    """Main entry point for real-robot policy evaluation.

    目标：在真实 SO100 / SO101 机器人上, 以 closed-loop 方式跑一整条语言条件任务。
    - 步骤 1: 用 `make_robot_from_config(cfg.robot)` 初始化和连接真实机器人硬件；
    - 步骤 2: 通过 `PolicyClient(policy_host, policy_port)` 连接远程 Gr00t 推理服务器, 用 `So100Adapter` 适配观测/动作；
    - 步骤 3: 在一个 while True 控制循环中:
      - 从 `robot.get_observation()` 读取当前传感器与关节状态, 加上语言指令 `cfg.lang_instruction`；
      - 调用 `policy.get_action(obs)` 得到一段长度为 cfg.action_horizon 的动作序列；
      - 以固定频率 (约 30Hz) 依次将每一步动作下发给 `robot.send_action` 执行, 实现真实 closed-loop 控制。
    """
    init_logging()
    logging.info(pformat(asdict(cfg)))

    # -------------------------------------------------------------------------
    # 1. Initialize Robot Hardware
    # -------------------------------------------------------------------------
    robot = make_robot_from_config(cfg.robot)
    robot.connect()

    log_say("Initializing robot", cfg.play_sounds, blocking=True)

    # -------------------------------------------------------------------------
    # 2. Initialize Policy Wrapper + Client
    # -------------------------------------------------------------------------
    policy_client = PolicyClient(host=cfg.policy_host, port=cfg.policy_port)
    policy = So100Adapter(policy_client)

    log_say(
        f'Policy ready with instruction: "{cfg.lang_instruction}"',
        cfg.play_sounds,
        blocking=True,
    )

    # -------------------------------------------------------------------------
    # 3. Main real-time control loop
    # -------------------------------------------------------------------------
    while True:
        obs = robot.get_observation()
        obs["lang"] = cfg.lang_instruction  # insert language

        # obs = {
        #     "front": np.zeros((480, 640, 3), dtype=np.uint8),
        #     "wrist": np.zeros((480, 640, 3), dtype=np.uint8),
        #     "shoulder_pan.pos": 0.0,
        #     "shoulder_lift.pos": 0.0,
        #     "elbow_flex.pos": 0.0,
        #     "wrist_flex.pos": 0.0,
        #     "wrist_roll.pos": 0.0,
        #     "gripper.pos": 0.0,
        #     "lang": cfg.lang_instruction,
        # }

        actions = policy.get_action(obs)

        for i, action_dict in enumerate(actions[: cfg.action_horizon]):
            tic = time.time()
            print(f"action[{i}]: {action_dict}")
            # action_dict = {
            #     "shoulder_pan.pos":    5.038022994995117,
            #     "shoulder_lift.pos":  17.09104347229004,
            #     "elbow_flex.pos":    -18.519847869873047,
            #     "wrist_flex.pos":     86.86847686767578,
            #     "wrist_roll.pos":      1.0669738054275513,
            #     "gripper.pos":        36.83877944946289,
            # }
            robot.send_action(action_dict)
            toc = time.time()
            if toc - tic < 1.0 / 30:
                time.sleep(1.0 / 30 - (toc - tic))


if __name__ == "__main__":
    eval()
