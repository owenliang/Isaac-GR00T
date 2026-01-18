"""Gr00t Policy implementation for inference.

This module provides the core policy classes for running Gr00t models:
- Gr00tPolicy: Base policy class for model inference
- Gr00tSimPolicyWrapper: Wrapper for compatibility with existing Gr00t simulation environments

【中文】模块功能概览：
- 负责“**把外部观测 → 喂给 VLA Processor/模型 → 解码为物理动作**”这一整条推理链；
- 默认使用 `AutoModel` + `AutoProcessor` 从 checkpoint 目录中恢复模型与 Processor；
- 支持两类观测格式：
  - 直接使用 `Gr00tPolicy` 时，期望输入是嵌套三模态结构：
    - video: dict[key -> np.ndarray[(B, T, H, W, C)]]
    - state: dict[key -> np.ndarray[(B, T, D)]]
    - language: dict[key -> list[list[str]]]  (形如 [["instruction"]])
  - 使用 `Gr00tSimPolicyWrapper` 时，支持旧版仿真环境的扁平 key 格式 (例如 'video.cam', 'state.joints')。

典型调用流程：
1. 使用 `Gr00tPolicy(embodiment_tag, model_path, device)` 构造策略对象；
2. 构造与训练时一致的 observation 结构，调用 `policy.get_action(observation)`；
3. 得到的返回值为 `(action_dict, info_dict)`：
   - action_dict: {action_key -> np.ndarray[(B, T_action, D_key)]}
   - info_dict: 预留调试信息，一般可忽略。
"""

from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModel, AutoProcessor

from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.interfaces import BaseProcessor
from gr00t.data.types import MessageType, ModalityConfig, VLAStepData

from .policy import BasePolicy, PolicyWrapper


def _rec_to_dtype(x: Any, dtype: torch.dtype) -> Any:
    """Recursively convert all floating point tensors in a nested structure to the given dtype.

    Args:
        x: Input data structure (tensor, dict, list, or other)
        dtype: Target torch dtype for floating point tensors

    Returns:
        Data structure with floating point tensors converted to target dtype

    Warning:
        Non-floating point tensors will be left as is.

    【中文】用途简述：
    - 递归地遍历 `x`（可以是张量 / dict / list 等），把所有**浮点张量**转换为指定精度 `dtype`；
    - 常用于在推理前统一把 Processor 输出中的 float32 张量转成 bfloat16，避免手动一层层 `.to()`；
    - 非浮点张量（int、bool 等）保持不变，保证数据类型安全。

    示例：
    - 输入: {"imgs": torch.randn(1, 3, 224, 224), "ids": torch.tensor([1, 2])}
    - 调用: `_rec_to_dtype(x, torch.bfloat16)`
    - 输出: imgs.dtype == bfloat16, ids.dtype 仍为原来的整型。
    """
    if isinstance(x, torch.Tensor) and torch.is_floating_point(x):
        return x.to(dtype=dtype)
    # Handle dict-like objects (tianshou.BatchFeature is not dict but has items() method)
    elif isinstance(x, dict) or hasattr(x, "items"):
        return {k: _rec_to_dtype(v, dtype) for k, v in x.items()}  # type: ignore
    elif isinstance(x, list):
        return [_rec_to_dtype(v, dtype) for v in x]
    else:
        return x


class Gr00tPolicy(BasePolicy):
    """Core policy class for Gr00t model inference.

    This policy handles the end-to-end inference pipeline:
    1. Validates input observations
    2. Processes observations with pretrained VLA processor
    3. Runs model inference
    4. Decodes and returns actions

    The policy expects observations with specific modalities (video, state, language)
    and returns actions in the format defined by the model's modality configuration.

    【中文】类职责总结：
    - 封装“**观测 → Processor → 模型 → 反归一化动作**”的完整推理链；
    - 内部持有：
      - `self.model`: 从 checkpoint 加载的 Gr00t 模型 (DiT 等)；
      - `self.processor`: 对应的 AutoProcessor, 负责 VLA 预处理与动作解码；
      - `self.modality_configs`: 当前具身标签下的模态配置 (video/state/action/language)；
    - 对外暴露的核心接口：
      - `get_action(observation) -> (action_dict, info_dict)`：给一批观测，返回每个 action key 的动作 chunk；
      - `check_observation` / `check_action`: 辅助验证输入输出格式是否与配置一致。
    """

    def __init__(
        self,
        embodiment_tag: EmbodimentTag,
        model_path: str,
        *,
        device: int | str,
        strict: bool = True,
    ):
        """Initialize the Gr00t Policy.

        Args:
            embodiment_tag: The embodiment tag defining the robot/environment type
            model_path: Path to the pretrained model checkpoint directory
            device: Device to run the model on (e.g., 'cuda:0', 0, 'cpu')
            strict: Whether to enforce strict input validation (default: True)

        【中文】初始化说明：
        - `embodiment_tag`: 指定本策略对应的具身形态 (例如 GR1、SO100 等)，用于从 Processor 中挑出正确的 `modality_configs` 与统计参数；
        - `model_path`: 预训练 checkpoint 目录，要求该目录下包含:
          - `config.json` / `model.safetensors` 等模型文件；
          - 与之配套的 Processor 配置 (例如 `processor_config.json`)；
        - `device`: 模型推理所在设备，通常传 "cuda" / "cuda:0" / 整数 GPU id 或 "cpu"；
        - 初始化过程中会：
          1. 通过 `AutoModel.from_pretrained(model_dir)` 恢复模型，并设置为 eval + bfloat16；
          2. 通过 `AutoProcessor.from_pretrained(model_dir)` 恢复 Processor；
          3. 记录当前 `embodiment_tag` 对应的模态配置与 collate 函数，后续 `get_action` 会用到。
        """
        # Import this to register all models.
        import gr00t.model  # noqa: F401

        super().__init__(strict=strict)
        model_dir = Path(model_path)

        # Load the pretrained model and move to target device with bfloat16 precision
        model = AutoModel.from_pretrained(model_dir)
        model.eval()  # Set model to evaluation mode
        model.to(device=device, dtype=torch.bfloat16)
        self.model = model

        # Load the processor for input/output transformation
        self.processor: BaseProcessor = AutoProcessor.from_pretrained(model_dir)
        self.processor.eval()

        # Store embodiment-specific configurations
        # 【中文】模态配置来源说明：
        # 【中文】- 训练/微调时可以通过命令行传入自定义的 .py（如 --modality_config_path），
        # 【中文】  在该 .py 顶层修改全局 `MODALITY_CONFIGS` 字典，加入/覆盖自己本体的定义；
        # 【中文】- 保存到 checkpoint 后，Processor 会带着这份配置一起保存；
        # 【中文】- 这里的 get_modality_configs() 就是从 Processor 中取回这份（可能包含你自定义本体的）模态配置。
        self.embodiment_tag = embodiment_tag
        self.modality_configs = self.processor.get_modality_configs()[self.embodiment_tag.value]
        self.collate_fn = self.processor.collator

        # Extract and validate language configuration
        # Currently only supports single language input per timestep
        language_keys = self.modality_configs["language"].modality_keys
        language_delta_indices = self.modality_configs["language"].delta_indices
        assert len(language_delta_indices) == 1, "Only one language delta index is supported"
        assert len(language_keys) == 1, "Only one language key is supported"
        self.language_key = language_keys[0]

    def _unbatch_observation(self, value: dict[str, Any]) -> list[dict[str, Any]]:
        """Unbatch a batched observation into a list of single observations.

        Args:
            value: Batched observation with shape (B, ...) for each modality

        Returns:
            List of B observations, each with the batch dimension removed

        【中文】用途与形状说明：
        - 输入 `value` 结构示例:
          - `value["video"][key]` ~ np.ndarray[(B, T_v, H, W, C)]
          - `value["state"][key]` ~ np.ndarray[(B, T_s, D)]
          - `value["language"][key]` ~ list[list[str]] 长度为 B
        - 输出为长度 B 的列表, 第 i 个元素形如:
          - `{"video": {key: (T_v,H,W,C)}, "state": {key: (T_s,D)}, "language": {key: [str]}}`
        - 主要用于 `_get_action` 内部，将一个 batch 拆成多个单样本，逐个喂给 Processor 做 VLA 预处理。
        """
        unbatched_obs = []
        # Infer batch size from the first video key
        batch_size = value["video"][list(value["video"].keys())[0]].shape[0]

        # Split each modality along the batch dimension
        for i in range(batch_size):
            unbatched_value = {
                "video": {k: v[i] for k, v in value["video"].items()},
                "state": {k: v[i] for k, v in value["state"].items()},
                "language": {k: v[i] for k, v in value["language"].items()},
            }
            unbatched_obs.append(unbatched_value)
        return unbatched_obs

    def _to_vla_step_data(self, observation: dict[str, Any]) -> VLAStepData:
        """Convert a single observation into a VLAStepData object for processing.

        Args:
            observation: Single observation dict with video, state, and language

        Returns:
            VLAStepData object ready for processor input

        【中文】输入/输出结构示例：
        - 输入 observation (单样本, 已去掉 batch 维度):
          - `observation["video"]`: dict[key -> np.ndarray[(T_v, H, W, C)]]
          - `observation["state"]`: dict[key -> np.ndarray[(T_s, D)]]
          - `observation["language"][self.language_key]`: list[str] (通常长度 1)
        - 输出 VLAStepData:
          - images: 同 observation["video"]
          - states: 同 observation["state"]
          - actions: 空 dict (推理阶段没有 GT 动作)
          - text: 取语言列表中的第一个字符串, 作为当前 step 的文本指令
          - embodiment: 当前策略的具身标签
        - 这个结构是下游 Processor 的统一输入格式, 便于在不同数据源之间复用处理逻辑。
        """
        return VLAStepData(
            images=observation["video"],
            states=observation["state"],
            actions={},  # No ground truth actions during inference
            text=observation["language"][self.language_key][0],
            embodiment=self.embodiment_tag,
        )

    def check_observation(self, observation: dict[str, Any]) -> None:
        """Validate that the observation has the correct structure and types.

        This method ensures that all required modalities are present and that their
        data types, shapes, and dimensions match the model's expectations.

        Expected observation structure:
            - video: dict[str, np.ndarray[np.uint8, (B, T, H, W, C)]]
                - B: batch size
                - T: temporal horizon (number of frames)
                - H, W: image height and width
                - C: number of channels (must be 3 for RGB)
            - state: dict[str, np.ndarray[np.float32, (B, T, D)]]
                - B: batch size
                - T: temporal horizon (number of state observations)
                - D: state dimension
            - language: dict[str, list[list[str]]]
                - Shape: (B, T) where each element is a string
                - T: temporal horizon (typically 1 for language)

        Args:
            observation: Dictionary containing video, state, and language modalities

        Raises:
            AssertionError: If any validation check fails

        【中文】结论先行：
        - 这是 `Gr00tPolicy` 对“嵌套三模态格式”的**强校验入口**，在开启 strict=True 时, 用于提前发现输入 shape / dtype / horizon 不符合配置的情况；
        - 若你在构造 observation 时遵循 `ModalityConfig` 中的 `delta_indices` 和 `modality_keys`, 通过这里的检查通常表示格式正确。
        """
        # Check that observation contains all required top-level modality keys
        for modality in ["video", "state", "language"]:
            assert modality in observation, f"Observation must contain a '{modality}' key"
            assert isinstance(observation[modality], dict), (
                f"Observation '{modality}' must be a dictionary. Got {type(observation[modality])}: {observation[modality]}"
            )

        # Track batch size across modalities to ensure consistency
        bs = -1

        # ===== VIDEO VALIDATION =====
        # Validate each video stream defined in the modality config
        for video_key in self.modality_configs["video"].modality_keys:
            # Set or verify batch size consistency across all video keys
            if bs == -1:
                bs = len(observation["video"][video_key])
            else:
                assert len(observation["video"][video_key]) == bs, (
                    f"Video key '{video_key}' must have batch size {bs}. Got {len(observation['video'][video_key])}"
                )

            # Check that the expected video key exists in the observation
            assert video_key in observation["video"], (
                f"Video key '{video_key}' must be in observation"
            )

            batched_video = observation["video"][video_key]

            # Verify data type is numpy array
            assert isinstance(batched_video, np.ndarray), (
                f"Video key '{video_key}' must be a numpy array. Got {type(batched_video)}"
            )

            # Verify dtype is uint8 (standard for image data, range 0-255)
            assert batched_video.dtype == np.uint8, (
                f"Video key '{video_key}' must be a numpy array of type np.uint8. Got {batched_video.dtype}"
            )

            # Verify shape has 5 dimensions: (B, T, H, W, C)
            assert batched_video.ndim == 5, (
                f"Video key '{video_key}' must be a numpy array of shape (B, T, H, W, C), got {batched_video.shape}"
            )

            # Verify temporal dimension matches the expected horizon from config
            assert batched_video.shape[1] == len(self.modality_configs["video"].delta_indices), (
                f"Video key '{video_key}'s horizon must be {len(self.modality_configs['video'].delta_indices)}. Got {batched_video.shape[1]}"
            )

            # Verify channel dimension is 3 (RGB images)
            assert batched_video.shape[-1] == 3, (
                f"Video key '{video_key}'s channel 'C' must be 3. Got {batched_video.shape[-1]}"
            )

        # ===== STATE VALIDATION =====
        # Validate each state stream defined in the modality config
        for state_key in self.modality_configs["state"].modality_keys:
            # Set or verify batch size consistency across all state keys
            if bs == -1:
                bs = len(observation["state"][state_key])
            else:
                assert len(observation["state"][state_key]) == bs, (
                    f"State key '{state_key}' must have batch size {bs}. Got {len(observation['state'][state_key])}"
                )

            # Check that the expected state key exists in the observation
            assert state_key in observation["state"], (
                f"State key '{state_key}' must be in observation"
            )

            batched_state = observation["state"][state_key]

            # Verify data type is numpy array
            assert isinstance(batched_state, np.ndarray), (
                f"State key '{state_key}' must be a numpy array. Got {type(batched_state)}"
            )

            # Verify dtype is float32 (standard for continuous state values)
            assert batched_state.dtype == np.float32, (
                f"State key '{state_key}' must be a numpy array of type np.float32. Got {batched_state.dtype}"
            )

            # Verify shape has 3 dimensions: (B, T, D)
            assert batched_state.ndim == 3, (
                f"State key '{state_key}' must be a numpy array of shape (B, T, D), got {batched_state.shape}"
            )

            # Verify temporal dimension matches the expected horizon from config
            assert batched_state.shape[1] == len(self.modality_configs["state"].delta_indices), (
                f"State key '{state_key}'s horizon must be {len(self.modality_configs['state'].delta_indices)}. Got {batched_state.shape[1]}"
            )

        # ===== LANGUAGE VALIDATION =====
        # Validate each language stream defined in the modality config
        for language_key in self.modality_configs["language"].modality_keys:
            # Set or verify batch size consistency (language uses len instead of .shape)
            if bs == -1:
                bs = len(observation["language"][language_key])
            else:
                assert len(observation["language"][language_key]) == bs, (
                    f"Language key '{language_key}' must have batch size {bs}. Got {len(observation['language'][language_key])}"
                )

            # Check that the expected language key exists in the observation
            assert language_key in observation["language"], (
                f"Language key '{language_key}' must be in observation"
            )

            batched_language: list[list[str]] = observation["language"][language_key]

            # Verify outer structure is a list (batch dimension)
            assert isinstance(batched_language, list), (
                f"Language key '{language_key}' must be a list. Got {type(batched_language)}"
            )

            # Validate each batch item
            for batch_item in batched_language:
                # Verify temporal dimension matches expected horizon
                assert len(batch_item) == len(self.modality_configs["language"].delta_indices), (
                    f"Language key '{language_key}'s horizon must be {len(self.modality_configs['language'].delta_indices)}. Got {len(batched_language)}"
                )

                # Verify inner structure is also a list (temporal dimension)
                assert isinstance(batch_item, list), (
                    f"Language batch item must be a list. Got {type(batch_item)}"
                )

                # Current implementation expects exactly one language instruction per timestep
                assert len(batch_item) == 1, (
                    f"Language batch item must have exactly one item. Got {len(batch_item)}"
                )

                # Verify the instruction itself is a string
                assert isinstance(batch_item[0], str), (
                    f"Language batch item must be a string. Got {type(batch_item[0])}"
                )

    # 核心推理方法
    def _get_action(
        self, observation: dict[str, Any], options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Internal method to compute actions from observations.

        Pipeline:
        1. Unbatch observations into individual samples
        2. Convert each to VLAStepData and process
        3. Collate into model input batch
        4. Run model inference
        5. Decode and unnormalize actions

        Args:
            observation: Batched observation dictionary
            options: Optional parameters (currently unused)

        Returns:
            Tuple of (actions_dict, info_dict)

        【中文】数据流简述（带形状示例）：
        - 输入 observation:
          - video/state/language 已是嵌套结构, 形状大致为:
            - video[key]: (B, T_v, H, W, C)
            - state[key]: (B, T_s, D)
            - language[key]: list[list[str]] 长度 B, 每个元素长度为 T_lang
        - 内部步骤:
          1) `_unbatch_observation` 将 batch 拆成单样本列表;
          2) 每个样本构成 `VLAStepData`, 交给 Processor 做预处理与特征提取;
          3) 使用 collator 将多个样本重新打包为模型的输入 batch (通常是张量字典);
          4) 调用 `self.model.get_action(**collated_inputs)` 得到规范化动作 `normalized_action`;
          5) 通过 `self.processor.decode_action` 结合原始 state 做反归一化 / 相对→绝对转换, 得到物理动作。
        - 输出 `actions_dict`:
          - 形如 {action_key: np.ndarray[(B, T_action, D_key)]}, 与 `modality_configs["action"].modality_keys` 对齐。
        """

        # Step 1: Split batched observation into individual observations
        # 第一步：将 batch 形式的观测拆分为单个样本列表
        unbatched_observations = self._unbatch_observation(observation)
        processed_inputs = []

        # Step 2: Process each observation through the VLA processor
        # 第二步：将每个样本转换为 VLA 步数据并交由 processor 处理
        states = []
        for obs in unbatched_observations:
            vla_step_data = self._to_vla_step_data(obs)
            states.append(vla_step_data.states)  # dict[str, np.ndarray[np.float32, (T, D)]]
            messages = [{"type": MessageType.EPISODE_STEP.value, "content": vla_step_data}]
            processed_inputs.append(self.processor(messages)) # processor接受 VLAStepData
        '''
        Processor输入带名字的state,action,做归一化、相对角度等，最后输出拼接好的state,action tensor
        输入：messages = [{
            "content": {
                "embodiment": EmbodimentTag,
                "states": {"joint_pos": [...], "gripper": [...]},
                "actions": {"joint_pos": [...], "gripper": [...]},
                "images": {"wrist_cam": [PIL.Image, ...], "front_cam": [...]},
                "text": "pick up the red cube"
            }
        }]
        
        输出：{
            "state": Tensor[T, max_state_dim],
            "action": Tensor[max_action_horizon, max_action_dim],
            "action_mask": Tensor[max_action_horizon, max_action_dim],
            "vlm_content": {"text": str, "images": [PIL.Image], "conversation": [...]},
            "embodiment_id": int
        }
        '''


        # Step 3: Collate processed inputs into a single batch for model
        # 第三步：将处理后的样本重新打包为模型的输入 batch，并转为 bfloat16 精度
        # 示例：collated_inputs = {"state": torch.Tensor(B, T, D), "images": torch.Tensor(B, T, C, H, W), ...}
        collated_inputs = self.collate_fn(processed_inputs)
        collated_inputs = _rec_to_dtype(collated_inputs, dtype=torch.bfloat16)

        # Step 4: Run model inference to predict actions
        # 第四步：模型推理，预测动作（归一化空间）
        # 示例：model_pred["action_pred"] 的形状通常为 (B, T_action, D_action)
        with torch.inference_mode():
            '''
            def get_action(self, inputs: dict) -> BatchFeature:
                """
                推理接口：生成动作序列。
                返回 action_pred: [B, action_horizon, action_dim]。
                
                Args:
                    inputs: 与训练时类似的输入字典，但通常不包含 "action"，例如：
                        {
                            "vlm_content": {...},
                            "state": torch.Tensor[B, state_dim],
                            "embodiment_id": torch.Tensor[B],
                            ...
                        }
                
                Returns:
                    BatchFeature: 输出示例：
                        {
                            "action_pred": torch.Tensor[B, action_horizon, action_dim],
                            "backbone_features": torch.Tensor[B, vl_seq_len, D],
                            "state_features": torch.Tensor[B, T_state, D],
                        }
                """
            '''
            model_pred = self.model.get_action(**collated_inputs)
        normalized_action = model_pred["action_pred"].float() # 推理返回的是归一化空间内、相对的action数值

        # Step 5: Decode actions from normalized space back to physical units
        # 第五步：将归一化动作解码并反归一化为物理单位
        # 1，相对action还原绝对action，依赖当前的state，所以这里把state.xxx每个单独stack成batch
        batched_states = {}
        for k in self.modality_configs["state"].modality_keys:
            batched_states[k] = np.stack([s[k] for s in states], axis=0)  # (B, T, D) 
        unnormalized_action = self.processor.decode_action( # 这里会把action分离到action.xxx，然后与state.xxx计算还原到绝对action.xxx
            normalized_action.cpu().numpy(), self.embodiment_tag, batched_states
        )

        # Cast all actions to float32 for consistency
        # 为了输出一致性，将所有动作数据类型转换为 float32
        casted_action = {
            key: value.astype(np.float32) for key, value in unnormalized_action.items()
        }
        return casted_action, {}

    def check_action(self, action: dict[str, Any]) -> None:
        """Validate that the action has the correct structure and types.

        This method ensures that all required action keys are present and that their
        data types, shapes, and dimensions match the model's action space.

        Expected action structure:
            - action: dict[str, np.ndarray[np.float32, (B, T, D)]]
                - B: batch size
                - T: action horizon (number of future action steps)
                - D: action dimension (e.g., joint positions, velocities, gripper state)

        Args:
            action: Dictionary containing action arrays for each action key

        Raises:
            AssertionError: If any validation check fails

        【中文】使用建议：
        - 在你自定义下游控制逻辑或对模型输出做后处理前, 可先调用 `check_action` 确认 shape / dtype / horizon 与配置一致；
        - 若遇到 `AssertionError`, 通常说明: 要么 `modality_configs["action"].delta_indices` 与数据不配, 要么你在中途改动了动作数组的形状。
        """
        # Validate each action key defined in the modality config
        for action_key in self.modality_configs["action"].modality_keys:
            # Check that the expected action key exists
            assert action_key in action, f"Action key '{action_key}' must be in action"

            action_arr = action[action_key]

            # Verify data type is numpy array
            assert isinstance(action_arr, np.ndarray), (
                f"Action key '{action_key}' must be a numpy array. Got {type(action_arr)}"
            )

            # Verify dtype is float32 (standard for continuous actions)
            assert action_arr.dtype == np.float32, (
                f"Action key '{action_key}' must be a numpy array of type np.float32. Got {action_arr.dtype}"
            )

            # Verify shape has 3 dimensions: (B, T, D)
            assert action_arr.ndim == 3, (
                f"Action key '{action_key}' must be a numpy array of shape (B, T, D), got {action_arr.shape}"
            )

            # Verify action horizon matches the expected temporal dimension from config
            assert action_arr.shape[1] == len(self.modality_configs["action"].delta_indices), (
                f"Action key '{action_key}'s horizon must be {len(self.modality_configs['action'].delta_indices)}. Got {action_arr.shape[1]}"
            )

    def get_modality_config(self) -> dict[str, ModalityConfig]:
        return self.modality_configs

    def reset(self, options: dict[str, Any] | None = None) -> dict[str, Any]:
        """Reset the policy to its initial state.

        Args:
            options: Dictionary containing the options for the reset

        Returns:
            Dictionary containing the info after resetting the policy
        """
        return {}


class Gr00tSimPolicyWrapper(PolicyWrapper):
    """Wrapper for Gr00tPolicy to enable compatibility with existing Gr00t simulation environments.

    This wrapper is specifically designed for retro-fitting the Gr00t policy with the current
    Gr00t simulation environment interface. It handles the transformation between the flat
    observation format used by Gr00t sim environments (with keys like 'video.camera_name',
    'state.joint_positions') and the nested format expected by Gr00tPolicy.

    **Important**: If you are using other environments, custom robots, or building new environments,
    you should use `Gr00tPolicy` directly and format your observations according to its interface.
    This wrapper is only needed for compatibility with the existing Gr00t sim infrastructure.

    Key transformations performed by this wrapper:
    - Observation keys: 'video.cam' -> observation['video']['cam']
    - Observation keys: 'state.joints' -> observation['state']['joints']
    - Language keys: 'task' or 'annotation.human.coarse_action' -> observation['language']['task']
    - Action keys: action['joints'] -> 'action.joints'

    【中文】类作用总结：
    - 适配 **老版 Gr00t 仿真环境** 的扁平观测/动作格式, 让其可以复用 `Gr00tPolicy` 的推理逻辑；
    - 对你自己写的新环境/机器人, 建议直接使用 `Gr00tPolicy`, 不需要本 wrapper, 只要按嵌套三模态格式组织输入即可。
    """

    def __init__(self, policy: Gr00tPolicy, *, strict: bool = True):
        """Initialize the wrapper around a Gr00tPolicy instance.

        Args:
            policy: The Gr00tPolicy instance to wrap
            strict: Whether to enforce strict validation (default: True)
        """
        super().__init__(policy, strict=strict)
        self.policy: Gr00tPolicy = policy
        assert len(self.policy.modality_configs["language"].delta_indices) == 1, (
            "Only one language delta index is supported"
        )

    def check_observation(self, observation: dict[str, Any]) -> None:
        """Validate observation from Gr00t sim environment format.

        This validation is specific to the flat observation format used by Gr00t sim environments.
        Unlike Gr00tPolicy.check_observation which expects nested dicts, this expects flat keys.

        Expected observation structure (Gr00t sim format):
            - Flat keys like 'video.camera_name': np.ndarray[np.uint8, (B, T, H, W, C)]
            - Flat keys like 'state.state_name': np.ndarray[np.float32, (B, T, D)]
            - Language keys: tuple[str] or list[str] with shape (B,)
                - Key can be 'task' or 'annotation.human.coarse_action' (for DC envs)

        Args:
            observation: Flat observation dictionary from Gr00t sim environment

        Raises:
            AssertionError: If any validation check fails

        【中文】注意：
        - 这里只检查“扁平格式”的 observation 是否满足旧版仿真环境约定；
        - 通过后 wrapper 会自动把它转换成嵌套三模态格式，再交给内部的 `Gr00tPolicy` 做真正的推理。
        """
        modality_configs = self.get_modality_config()

        # ===== VIDEO VALIDATION =====
        # Check video modalities with flat key format: 'video.camera_name'
        for video_key in modality_configs["video"].modality_keys:
            # Construct flat key expected in Gr00t sim environment
            parsed_key = f"video.{video_key}"
            assert parsed_key in observation, f"Video key '{parsed_key}' must be in observation"

            batched_video = observation[parsed_key]

            # Verify data type is numpy array
            assert isinstance(batched_video, np.ndarray), (
                f"Video key '{video_key}' must be a numpy array. Got {type(batched_video)}"
            )

            # Verify dtype is uint8 (standard for image data, range 0-255)
            assert batched_video.dtype == np.uint8, (
                f"Video key '{video_key}' must be a numpy array of type np.uint8. Got {batched_video.dtype}"
            )

            # Verify shape has 5 dimensions: (B, T, H, W, C)
            assert batched_video.ndim == 5, (
                f"Video key '{video_key}' must be a numpy array of shape (B, T, H, W, C), got {batched_video.shape}"
            )

            # Verify temporal dimension matches the expected horizon from config
            assert batched_video.shape[1] == len(modality_configs["video"].delta_indices), (
                f"Video key '{video_key}'s horizon must be {len(modality_configs['video'].delta_indices)}. Got {batched_video.shape[1]}"
            )

            # Verify channel dimension is 3 (RGB images)
            assert batched_video.shape[-1] == 3, (
                f"Video key '{video_key}'s channel 'C' must be 3. Got {batched_video.shape[-1]}"
            )

        # ===== STATE VALIDATION =====
        # Check state modalities with flat key format: 'state.state_name'
        for state_key in modality_configs["state"].modality_keys:
            # Construct flat key expected in Gr00t sim environment
            parsed_key = f"state.{state_key}"
            assert parsed_key in observation, f"State key '{parsed_key}' must be in observation"

            batched_state = observation[parsed_key]

            # Verify data type is numpy array
            assert isinstance(batched_state, np.ndarray), (
                f"State key '{state_key}' must be a numpy array. Got {type(batched_state)}"
            )

            # Verify dtype is float32 (standard for continuous state values)
            assert batched_state.dtype == np.float32, (
                f"State key '{state_key}' must be a numpy array of type np.float32. Got {batched_state.dtype}"
            )

            # Verify shape has 3 dimensions: (B, T, D)
            assert batched_state.ndim == 3, (
                f"State key '{state_key}' must be a numpy array of shape (B, T, D), got {batched_state.shape}"
            )

            # Verify temporal dimension matches the expected horizon from config
            assert batched_state.shape[1] == len(modality_configs["state"].delta_indices), (
                f"State key '{state_key}'s horizon must be {len(modality_configs['state'].delta_indices)}. Got {batched_state.shape[1]}"
            )

        # ===== LANGUAGE VALIDATION =====
        # Check language modalities (special handling for DC environment compatibility)
        for language_key in modality_configs["language"].modality_keys:
            # PATCH: Legacy compatibility for DC environments
            # DC envs use 'annotation.human.coarse_action' instead of 'task'
            if language_key == "task" and "annotation.human.coarse_action" in observation:
                language_key = "annotation.human.coarse_action"
            # /PATCH

            # Check that the expected language key exists
            assert language_key in observation, (
                f"Language key '{language_key}' must be in observation"
            )

            # In Gr00t sim format, language is a tuple of strings (B,)
            batched_language: tuple[str] | list[str] = observation[language_key]  # (B,)

            # Verify outer structure is a tuple (batch dimension)
            assert isinstance(batched_language, (tuple, list)), (
                f"Language key '{language_key}' must be a tuple or list. Got {type(batched_language)}"
            )

            # Verify each batch item is a string
            assert isinstance(batched_language[0], str), (
                f"Language batch item must be a string. Got {type(batched_language[0])}"
            )

    def _get_action(
        self, observation: dict[str, Any], options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Transform Gr00t sim observation format and compute actions.

        This method transforms the flat observation format from Gr00t sim environments
        into the nested format expected by Gr00tPolicy, computes actions, and transforms
        them back to the flat format expected by Gr00t sim environments.

        Input format (Gr00t sim):
            - Flat keys: 'video.camera_name', 'state.state_name'
            - Language: tuple[str] (B,)

        Output format (Gr00t sim):
            - Flat keys: 'action.action_name'

        Args:
            observation: Flat observation dictionary from Gr00t sim environment
            options: Optional parameters (currently unused)

        Returns:
            Tuple of (flat_actions_dict, info_dict)

        【中文】数据流简述：
        - 输入: 扁平观测字典 (仿真环境原始格式);
        - 中间: 按 `modality_configs` 把扁平 key 拆成 video/state/language 三模态嵌套结构, 调用内部 `Gr00tPolicy.get_action`；
        - 输出: 再次扁平化为 {"action.xxx": np.ndarray[(B, T, D)]} 的形式, 以便旧环境无缝接入。
        """
        # Transform flat observation format to nested format expected by Gr00tPolicy
        new_obs = {}
        for modality in ["video", "state", "language"]:
            new_obs[modality] = {}
            for key in self.policy.modality_configs[modality].modality_keys:
                if modality == "language":
                    # PATCH: Legacy compatibility for DC environments
                    if key == "task" and "annotation.human.coarse_action" in observation:
                        parsed_key = "annotation.human.coarse_action"
                    # /PATCH
                    else:
                        parsed_key = key
                else:
                    # Construct flat key (e.g., 'video.camera' or 'state.joints')
                    parsed_key = f"{modality}.{key}"

                arr = observation[parsed_key]

                # Transform to nested format
                if modality == "language":
                    # Convert from tuple[str] or list[str] (B,) to list[list[str]] (B, 1)
                    # Each element becomes a list with one string for temporal dimension
                    new_obs[modality][key] = [[str(item)] for item in arr]
                else:
                    # Video and state arrays are already in correct format (B, T, ...)
                    new_obs[modality][key] = arr

        # Compute actions using the underlying Gr00tPolicy
        action, info = self.policy.get_action(new_obs, options)

        # Transform actions back to flat format for Gr00t sim environment
        # action['joints'] -> 'action.joints'
        return {f"action.{key}": action[key] for key in action}, info

    def check_action(self, action: dict[str, Any]) -> None:
        """Validate action in Gr00t sim environment format.

        This validation is specific to the flat action format used by Gr00t sim environments.
        Unlike Gr00tPolicy.check_action which expects nested dicts, this expects flat keys.

        Expected action structure (Gr00t sim format):
            - Flat keys like 'action.action_name': np.ndarray[np.float32, (B, T, D)]
                - B: batch size
                - T: action horizon (number of future action steps)
                - D: action dimension

        Args:
            action: Flat action dictionary for Gr00t sim environment

        Raises:
            AssertionError: If any validation check fails

        【中文】补充说明：
        - 这是旧仿真环境下的 action 检查, 观看 shape 是否满足 (B, T, D) 以及 horizon 是否等于 `delta_indices` 长度；
        - 若你在环境内部对动作做裁剪/拼接, 建议在返回前调用一次 `check_action` 确保没有破坏形状契约。
        """
        modality_configs = self.get_modality_config()

        # Validate each action key defined in the modality config
        for action_key in modality_configs["action"].modality_keys:
            # Construct flat key expected in Gr00t sim environment (e.g., 'action.joints')
            parsed_key = f"action.{action_key}"
            assert parsed_key in action, f"Action key '{parsed_key}' must be in action"

            action_arr = action[parsed_key]

            # Verify data type is numpy array
            assert isinstance(action_arr, np.ndarray), (
                f"Action key '{action_key}' must be a numpy array. Got {type(action_arr)}"
            )

            # Verify dtype is float32 (standard for continuous actions)
            assert action_arr.dtype == np.float32, (
                f"Action key '{action_key}' must be a numpy array of type np.float32. Got {action_arr.dtype}"
            )

            # Verify shape has 3 dimensions: (B, T, D)
            assert action_arr.ndim == 3, (
                f"Action key '{action_key}' must be a numpy array of shape (B, T, D), got {action_arr.shape}"
            )

            # Verify action horizon matches the expected temporal dimension from config
            assert action_arr.shape[1] == len(modality_configs["action"].delta_indices), (
                f"Action key '{action_key}'s horizon must be {len(modality_configs['action'].delta_indices)}. Got {action_arr.shape[1]}"
            )

    def get_modality_config(self) -> dict[str, ModalityConfig]:
        """Get the modality configuration from the underlying policy.

        Returns:
            Dictionary mapping modality names to their configurations
        """
        return self.policy.get_modality_config()
