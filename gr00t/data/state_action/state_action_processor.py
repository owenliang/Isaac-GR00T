"""
Unified processor for robot state and action data.

Handles:
- State normalization (min/max, mean/std, sin/cos encoding)
- Action normalization
- Absolute <-> Relative action representation conversion
- Action processing with state dependency

【中文】统一的机器人状态与动作处理模块。
【中文】主要职责：
【中文】- 对不同具身形态的关节状态做归一化（min/max、mean/std、或 sin/cos 编码）；
【中文】- 对动作做归一化，并在需要时在“绝对动作”和“相对动作”表示之间互相转换；
【中文】- 在相对动作表示下，利用当前状态（state）作为参考系进行转换；
【中文】- 提供便捷接口，将状态/动作的一整套处理逻辑封装为 `apply` / `unapply` 调用，供 Processor 直接使用。
"""

from copy import deepcopy

from gr00t.configs.data.embodiment_configs import (
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)
from gr00t.data.state_action.action_chunking import EndEffectorActionChunk, JointActionChunk
from gr00t.data.state_action.pose import EndEffectorPose, JointPose
from gr00t.data.utils import (
    apply_sin_cos_encoding,
    nested_dict_to_numpy,
    normalize_values_meanstd,
    normalize_values_minmax,
    parse_modality_configs,
    unnormalize_values_meanstd,
    unnormalize_values_minmax,
)
import numpy as np


class StateActionProcessor:
    """
    Unified processor for robot state and action data.

    Handles:
    - State normalization (min/max, mean/std, sin/cos encoding)
    - Action normalization
    - Absolute <-> Relative action representation conversion
    - Action processing with state dependency
    """

    def __init__(
        self,
        modality_configs: dict[str, dict[str, ModalityConfig]],
        statistics: dict[str, dict[str, dict[str, dict[str, list[float]]]]] | None = None,
        use_percentiles: bool = False,
        clip_outliers: bool = True,
        apply_sincos_state_encoding: bool = False,
        use_relative_action: bool = False,
    ):
        """
        Initialize unified state and action processor.

        Args:
            modality_configs: Nested dict with structure:
                {embodiment_tag: {modality: ModalityConfig}}
                where modality in ["state", "action"]
                Example: {"gr1": {"state": ModalityConfig(...), "action": ModalityConfig(...)}}
            statistics: Optional nested dict with structure:
                {embodiment_tag: {modality: {joint_group: {stat_type: values}}}}
                where modality in ["state", "action", "relative_action"]
                and stat_type in ["min", "max", "mean", "std", "q01", "q99"]
                Example: {"gr1": {"state": {"left_arm": {"min": [...], "max": [...], ...}}}}
            use_percentiles: Whether to use percentiles (q01/q99) instead of min/max
            clip_outliers: Whether to clip normalized values to [-1, 1]
            apply_sincos_state_encoding: Global flag to enable sin/cos encoding for states

        【中文】初始化统一的状态/动作处理器。
        【中文】- modality_configs：按具身形态组织的模态配置，用来约定有哪些关节组、哪些模态（state/action）；
        【中文】- statistics：来自数据集统计脚本（stats.json/relative_stats.json）的统计量，用于归一化；
        【中文】- use_percentiles：是否使用百分位（q01/q99）代替 min/max，提高鲁棒性；
        【中文】- clip_outliers：是否对归一化后的值裁剪到 [-1, 1]；
        【中文】- apply_sincos_state_encoding：是否对部分状态用 sin/cos 编码（角度型量）；
        【中文】- use_relative_action：是否启用相对动作表示（例如以当前末端位姿为参考）。
        """
        self.modality_configs = parse_modality_configs(modality_configs)
        self.statistics: dict[str, dict[str, dict[str, dict[str, list[float]]]]] = {}
        self.use_percentiles = use_percentiles
        self.clip_outliers = clip_outliers
        self.apply_sincos_state_encoding = apply_sincos_state_encoding
        self.use_relative_action = use_relative_action

        # Normalization parameters computed from statistics
        self.norm_params: dict[str, dict[str, dict[str, dict[str, np.ndarray]]]] = {}
        # Format: norm_params[embodiment_tag][modality][joint_group][stat_type]
        # where stat_type in ["min", "max", "mean", "std", "dim"]

        if statistics is not None:
            self.set_statistics(statistics)

        self.train()

    def train(self):
        self.training = True

    def eval(self):
        self.training = False

    def set_statistics(
        self,
        statistics: dict[str, dict[str, dict[str, dict[str, list[float]]]]],
        override: bool = False,
    ) -> None:
        """
        Set dataset statistics for normalization.

        Args:
            statistics: Nested dict with structure:
                {embodiment_tag: {modality: {joint_group: {stat_type: values}}}}

        【中文】设置用于归一化的“数据集统计信息”。
        【中文】- 统计结构按 `embodiment_tag → modality(state/action/relative_action) → joint_group` 组织；
        【中文】- 每个 joint_group 下包含 min/max/mean/std/q01/q99 等标量数组；
        【中文】- 若 override=False，则已存在的具身不会被覆盖（方便增量添加新具身）。
        【中文】本函数会在内部调用 `_compute_normalization_parameters`，把原始统计量转换为便于数值计算的 numpy 形式并缓存到 `norm_params`。
        """
        for key in statistics:
            if key not in self.statistics or override:
                self.statistics[key] = deepcopy(statistics[key])
            else:
                print(f"Embodiment tag {key} already in statistics, skipping updating")
        self._compute_normalization_parameters()

    def _compute_normalization_parameters(self) -> None:
        """Compute and cache normalization parameters from statistics for all embodiments and modalities.

        【中文】根据当前 `self.statistics` 计算并缓存归一化参数 `norm_params`：
        【中文】- 对每个具身、每个模态（state/action）、每个 joint_group：
        【中文】  - 选择 min/max 或 q01/q99 作为归一化区间；
        【中文】  - 记录 mean/std 用于均值方差归一化；
        【中文】  - 记录 dim（维度）方便后续拼接和推断整体维度；
        【中文】- 若启用了相对动作，并在 `statistics[embodiment]["relative_action"]` 中提供了相对动作统计，会用其覆盖对应动作组的 absolute 统计。
        """
        for embodiment_tag in self.statistics:
            self.norm_params[embodiment_tag] = {}

            for modality in ["state", "action"]:
                if modality not in self.statistics[embodiment_tag]:
                    continue

                self.norm_params[embodiment_tag][modality] = {}

                for joint_group, stats in self.statistics[embodiment_tag][modality].items():
                    if self.use_percentiles:
                        min_vals = np.array(stats["q01"])
                        max_vals = np.array(stats["q99"])
                    else:
                        min_vals = np.array(stats["min"])
                        max_vals = np.array(stats["max"])

                    mean_vals = np.array(stats["mean"])
                    std_vals = np.array(stats["std"])

                    # Compute range, ensuring it's not zero
                    range_vals = max_vals - min_vals
                    range_vals = np.maximum(range_vals, 1e-8)

                    self.norm_params[embodiment_tag][modality][joint_group] = {
                        "min": min_vals,
                        "max": max_vals,
                        "dim": np.array(range_vals.shape[0]),
                        "mean": mean_vals,
                        "std": std_vals,
                    }

            # Override absolute action stats with relative stats where specified
            if "action" in self.modality_configs[embodiment_tag]:
                modality_keys = self.modality_configs[embodiment_tag]["action"].modality_keys
                action_configs = self.modality_configs[embodiment_tag]["action"].action_configs

                if action_configs is not None:
                    for key, action_config in zip(modality_keys, action_configs):
                        if (
                            action_config.rep == ActionRepresentation.RELATIVE
                            and self.use_relative_action
                        ):
                            if "relative_action" not in self.statistics[embodiment_tag]:
                                raise ValueError(
                                    f"Relative action statistics required for embodiment '{embodiment_tag}' "
                                    f"but 'relative_action' not found in statistics"
                                )
                            if key not in self.statistics[embodiment_tag]["relative_action"]:
                                raise ValueError(
                                    f"Relative action statistics required for key '{key}' "
                                    f"in embodiment '{embodiment_tag}' but not found"
                                )
                            action_dim = self.norm_params[embodiment_tag]["action"][key]["dim"]
                            self.norm_params[embodiment_tag]["action"][key] = nested_dict_to_numpy(
                                self.statistics[embodiment_tag]["relative_action"][key]
                            )
                            self.norm_params[embodiment_tag]["action"][key]["dim"] = action_dim

    def apply_state(
        self,
        state: dict[str, np.ndarray],
        embodiment_tag: str,
    ) -> dict[str, np.ndarray]:
        """
        Apply state processing (normalization, encoding).

        Args:
            state: Dict mapping joint_group -> raw state values
                Shape per group: (..., D) where D is state dimension
            embodiment_tag: Embodiment identifier (e.g., "gr1")

        Returns:
            Dict mapping joint_group -> processed state values
                - Sin/cos encoded groups: (..., 2*D)
                - Other groups: (..., D)

        【中文】对“原始状态”做归一化/编码处理：
        【中文】- 按具身和 joint_group 查找对应统计量与配置；
        【中文】- 若该关节组配置了 sin/cos 编码，则直接对角度向量做三角编码（维度翻倍）；
        【中文】- 若配置了 mean/std 归一化，则以均值方差方式缩放；
        【中文】- 否则使用 min/max 映射到 [-1, 1]，并在 `clip_outliers=True` 时做裁剪；
        【中文】最终返回的字典结构与输入的 state keys 一致，但每个 value 已经是“模型空间”的归一化值。
        """
        normalized_values = {}
        state = deepcopy(state)  # Avoid modifying input

        # Get sin/cos embedding keys if enabled
        sin_cos_keys = None
        if self.apply_sincos_state_encoding:
            state_config = self.modality_configs[embodiment_tag].get("state")
            if state_config and hasattr(state_config, "sin_cos_embedding_keys"):
                sin_cos_keys = state_config.sin_cos_embedding_keys

        for joint_group in self.modality_configs[embodiment_tag]["state"].modality_keys:
            if joint_group not in state:
                raise KeyError(
                    f"Joint group '{joint_group}' not found in state dict for embodiment '{embodiment_tag}'"
                )

            # Strategy 1: Sin/cos encoding (doubles dimension)
            if sin_cos_keys and joint_group in sin_cos_keys:
                normalized_values[joint_group] = apply_sin_cos_encoding(state[joint_group])

            # Strategy 2: Mean/std normalization
            elif (
                hasattr(self.modality_configs[embodiment_tag]["state"], "mean_std_embedding_keys")
                and self.modality_configs[embodiment_tag]["state"].mean_std_embedding_keys
                and joint_group
                in self.modality_configs[embodiment_tag]["state"].mean_std_embedding_keys
            ):
                params = self.norm_params[embodiment_tag]["state"][joint_group]
                normalized = normalize_values_meanstd(state[joint_group], params)
                normalized_values[joint_group] = normalized

            # Strategy 3: Min/max normalization to [-1, 1]
            else:
                params = self.norm_params[embodiment_tag]["state"][joint_group]
                normalized = normalize_values_minmax(state[joint_group], params)

                if self.clip_outliers:
                    normalized = np.clip(normalized, -1.0, 1.0)

                normalized_values[joint_group] = normalized

        return normalized_values

    def unapply_state(
        self,
        state: dict[str, np.ndarray],
        embodiment_tag: str,
    ) -> dict[str, np.ndarray]:
        """
        Reverse state processing (denormalization).

        Args:
            state: Dict mapping joint_group -> processed state values
            embodiment_tag: Embodiment identifier

        Returns:
            Dict mapping joint_group -> raw state values

        Raises:
            ValueError: If attempting to reverse sin/cos encoding (not reversible)
        """
        unnormalized_values = {}

        # Get sin/cos embedding keys if enabled
        sin_cos_keys = None
        if self.apply_sincos_state_encoding:
            state_config = self.modality_configs[embodiment_tag].get("state")
            if state_config and hasattr(state_config, "sin_cos_embedding_keys"):
                sin_cos_keys = state_config.sin_cos_embedding_keys

        for joint_group in self.modality_configs[embodiment_tag]["state"].modality_keys:
            if joint_group not in state:
                raise KeyError(
                    f"Joint group '{joint_group}' not found in state dict for embodiment '{embodiment_tag}'"
                )

            # Sin/cos encoding is not reversible
            if sin_cos_keys and joint_group in sin_cos_keys:
                raise ValueError(
                    f"Cannot unapply sin/cos encoding for joint group '{joint_group}' "
                    f"in embodiment '{embodiment_tag}'. This transformation is not reversible."
                )

            # Reverse mean/std normalization
            elif (
                hasattr(self.modality_configs[embodiment_tag]["state"], "mean_std_embedding_keys")
                and self.modality_configs[embodiment_tag]["state"].mean_std_embedding_keys
                and joint_group
                in self.modality_configs[embodiment_tag]["state"].mean_std_embedding_keys
            ):
                params = self.norm_params[embodiment_tag]["state"][joint_group]
                unnormalized = unnormalize_values_meanstd(state[joint_group], params)
                unnormalized_values[joint_group] = unnormalized

            # Reverse min/max normalization
            else:
                params = self.norm_params[embodiment_tag]["state"][joint_group]
                unnormalized_values[joint_group] = unnormalize_values_minmax(
                    state[joint_group], params
                )

        return unnormalized_values

    def apply_action(
        self,
        action: dict[str, np.ndarray],
        embodiment_tag: str,
        state: dict[str, np.ndarray] | None = None,
    ) -> dict[str, np.ndarray]:
        """
        Apply action processing (absolute->relative conversion, normalization).

        Processing order:
        1. Convert absolute actions to relative (if configured)
        2. Normalize actions

        Args:
            action: Dict mapping joint_group -> raw action values
                Shape per group: (T, D) where T is action horizon, D is action dimension
            embodiment_tag: Embodiment identifier
            state: Optional dict mapping joint_group -> raw state values
                Required if any action group uses ActionRepresentation.RELATIVE
                Shape per group: (T_state, D) where last timestep is used as reference

        Returns:
            Dict mapping joint_group -> processed action values
                Shape per group: (T, D)

        Raises:
            ValueError: If state is None but required for relative action conversion

        【中文】对“原始动作序列”做两步处理：
        【中文】1. 若配置为相对动作（RELATIVE）且启用 use_relative_action，则以给定 state 末步为参考系，将绝对动作转换为相对动作；
        【中文】2. 按统计量做归一化（mean/std 或 min/max），并在需要时裁剪到 [-1, 1]；
        【中文】返回的字典与输入 action 的关节组 key 保持一致，但数值已经是归一化后的表示。
        """
        action = deepcopy(action)  # Avoid modifying input

        # Step 1: Convert absolute actions to relative (if needed)
        modality_keys = self.modality_configs[embodiment_tag]["action"].modality_keys
        action_configs = self.modality_configs[embodiment_tag]["action"].action_configs

        if action_configs is not None:
            for key, action_config in zip(modality_keys, action_configs):
                if action_config.rep == ActionRepresentation.RELATIVE and self.use_relative_action:
                    if state is None:
                        raise ValueError(
                            f"State dict required for relative action processing of key '{key}' "
                            f"in embodiment '{embodiment_tag}'"
                        )

                    # Determine which state key to use as reference
                    state_key = action_config.state_key if action_config.state_key else key

                    if state_key not in state:
                        raise KeyError(
                            f"Reference state key '{state_key}' not found in state dict "
                            f"for embodiment '{embodiment_tag}'"
                        )

                    # Use last state as reference frame
                    reference_state = state[state_key][-1]

                    # Convert absolute to relative
                    action[key] = self._convert_to_relative_action(
                        action=action[key],
                        reference_state=reference_state,
                        action_type=action_config.type,
                        action_format=action_config.format,
                    )

        # Step 2: Normalize actions
        normalized_values = {}
        for joint_group in modality_keys:
            if joint_group not in action:
                raise KeyError(
                    f"Joint group '{joint_group}' not found in action dict for embodiment '{embodiment_tag}'"
                )

            params = self.norm_params[embodiment_tag]["action"][joint_group]
            if (
                self.modality_configs[embodiment_tag]["action"].mean_std_embedding_keys is not None
                and joint_group
                in self.modality_configs[embodiment_tag]["action"].mean_std_embedding_keys
            ):
                normalized = normalize_values_meanstd(action[joint_group], params)
            else:
                normalized = normalize_values_minmax(action[joint_group], params)

            if self.clip_outliers:
                normalized = np.clip(normalized, -1.0, 1.0)

            normalized_values[joint_group] = normalized

        return normalized_values

    def unapply_action(
        self,
        action: dict[str, np.ndarray],
        embodiment_tag: str,
        state: dict[str, np.ndarray] | None = None,
    ) -> dict[str, np.ndarray]:
        """
        Reverse action processing (denormalization, relative->absolute conversion).

        Processing order:
        1. Denormalize actions
        2. Convert relative actions to absolute (if configured)

        Args:
            action: Dict mapping joint_group -> processed action values
                Shape per group: (T, D) or (B, T, D) for batched
            embodiment_tag: Embodiment identifier
            state: Optional dict mapping joint_group -> raw state values
                Required if any action group uses ActionRepresentation.RELATIVE
                Shape per group: (T_state, D) or (B, T_state, D) for batched

        Returns:
            Dict mapping joint_group -> raw absolute action values
                Shape per group: (T, D) or (B, T, D) for batched

        Raises:
            ValueError: If state is None but required for relative->absolute conversion

        【中文】撤销对动作的处理：
        【中文】- 第一步：将归一化后的动作反归一化回原始数值尺度；
        【中文】- 第二步：若配置为相对动作且启用 use_relative_action，则结合给定状态，把相对动作还原为绝对动作轨迹；
        【中文】该接口常用于“把模型输出的归一化相对动作，转换成机器人可以直接执行的绝对关节/末端动作”。
        """

        # Step 1: Unnormalize actions
        # 【中文】第一步：将动作值反归一化。
        unnormalized_values = {}
        modality_keys = self.modality_configs[embodiment_tag]["action"].modality_keys
        
        for joint_group in modality_keys:
            if joint_group not in action:
                raise KeyError(
                    f"Joint group '{joint_group}' not found in action dict for embodiment '{embodiment_tag}'"
                )

            params = self.norm_params[embodiment_tag]["action"][joint_group]
            group_values = action[joint_group]

            if ( # mean-std归一化
                self.modality_configs[embodiment_tag]["action"].mean_std_embedding_keys is not None
                and joint_group
                in self.modality_configs[embodiment_tag]["action"].mean_std_embedding_keys
            ):
                unnormalized = unnormalize_values_meanstd(group_values, params)
            else: # min-max归一化（默认）
                unnormalized = unnormalize_values_minmax(group_values, params)

            unnormalized_values[joint_group] = unnormalized

        # Step 2: Convert relative actions to absolute (if needed)
        # 【中文】第二步：如果配置了相对动作且启用，则将相对动作转换为绝对动作。
        action_configs = self.modality_configs[embodiment_tag]["action"].action_configs

        if action_configs is not None:
            for key, action_config in zip(modality_keys, action_configs):
                if action_config.rep == ActionRepresentation.RELATIVE and self.use_relative_action:
                    if state is None:
                        raise ValueError(
                            f"State dict required for relative->absolute conversion of key '{key}' "
                            f"in embodiment '{embodiment_tag}'"
                        )

                    # Determine which state key to use as reference
                    # 【中文】确定用作参考系的状态键（默认为当前关节组键）。
                    state_key = action_config.state_key if action_config.state_key else key

                    if state_key not in state:
                        raise KeyError(
                            f"Reference state key '{state_key}' not found in state dict "
                            f"for embodiment '{embodiment_tag}'"
                        )

                    relative_action = unnormalized_values[key]


                    # Handle batched and unbatched cases
                    # 【中文】处理 batch 和非 batch 的情况，确保数据维度一致。
                    # 使用 None (或 np.newaxis) 增加一个维度，将 (T, D) 转换为 (1, T, D)
                    is_batched = relative_action.ndim == 3
                    if not is_batched:
                        assert relative_action.ndim == 2
                        reference_state = state[state_key]
                        if reference_state.ndim == 2:
                            # 将 (T_state, D) 转换为 (1, T_state, D)
                            reference_state = reference_state[None, :]
                        # 将 (T, D) 转换为 (1, T, D)
                        relative_action = relative_action[None, :]
                    else:
                        reference_state = state[state_key]
                        if reference_state.ndim == 2:
                            # 如果 action 是 batched (B, T, D) 但 state 只有 (T_state, D)
                            # 则将 state 转换为 (1, T_state, D) 以便后续 zip 迭代
                            reference_state = reference_state[None, :]

                    # Convert batched relative actions to absolute
                    # 【中文】将 batch 中的相对动作转换为绝对动作。
                    absolute_actions = []
                    for s, a in zip(reference_state, relative_action):
                        # Use last timestep of state as reference
                        # 【中文】使用状态序列的最后一帧作为参考位姿。

                        # 示例：a 的形状为 (T, D)，表示动作序列；s[-1] 的形状为 (D,)，表示参考状态（如末端位姿或关节角）
                        absolute_action = self._convert_to_absolute_action(
                            action=a,
                            reference_state=s[-1], # 基于样本state horizon的最后时刻状态计算相对动作的绝对值
                            action_type=action_config.type,
                            action_format=action_config.format,
                        )
                        absolute_actions.append(absolute_action)

                    if is_batched:
                        unnormalized_values[key] = np.stack(absolute_actions, axis=0)
                    else:
                        unnormalized_values[key] = absolute_actions[0]

        return unnormalized_values

    def apply(
        self,
        state: dict[str, np.ndarray],
        action: dict[str, np.ndarray],
        embodiment_tag: str,
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """
        Apply both state and action processing together.

        Convenience method that processes state and action in one call,
        automatically passing raw state to action processor for relative conversion.

        Args:
            state: Dict mapping joint_group -> raw state values
            action: Dict mapping joint_group -> raw action values
            embodiment_tag: Embodiment identifier

        Returns:
            Tuple of (processed_state, processed_action)

        【中文】同时对“状态 + 动作”做处理的便捷接口：
        【中文】- 先调用 `apply_state` 对状态归一化；
        【中文】- 再将**原始状态**（未归一化）传给 `apply_action`，以便在相对动作模式下使用原始参考系；
        【中文】- 训练时要求 action 非空；推理时允许 action 为空（只处理状态）。
        """
        processed_state = self.apply_state(state, embodiment_tag)
        if action:
            processed_action = self.apply_action(action, embodiment_tag, state=state)
        else:
            assert not self.training, "Action is required in training mode"
            processed_action = {}
        return processed_state, processed_action

    def unapply(
        self,
        state: dict[str, np.ndarray],
        action: dict[str, np.ndarray],
        embodiment_tag: str,
        raw_state: dict[str, np.ndarray] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """
        Reverse both state and action processing together.

        Args:
            state: Dict mapping joint_group -> processed state values
            action: Dict mapping joint_group -> processed action values
            embodiment_tag: Embodiment identifier
            raw_state: Optional dict of raw states for relative->absolute conversion
                If None, will use unapplied state (but won't work for sin/cos encoded states)

        Returns:
            Tuple of (raw_state, raw_action)

        【中文】同时撤销“状态 + 动作”的处理：
        【中文】- 先尝试调用 `unapply_state` 将状态从归一化空间还原回原始空间；
        【中文】- 再用还原后的状态（或调用者提供的 raw_state）作为参考系，把动作从归一化/相对表示还原为原始动作；
        【中文】若状态经过 sin/cos 编码而又没有提供 raw_state，则无法完全恢复，会抛出明确异常提示。
        """
        # Unapply state first
        try:
            unapplied_state = self.unapply_state(state, embodiment_tag)
        except ValueError as e:
            if "sin/cos encoding" in str(e) and raw_state is None:
                raise ValueError(
                    "Cannot unapply sin/cos encoded state. Please provide raw_state parameter."
                ) from e
            raise

        # Use provided raw_state if available, otherwise use unapplied state
        state_for_action = raw_state if raw_state is not None else unapplied_state

        # Unapply action
        unapplied_action = self.unapply_action(action, embodiment_tag, state=state_for_action)

        return unapplied_state, unapplied_action

    def get_state_dim(self, embodiment_tag: str, include_sincos_expansion: bool = False) -> int:
        """
        Get total state dimension after processing.

        Args:
            embodiment_tag: Embodiment identifier
            include_sincos_expansion: If True, accounts for sin/cos encoding doubling dimensions

        Returns:
            Total state dimension across all joint groups

        【中文】获取“处理后的状态总维度”：
        【中文】- 先按 `modality_configs[embodiment_tag]["state"].modality_keys` 遍历所有关节组；
        【中文】- 每个关节组使用 `norm_params` 中记录的 `dim` 作为基础维度；
        【中文】- 若 `include_sincos_expansion=True` 且该组启用了 sin/cos 编码，则维度乘以 2；
        【中文】常用于推断模型输入层的状态特征维度，或在 Processor 之外构建自定义网络时做 sanity check。
        """
        total_dim = 0
        state_config = self.modality_configs[embodiment_tag]["state"]

        # Get sin/cos embedding keys if enabled
        sin_cos_keys = set()
        if self.apply_sincos_state_encoding and hasattr(state_config, "sin_cos_embedding_keys"):
            sin_cos_keys = set(state_config.sin_cos_embedding_keys)

        for joint_group in state_config.modality_keys:
            base_dim = self.norm_params[embodiment_tag]["state"][joint_group]["dim"].item()

            # Sin/cos encoding doubles the dimension
            if include_sincos_expansion and joint_group in sin_cos_keys:
                total_dim += base_dim * 2
            else:
                total_dim += base_dim

        return total_dim

    def get_action_dim(self, embodiment_tag: str) -> int:
        """
        Get total action dimension.

        Args:
            embodiment_tag: Embodiment identifier

        Returns:
            Total action dimension across all joint groups

        【中文】获取“动作总维度”：遍历当前具身配置中的各个动作关节组，将其 `dim` 相加，
        【中文】通常用于构造策略网络的输出维度或检查 Processor 与模型头部的一致性。
        """
        total_dim = 0
        for joint_group in self.modality_configs[embodiment_tag]["action"].modality_keys:
            total_dim += self.norm_params[embodiment_tag]["action"][joint_group]["dim"].item()
        return total_dim

    def _convert_to_relative_action(
        self,
        action: np.ndarray,
        reference_state: np.ndarray,
        action_type: ActionType,
        action_format: ActionFormat,
    ) -> np.ndarray:
        """Convert absolute action to relative action using reference state.

        【中文】将“绝对动作序列”转换为“相对参考状态”的动作：
        【中文】- 对末端动作（EEF）使用 `EndEffectorActionChunk` + `EndEffectorPose` 做位姿差分；
        【中文】- 对关节空间动作（NON_EEF）使用 `JointActionChunk` + `JointPose` 做关节差分；
        【中文】- 最终再按指定的 `action_format`（如 xyz+rot6d）导出为数值数组。
        """
        assert action.ndim == 2, f"Expected action shape (T, D), got {action.shape}"
        assert reference_state.ndim == 1, f"Expected state shape (D,), got {reference_state.shape}"

        '''
        控制空间不同：
        EEF：在笛卡尔空间（3D 世界坐标）控制末端位姿。
        NON_EEF：在关节空间（每个关节角度/位置）控制整条机械臂。
        优缺点直觉：
        EEF：
        优点：对「手要去哪里」的语义更直观，跟任务描述更贴近（抓、放、插入等），适合高层策略/模仿动作。
        缺点：需要底层的逆运动学/控制器，把末端位姿转成关节命令；有些姿态在某些机器人上不可达或有多解。
        NON_EEF：
        优点：直接跟电机命令一一对应，控制链条短、可预测性强；没有额外 IK 误差。
        缺点：对模型来说，学到的是“关节怎么动”而不是“手去哪”，语义没那么直观，也更依赖具体机器人结构。
        '''
        # 末端爪子
        if action_type == ActionType.EEF:
            assert action.shape[1] == 9, (
                f"Expected action dim 9 (xyz + rot6d) for EEF, got {action.shape[1]}"
            )

            action_chunking = EndEffectorActionChunk(
                [
                    EndEffectorPose(translation=m[:3], rotation=m[3:], rotation_type="rot6d")
                    for m in action # T个连续的action
                ]
            )
            # state作为基准，计算相对变化
            reference_frame = EndEffectorPose(
                translation=reference_state[:3],
                rotation=reference_state[3:],
                rotation_type="rot6d",
            )

        elif action_type == ActionType.NON_EEF: # 非末端的关节
            action_chunking = JointActionChunk([JointPose(m) for m in action])
            reference_frame = JointPose(reference_state)            # state作为基准，计算相对变化

        else:
            raise ValueError(f"Unknown ActionType: {action_type}")


        # 将绝对动作转换为相对于参考帧的动作
        # 对于末端执行器(EEF)动作：计算目标位姿相对于当前末端位姿的偏移量（位置差+旋转差）
        # 对于关节空间(NON_EEF)动作：计算目标关节角度相对于当前关节角度的偏移量
        relative_action_chunking = action_chunking.relative_chunking(
            reference_frame=reference_frame
        )
        return relative_action_chunking.to(action_format)

    def _convert_to_absolute_action(
        self,
        action: np.ndarray,
        reference_state: np.ndarray,
        action_type: ActionType,
        action_format: ActionFormat,
    ) -> np.ndarray:
        """Convert relative action to absolute action using reference state.

        【中文】将“相对动作序列”恢复为“绝对动作”：
        【中文】- 逻辑上与 `_convert_to_relative_action` 相反，给定参考状态与相对动作；
        【中文】- 对末端/关节空间分别使用对应的 chunking/pose 类型；
        【中文】- 最终得到与 `action_format` 对应的绝对动作数组，可直接下发给机器人控制器或仿真环境。
        """
        assert action.ndim == 2, f"Expected action shape (T, D), got {action.shape}"
        assert reference_state.ndim == 1, f"Expected state shape (D,), got {reference_state.shape}"
        assert reference_state.shape[0] == action.shape[1], (
            f"State dim {reference_state.shape[0]} != action dim {action.shape[1]}"
        )

        if action_type == ActionType.EEF:
            assert action.shape[1] == 9, (
                f"Expected action dim 9 (xyz + rot6d) for EEF, got {action.shape[1]}"
            )

            # action chunk
            rel_action = EndEffectorActionChunk(
                [
                    EndEffectorPose(translation=m[:3], rotation=m[3:], rotation_type="rot6d")
                    for m in action # 每个horizon转成xyz-rot6d
                ]
            )
            # reference state
            reference_frame = EndEffectorPose(
                translation=reference_state[:3],
                rotation=reference_state[3:],
                rotation_type="rot6d",
            )

        elif action_type == ActionType.NON_EEF:
            # action chunk
            rel_action = JointActionChunk([JointPose(pose) for pose in action])
            # reference state
            reference_frame = JointPose(reference_state)

        else:
            raise ValueError(f"Unknown ActionType: {action_type}")

        abs_action = rel_action.to_absolute_chunking(reference_frame=reference_frame)
        return abs_action.to(action_format)

    def __str__(self) -> str:
        return f"StateActionProcessor(modality_configs={self.modality_configs}, statistics={self.statistics}, use_percentiles={self.use_percentiles}, clip_outliers={self.clip_outliers}, apply_sincos_state_encoding={self.apply_sincos_state_encoding}, use_relative_action={self.use_relative_action})"
