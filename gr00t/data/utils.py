from dataclasses import asdict, is_dataclass
from enum import Enum
from typing import Any

import numpy as np

from gr00t.configs.data.embodiment_configs import ModalityConfig


def apply_sin_cos_encoding(values: np.ndarray) -> np.ndarray:
    """Apply sin/cos encoding to values.

    Args:
        values: Array of shape (..., D) containing values to encode

    Returns:
        Array of shape (..., 2*D) with [sin, cos] concatenated

    Note: This DOUBLES the dimension. For example:
        Input:  [v₁, v₂, v₃] with shape (..., 3)
        Output: [sin(v₁), sin(v₂), sin(v₃), cos(v₁), cos(v₂), cos(v₃)] with shape (..., 6)

    【中文】对输入向量做逐元素的 sin/cos 编码：
    【中文】- 输入 shape 为 (..., D)，通常对应一组角度或周期型信号；
    【中文】- 输出 shape 为 (..., 2*D)，前一半是 sin(values)，后一半是 cos(values)；
    【中文】- 常用于将角度类状态映射到无界空间，消除 2π 周期带来的不连续性，以利于模型学习。
    """
    sin_values = np.sin(values)
    cos_values = np.cos(values)
    # Concatenate sin and cos: [sin(v1), sin(v2), ..., cos(v1), cos(v2), ...]
    return np.concatenate([sin_values, cos_values], axis=-1)


def nested_dict_to_numpy(data):
    """
    Recursively converts bottom-level list of lists to NumPy arrays.

    Args:
        data: A nested dictionary where bottom nodes are list of lists,
              and parent nodes are strings (keys)

    Returns:
        The same dictionary structure with bottom-level lists converted to NumPy arrays

    Example:
        >>> data = {"a": {"b": [[0, 1], [2, 3]]}}
        >>> result = nested_dict_to_numpy(data)
        >>> print(result["a"]["b"])
        [[0 1]
         [2 3]]

    【中文】递归地把“嵌套字典”中最底层的 list/list of lists 转成 numpy.ndarray：
    【中文】- 输入通常是从 JSON 或 dataclass 转换而来的多层结构；
    【中文】- 遇到 dict 就对子节点递归调用，遇到 list 则直接用 np.array 包装；
    【中文】- 常用于将统计量或配置中的列表结构变成方便数值运算的 numpy 形式。
    """
    if isinstance(data, dict):
        return {key: nested_dict_to_numpy(value) for key, value in data.items()}
    elif isinstance(data, list):
        # Convert lists to numpy arrays
        # NumPy will handle both 1D and 2D cases appropriately
        return np.array(data)
    else:
        return data


def normalize_values_minmax(values, params):
    """
    Normalize values using min-max normalization to [-1, 1] range.

    Args:
        values: Input values to normalize
            - Shape: (T, D) or (B, T, D) where B is batch, T is time/step, D is feature dimension
            - Can handle 2D or 3D arrays where last axis represents features
        params: Dictionary with "min" and "max" keys
            - params["min"]: Minimum values for normalization
                * Case 1 - 1D bounds: Shape (D,) - same min/max for all steps
                * Case 2 - 2D bounds: Shape (T, D) - different min/max per step
            - params["max"]: Maximum values for normalization
                * Case 1 - 1D bounds: Shape (D,) - same min/max for all steps
                * Case 2 - 2D bounds: Shape (T, D) - different min/max per step
        joint_group: Optional indexing for joint groups (legacy parameter)

    Returns:
        Normalized values in [-1, 1] range
            - Same shape as input values: (T, D) or (B, T, D)
            - Values are linearly mapped from [min, max] to [-1, 1]
            - For features where min == max, normalized value is 0

    Examples:
        # 1D bounds - same normalization for all steps
        values: (10, 5), params["min"]: (5,), params["max"]: (5,)

        # 2D bounds - per-step normalization
        values: (8, 4), params["min"]: (8, 4), params["max"]: (8, 4)

    【中文】按逐特征的 min/max 将数值线性映射到 [-1, 1] 区间：
    【中文】- 支持 2D (T, D) 或 3D (B, T, D) 形状，最后一维为特征维；
    【中文】- params["min"], params["max"] 可以是一维（所有时间步共用边界），也可以是二维（每个时间步单独边界）；
    【中文】- 对于 min==max 的特征维，归一化结果直接置为 0，避免除以 0；
    【中文】常用于将状态/动作缩放到统一尺度，方便在不同维度间共享网络结构。
    """
    min_vals = params["min"]
    max_vals = params["max"]
    normalized = np.zeros_like(values)

    mask = ~np.isclose(max_vals, min_vals)

    normalized[..., mask] = (values[..., mask] - min_vals[..., mask]) / ( # 0~1
        max_vals[..., mask] - min_vals[..., mask]
    )
    normalized[..., mask] = 2 * normalized[..., mask] - 1  # -1~1

    return normalized


def unnormalize_values_minmax(normalized_values, params):
    """
    Min-max unnormalization from [-1, 1] range back to original range.

    Args:
        normalized_values: Normalized input values in [-1, 1] range
            - Shape: (T, D) or (B, T, D) where B is batch, T is time/step, D is feature dimension
            - Values outside [-1, 1] are automatically clipped
        params: Dictionary with "min" and "max" keys
            - params["min"]: Original minimum values used for normalization
                * Case 1 - 1D bounds: Shape (D,) - same min/max for all steps
                * Case 2 - 2D bounds: Shape (T, D) - different min/max per step
            - params["max"]: Original maximum values used for normalization
                * Case 1 - 1D bounds: Shape (D,) - same min/max for all steps
                * Case 2 - 2D bounds: Shape (T, D) - different min/max per step

    Returns:
        Unnormalized values in original range [min, max]
            - Same shape as input normalized_values: (T, D) or (B, T, D)
            - Values are linearly mapped from [-1, 1] back to [min, max]
            - Input values are clipped to [-1, 1] before unnormalization

    Examples:
        # 1D bounds - same unnormalization for all steps
        normalized_values: (10, 5), params["min"]: (5,), params["max"]: (5,)

        # 2D bounds - per-step unnormalization
        normalized_values: (8, 4), params["min"]: (8, 4), params["max"]: (8, 4)

    【中文】将经过 min-max 归一化到 [-1, 1] 的值还原回原始数值区间：
    【中文】- 先把输入裁剪到 [-1, 1]，再按线性映射反变换到 [min, max]；
    【中文】- 支持全局边界和逐步边界两种形式，与 `normalize_values_minmax` 完全对偶；
    【中文】常用于从模型输出的归一化动作/状态恢复到物理空间的实际量纲。
    """

    min_vals = params["min"]
    max_vals = params["max"]
    range_vals = max_vals - min_vals

    # Unnormalize from [-1, 1]

    # 将 [-1, 1] 范围内的归一化值映射回原始范围 [min, max]
    # 步骤：
    # 1. np.clip(normalized_values, -1.0, 1.0): 确保输入值在预期范围内，处理由于浮点误差可能产生的越界值
    # 2. + 1.0: 将 [-1, 1] 映射到 [0, 2]
    # 3. / 2.0: 将 [0, 2] 映射到 [0, 1] (0-1 归一化)
    # 4. * range_vals: 将 [0, 1] 缩放到原始的跨度大小 [0, max-min]
    # 5. + min_vals: 平移到原始的最小值，得到最终的 [min, max]
    unnormalized = (np.clip(normalized_values, -1.0, 1.0) + 1.0) / 2.0 * range_vals + min_vals
    return unnormalized


def normalize_values_meanstd(values, params):
    """
    Normalize values using mean-std (z-score) normalization.

    Args:
        values: Input values to normalize
            - Shape: (T, D) or (B, T, D) where B is batch, T is time/step, D is feature dimension
            - Can handle 2D or 3D arrays where last axis represents features
        params: Dictionary with "mean" and "std" keys
            - params["mean"]: Mean values for normalization
                * Case 1 - 1D params: Shape (D,) - same mean for all steps
                * Case 2 - 2D params: Shape (T, D) - different mean per step
            - params["std"]: Standard deviation values for normalization
                * Case 1 - 1D params: Shape (D,) - same std for all steps
                * Case 2 - 2D params: Shape (T, D) - different std per step

    Returns:
        Normalized values using z-score normalization
            - Same shape as input values: (T, D) or (B, T, D)
            - Values are transformed as: (x - mean) / std
            - For features where std == 0, normalized value equals original value

    Examples:
        # 1D params - same normalization for all steps
        values: (10, 5), params["mean"]: (5,), params["std"]: (5,)

        # 2D params - per-step normalization
        values: (8, 4), params["mean"]: (8, 4), params["std"]: (8, 4)

    【中文】使用均值/方差做 z-score 归一化：
    【中文】- 公式为 (x - mean) / std，在不同时间步或不同维度上可以使用各自的 mean/std；
    【中文】- std==0 时不做缩放，直接返回原值，避免数值溢出；
    【中文】常用于“高斯化”某些特征（例如速度、加速度），使其在网络输入中具有零均值与单位方差。
    """
    mean_vals = params["mean"]
    std_vals = params["std"]

    # Create mask for non-zero standard deviations
    mask = std_vals != 0

    # Initialize normalized array
    normalized = np.zeros_like(values)

    # Normalize only features with non-zero std
    normalized[..., mask] = (values[..., mask] - mean_vals[..., mask]) / std_vals[..., mask]

    # Keep original values for zero-std features
    normalized[..., ~mask] = values[..., ~mask]

    return normalized


def unnormalize_values_meanstd(normalized_values, params):
    """
    Mean-std unnormalization (reverse z-score normalization).

    Args:
        normalized_values: Normalized input values (z-scores)
            - Shape: (T, D) or (B, T, D) where B is batch, T is time/step, D is feature dimension
            - Can handle 2D or 3D arrays where last axis represents features
        params: Dictionary with "mean" and "std" keys
            - params["mean"]: Original mean values used for normalization
                * Case 1 - 1D params: Shape (D,) - same mean for all steps
                * Case 2 - 2D params: Shape (T, D) - different mean per step
            - params["std"]: Original standard deviation values used for normalization
                * Case 1 - 1D params: Shape (D,) - same std for all steps
                * Case 2 - 2D params: Shape (T, D) - different std per step

    Returns:
        Unnormalized values in original scale
            - Same shape as input normalized_values: (T, D) or (B, T, D)
            - Values are transformed as: x * std + mean
            - For features where std == 0, unnormalized value equals normalized value

    Examples:
        # 1D params - same unnormalization for all steps
        normalized_values: (10, 5), params["mean"]: (5,), params["std"]: (5,)

        # 2D params - per-step unnormalization
        normalized_values: (8, 4), params["mean"]: (8, 4), params["std"]: (8, 4)

    【中文】撤销均值/方差归一化（z-score 的反变换）：
    【中文】- 使用公式 x = normalized * std + mean 将值从标准化空间还原回原始尺度；
    【中文】- std==0 的特征维直接保留原值，确保数值稳定；
    【中文】通常与 `normalize_values_meanstd` 配套，用于把模型输出或中间特征还原到物理量纲。
    """
    mean_vals = params["mean"]
    std_vals = params["std"]

    # Create mask for non-zero standard deviations
    mask = std_vals != 0

    # Initialize unnormalized array
    unnormalized = np.zeros_like(normalized_values)

    # Unnormalize only features with non-zero std
    unnormalized[..., mask] = (
        normalized_values[..., mask] * std_vals[..., mask] + mean_vals[..., mask]
    )

    # Keep normalized values for zero-std features
    unnormalized[..., ~mask] = normalized_values[..., ~mask]

    return unnormalized


def to_json_serializable(obj: Any) -> Any:
    """
    Recursively convert dataclasses and numpy arrays to JSON-serializable format.

    Args:
        obj: Object to convert (can be dataclass, numpy array, dict, list, etc.)

    Returns:
        JSON-serializable representation of the object

    【中文】递归地将各种 Python / NumPy / dataclass 对象转换为“可 JSON 序列化”的基础类型：
    【中文】- dataclass → dict → 继续递归处理；
    【中文】- numpy 数组/标量 → list/int/float/bool；
    【中文】- dict / list / tuple / set → 逐元素转换；
    【中文】- Enum → 使用枚举名（name）；
    【中文】常用于将配置、统计信息或 Processor 状态写入 json 文件，保证不会出现无法序列化的对象。
    """
    if is_dataclass(obj) and not isinstance(obj, type):
        # Convert dataclass to dict, then recursively process the dict
        return to_json_serializable(asdict(obj))
    elif isinstance(obj, np.ndarray):
        # Convert numpy array to list
        return obj.tolist()
    elif isinstance(obj, np.integer):
        # Convert numpy integers to Python int
        return int(obj)
    elif isinstance(obj, np.floating):
        # Convert numpy floats to Python float
        return float(obj)
    elif isinstance(obj, np.bool_):
        # Convert numpy bool to Python bool
        return bool(obj)
    elif isinstance(obj, dict):
        # Recursively process dictionary values
        return {key: to_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        # Recursively process list/tuple elements
        return [to_json_serializable(item) for item in obj]
    elif isinstance(obj, set):
        # Convert set to list
        return [to_json_serializable(item) for item in obj]
    elif isinstance(obj, (str, int, float, bool, type(None))):
        # Already JSON-serializable
        return obj
    elif isinstance(obj, Enum):
        return obj.name
    else:
        # For other types, try to convert to string as fallback
        # You might want to handle specific types differently
        return str(obj)


def parse_modality_configs(
    modality_configs: dict[str, dict[str, ModalityConfig]],
) -> dict[str, dict[str, ModalityConfig]]:
    """Parse nested modality config dicts into ModalityConfig objects.
    【中文】解析 `embodiment_configs.py` 中的模态配置结构：
    【中文】- 外层 key: 具身形态标签（如 "unitree_g1"、"libero_panda"）
    【中文】- 内层 key: 模态名称（"video" / "state" / "action" / "language" 等）
    【中文】- 内层 value: 若为 dict，则用 ModalityConfig(**config) 构造成对象；否则直接使用现有 ModalityConfig
    【中文】典型字段包括：
    【中文】- delta_indices: 时间维索引（哪些时间步参与该模态）
    【中文】- modality_keys: 该模态下的信号名称列表（例如各关节、图像视角）
    【中文】- action_configs: 对每个动作子模态的表示方式（绝对/相对、EEF/非EEF、格式等）
    【中文】- mean_std_embedding_keys: 参与统计均值/方差的子模态名称
    """
    parsed_modality_configs = {}
    for embodiment_tag, modality_config in modality_configs.items():
        parsed_modality_configs[embodiment_tag] = {}
        for modality, config in modality_config.items():
            if isinstance(config, dict):
                parsed_modality_configs[embodiment_tag][modality] = ModalityConfig(**config)
            else:
                parsed_modality_configs[embodiment_tag][modality] = config
    return parsed_modality_configs
