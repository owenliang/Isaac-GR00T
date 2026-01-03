# 【中文】数据配置模块：定义单个数据集和混合数据集的配置
from dataclasses import dataclass, field
from typing import Any, List, Optional

from gr00t.data.types import ModalityConfig

from .embodiment_configs import MODALITY_CONFIGS


@dataclass
class SingleDatasetConfig:
    """Configuration for a single dataset in a mixed-training setup.

    A list of these objects can be supplied in ``DataConfig.datasets`` to mix
    multiple datasets at arbitrary ratios.  For convenience the *legacy*
    single-dataset fields still exist; if ``datasets`` is non-empty they take
    precedence.
    
    【中文】单个数据集的配置类，用于混合训练。
    【中文】可以在 ``DataConfig.datasets`` 中提供多个此类对象，按任意比例混合多个数据集。
    """

    # Path to the dataset root directory (can be strings or dicts for complex configs)
    # 【中文】数据集根目录路径（可以是字符串或字典）
    dataset_paths: List[Any]

    # Robot embodiment identifier (e.g. "gr1", "franka")
    # 【中文】机器人具身形态标识符（例如 "gr1", "franka"）
    embodiment_tag: Optional[str] = None

    # Relative sampling probability (will be normalised across the list)
    # 【中文】相对采样概率（会在列表中归一化）
    mix_ratio: float = 1.0

    # 【中文】数据集类型，默认为物理具身形态
    dataset_type: str = "physical_embodiment"

    # Optional validation dataset path for open-loop evaluation
    # If not provided, falls back to dataset_paths for evaluation
    # 【中文】可选的验证数据集路径，用于开环评估
    # 【中文】如果未提供，则回退到 dataset_paths 用于评估
    val_dataset_path: Optional[str] = None


@dataclass
class DataConfig:
    """Dataset configuration (supports single or multiple datasets).
    【中文】数据集配置类（支持单个或多个数据集）。
    """

    # Leave empty by default for backwards-compatibility with the original
    # single-dataset workflow.  Users can supply one or more configs via CLI or
    # YAML when they need mixing.
    # 【中文】默认留空以保持向后兼容。用户可以通过CLI或YAML提供一个或多个配置来进行混合训练
    datasets: List[SingleDatasetConfig] = field(default_factory=list)

    # Modality configs
    # There are three sources of modality configs:
    # 1. Default modality configs in code: gr00t/configs/data/embodiment_configs.py
    # 2. Modality configs supplied through command line: --data.modality_configs (although rare and inconvenient)
    # 1 and 2 are unified through `config.data.modality_configs`.
    # 3. modality configs saved in the pretrained checkpoint.
    # 【中文】模态配置有三个来源：
    # 【中文】1. 代码中的默认配置：gr00t/configs/data/embodiment_configs.py
    # 【中文】2. 通过命令行提供：--data.modality_configs（虽然很少用且不方便）
    # 【中文】3. 预训练检查点中保存的配置
    modality_configs: dict[str, dict[str, ModalityConfig]] = field(
        default_factory=lambda: MODALITY_CONFIGS
    )

    # Sharded dataset configuration
    # 【中文】分片数据集配置
    download_cache: bool = False  # 【中文】是否下载缓存
    shard_size: int = 2**10  # 【中文】分片大小（1024）
    episode_sampling_rate: float = 0.1  # 【中文】episode采样率
    num_shards_per_epoch: int = int(1e5)  # 【中文】每个epoch的分片数量

    # Override statistics from the pretrained checkpoint
    # 【中文】是否覆盖预训练检查点的统计信息
    override_pretraining_statistics: bool = False

    # General task / mode config (shared across datasets)
    # 【中文】通用任务/模式配置（跨数据集共享）
    mode: str = "single_turn"  # 【中文】模式：单轮对话
    random_chop: float = 0.0  # 【中文】随机裁剪概率
    mock_dataset_mode: bool = False  # if True, cache the first datapoint of each dataset and always return one of them to simulate best-case dataloading
    # 【中文】模拟数据集模式：如果为True，缓存每个数据集的第一个数据点并始终返回其中之一，模拟最佳情况的数据加载

    # Data loading
    # 【中文】数据加载配置
    shuffle: bool = True  # 【中文】是否打乱数据
    seed: int = 42  # 【中文】随机种子
    multiprocessing_context: str = "fork"  # Options: "fork", "spawn", and "forkserver"
    # 【中文】多进程上下文：选项包括 "fork"、"spawn" 和 "forkserver"
    allow_padding: bool = False  # 【中文】是否允许填充

    # Subsample ratio for the dataset
    # 【中文】数据集的子采样比例
    subsample_ratio: float = 1.0

    # DP Image Config
    # 【中文】扩散策略图像配置
    image_crop_size: List[int] = field(default_factory=lambda: [244, 244])  # 【中文】图像裁剪尺寸
    image_target_size: List[int] = field(default_factory=lambda: [224, 224])  # 【中文】图像目标尺寸
    video_backend: str = "torchcodec"  # 【中文】视频后端
