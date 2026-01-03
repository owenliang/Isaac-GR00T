# Finetune config used for single node post-training.
# 【中文】单节点微调的配置类，定义了所有微调所需的参数
from dataclasses import dataclass

from gr00t.data.embodiment_tags import EmbodimentTag


@dataclass
class FinetuneConfig:
    """
    Configuration for fine-tuning a Vision-Language-Action (VLA) model.

    This dataclass defines all parameters needed to launch a fine-tuning job
    on a pretrained base model using a custom dataset and embodiment-specific
    modality configuration. It controls model tuning options, data augmentation,
    and training hyperparameters.
    
    【中文】视觉-语言-动作(VLA)模型微调配置类。
    【中文】定义了从预训练模型开始微调所需的所有参数，包括：
    - 数据和模型路径
    - 模型各模块的微调开关（LLM、视觉编码器、投影层、扩散模型）
    - 数据增强参数（旋转、颜色抖动等）
    - 训练超参数（batch size、学习率、优化器等）
    """

    # --- Data and Model Paths ---
    # 【中文】--- 数据和模型路径 ---
    base_model_path: str
    """Path to the pretrained base model checkpoint (e.g., Hugging Face model hub or local directory).
    【中文】预训练基础模型的路径（HuggingFace模型仓库或本地目录）"""

    dataset_path: str
    """Path to the dataset root directory containing trajectory data for fine-tuning.
    【中文】微调数据集根目录路径，包含机器人轨迹数据"""

    embodiment_tag: EmbodimentTag
    """Identifier specifying which embodiment (robot configuration) this fine-tuning run targets.
    【中文】指定此次微调针对的具身形态（机器人配置）标识符"""

    modality_config_path: str | None = None
    """
    Path to a Python file defining the modality configuration for the given embodiment. 
    If None, use the pre-registered modality config in `gr00t/configs/data/embodiment_configs.py`. 
    
    【中文】定义该具身形态模态配置的Python文件路径。
    【中文】如果为None，则使用 `gr00t/configs/data/embodiment_configs.py` 中预注册的模态配置。
    """

    # --- Model Tuning Flags ---
    # 【中文】--- 模型微调开关 ---
    tune_llm: bool = False
    """If True, fine-tune the language model (LLM) backbone during training.
    【中文】如果为True，在训练期间微调语言模型(LLM)骨干网络"""

    tune_visual: bool = False
    """If True, fine-tune the visual encoder (e.g., ViT or CNN backbone).
    【中文】如果为True，微调视觉编码器（例如ViT或CNN骨干网络）"""

    tune_projector: bool = True
    """If True, fine-tune the multimodal projector layers that map vision/language features to a shared space.
    【中文】如果为True，微调多模态投影层（将视觉/语言特征映射到共享空间）"""

    tune_diffusion_model: bool = True
    """If True, fine-tune the diffusion-based action decoder (if present in the model).
    【中文】如果为True，微调基于扩散的动作解码器（如果模型中存在）"""

    state_dropout_prob: float = 0.0
    """
    Dropout probability applied to state inputs for regularization during training.
    【中文】训练期间应用于状态输入的Dropout概率，用于正则化
    """

    # --- Data Augmentation ---
    # 【中文】--- 数据增强 ---
    random_rotation_angle: int | None = None
    """Maximum rotation angle (in degrees) for random rotation augmentation of input images.
    【中文】输入图像随机旋转增强的最大旋转角度（以度为单位）"""

    color_jitter_params: dict[str, float] | None = None
    """
    Parameters for color jitter augmentation on images.

    Expected keys include:
      - "brightness": float
      - "contrast": float
      - "saturation": float
      - "hue": float
    Example: {"brightness": 0.4, "contrast": 0.4, "saturation": 0.4, "hue": 0.1}

    If None, applying the default color jitter augmentation from the pretrained model.
    
    【中文】图像颜色抖动增强的参数。
    【中文】预期的键包括：brightness(亮度)、contrast(对比度)、saturation(饱和度)、hue(色调)
    【中文】示例：{"brightness": 0.4, "contrast": 0.4, "saturation": 0.4, "hue": 0.1}
    【中文】如果为None，则应用预训练模型的默认颜色抖动增强
    """

    # --- Training Configuration ---
    # 【中文】--- 训练配置 ---
    global_batch_size: int = 64
    """Total effective batch size across all GPUs and accumulation steps.
    【中文】所有GPU和累积步骤的总有效批次大小"""

    dataloader_num_workers: int = 2
    """Number of parallel worker processes used for data loading.
    【中文】用于数据加载的并行工作进程数"""

    learning_rate: float = 1e-4
    """Initial learning rate for optimizer.
    【中文】优化器的初始学习率"""

    gradient_accumulation_steps: int = 1
    """Number of forward passes to accumulate before performing a backward/update step.
    【中文】执行反向传播/更新步骤之前累积的前向传播次数"""

    output_dir: str = "./outputs"
    """Directory where model checkpoints, logs, and outputs are saved.
    【中文】保存模型检查点、日志和输出的目录"""

    save_steps: int = 1000
    """Frequency (in training steps) at which to save checkpoints.
    【中文】保存检查点的频率（以训练步数为单位）"""

    save_total_limit: int = 5
    """Maximum number of checkpoints to keep before older ones are deleted.
    【中文】保留的最大检查点数量，超过后删除旧的检查点"""

    num_gpus: int = 1
    """Number of GPUs available for distributed or single-node training.
    【中文】可用于分布式或单节点训练的GPU数量"""

    use_wandb: bool = False
    """
    If True, log metrics and artifacts to Weights & Biases (wandb).
    The project is `finetune-gr00t-n1d6`.
    You need to login to wandb to view the logs.
    
    【中文】如果为True，将指标和产物记录到Weights & Biases (wandb)。
    【中文】项目名称为 `finetune-gr00t-n1d6`。
    【中文】需要登录wandb才能查看日志。
    """

    max_steps: int = 10000
    """Total number of training steps to run before stopping.
    【中文】停止前要运行的训练步数总数"""

    weight_decay: float = 1e-5
    """Weight decay coefficient for optimizer (L2 regularization).
    【中文】优化器的权重衰减系数（L2正则化）"""

    warmup_ratio: float = 0.05
    """Proportion of total training steps used for learning rate warm-up.
    【中文】用于学习率预热的训练步数比例"""

    shard_size: int = 2**10
    """Size of the shard to use for the dataset during preloading.
    【中文】预加载期间用于数据集的分片大小"""

    episode_sampling_rate: float = 0.1
    """Sampling rate for the episodes.
    【中文】episode的采样率"""

    num_shards_per_epoch: int = int(1e5)
    """Number of shards to use for the dataset. reduce this number if vram is limited.
    【中文】用于数据集的分片数量。如果显存有限，请减少此数值"""
