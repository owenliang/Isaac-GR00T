import os

from gr00t.configs.base_config import get_default_config
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import torch
from gr00t.data.dataset import lerobot_episode_loader
from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.stats import generate_rel_stats
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
    MessageType,
    VLAStepData
)
from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.model.gr00t_n1d6.processing_gr00t_n1d6 import Gr00tN1d6Processor
# SO-ARMS101数据集：https://huggingface.co/datasets/lerobot/svla_so101_pickplace
# 先把v3数据集转换成v2数据集
# python scripts/lerobot_conversion/convert_v3_to_v2.py --repo-id "lerobot/svla_so101_pickplace" --root ./demo_data/
# 再手动编辑modality.json描述原始数据集，再手动编辑modality_config描述训练样本

# 1,加载modality_config配置
modality_config={
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=["up", "side"],
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "shoulder_pan.pos",
            "shoulder_lift.pos",
            "elbow_flex.pos",
            "wrist_flex.pos",
            "wrist_roll.pos",
            "gripper.pos"
        ],
    ),
    "action": ModalityConfig(
        delta_indices=list(range(0, 16)),
        modality_keys=[
            "shoulder_pan.pos",
            "shoulder_lift.pos",
            "elbow_flex.pos",
            "wrist_flex.pos",
            "wrist_roll.pos",
            "gripper.pos"
        ],
        action_configs=[
            # shoulder_pan.pos
            ActionConfig(
                rep=ActionRepresentation.RELATIVE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            # shoulder_lift.pos
            ActionConfig(
                rep=ActionRepresentation.RELATIVE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            # elbow_flex.pos
            ActionConfig(
                rep=ActionRepresentation.RELATIVE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            # wrist_flex.pos
            ActionConfig(
                rep=ActionRepresentation.RELATIVE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            # wrist_roll.pos
            ActionConfig(
                rep=ActionRepresentation.RELATIVE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            # gripper.pos
            ActionConfig(
                rep=ActionRepresentation.RELATIVE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
        ]
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.action.task_description"],
    ),
}
# 注册新本体
register_modality_config(modality_config, EmbodimentTag.NEW_EMBODIMENT)
# 生成相对统计
generate_rel_stats('./demo_data/svla_so101_pickplace',EmbodimentTag.NEW_EMBODIMENT)
# 加载原始dataframe
lerobot_episode_loader=LeRobotEpisodeLoader(
    './demo_data/svla_so101_pickplace',
    modality_config,
    video_backend='decord'
)
print(f'Number of episodes: {len(lerobot_episode_loader)}')
print(f'episode 0: {lerobot_episode_loader[0]}')
# 取1个episode中的1个step，加工成1条样本（也就是合成action horizon)
episode0_df = lerobot_episode_loader[0]
vla_step_data = extract_step_data(episode_data=episode0_df,step_index=100,modality_configs=modality_config,embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
print(f'vla_step_data: {vla_step_data}')

# 初始化processor（modality_configs键必须为字符串）
statistics = lerobot_episode_loader.get_dataset_statistics()
config = get_default_config()

processor = Gr00tN1d6Processor(
    modality_configs={EmbodimentTag.NEW_EMBODIMENT.value: modality_config},  
    statistics={EmbodimentTag.NEW_EMBODIMENT.value: statistics},  
    image_crop_size=config.model.image_crop_size,
    image_target_size=config.model.image_target_size,
    random_rotation_angle=config.model.random_rotation_angle,
    color_jitter_params=config.model.color_jitter_params,
    model_name=config.model.model_name,
    model_type=config.model.backbone_model_type,
    formalize_language=config.model.formalize_language,
    max_state_dim=config.model.max_state_dim,
    max_action_dim=config.model.max_action_dim,
    apply_sincos_state_encoding=config.model.apply_sincos_state_encoding,
    max_action_horizon=config.model.action_horizon,
    use_albumentations=config.model.use_albumentations_transforms,
    shortest_image_edge=config.model.shortest_image_edge,
    crop_fraction=config.model.crop_fraction,
    use_relative_action=config.model.use_relative_action,
)
processor.eval()
# 交给processor转relative action、做state/action normalize
messages = [{"type": MessageType.EPISODE_STEP.value, "content": vla_step_data}]
processed_data = processor(messages)
print(processed_data)
