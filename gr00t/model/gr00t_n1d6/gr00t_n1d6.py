# Gr00t N1.6 模型实现
# 【中文】核心架构：视觉-语言基础模型 (EagleBackbone) + 流匹配扩散动作头 (Gr00tN1d6ActionHead)
# 【中文】用于从多模态观测（图像、文本指令、机器人状态）预测未来动作序列

from typing import Tuple

from gr00t.configs.model.gr00t_n1d6 import Gr00tN1d6Config
from gr00t.model.modules.dit import AlternateVLDiT, DiT
from gr00t.model.modules.eagle_backbone import EagleBackbone
from gr00t.model.modules.embodiment_conditioned_mlp import (
    CategorySpecificMLP,
    MultiEmbodimentActionEncoder,
)
import torch
from torch import nn
from torch.distributions import Beta
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, PreTrainedModel
from transformers.feature_extraction_utils import BatchFeature
import tree


class Gr00tN1d6ActionHead(nn.Module):
    """Action head component for flow matching diffusion policy.
    
    【中文】动作头：基于流匹配扩散策略的动作预测模块。
    
    【中文】核心组件：
    1. state_encoder: 将机器人状态编码为特征向量（支持多具身形态）
    2. action_encoder: 将噪声动作轨迹和时间步编码为特征
    3. model (DiT/AlternateVLDiT): 扩散Transformer，融合视觉、语言、状态、动作特征
    4. action_decoder: 将DiT输出解码为动作空间
    
    【中文】训练时：通过流匹配学习从噪声到真实动作的速度场
    【中文】推理时：通过ODE积分从噪声逐步去噪得到动作轨迹
    """

    supports_gradient_checkpointing = True

    def __init__(self, config: Gr00tN1d6Config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size  # 【中文】DiT的隐藏层维度（默认1536）
        self.input_embedding_dim = config.input_embedding_dim  # 【中文】状态/动作编码后的特征维度（默认1536）

        # ==================== 初始化扩散Transformer模型 ====================
        # 【选择DiT变体】
        # - DiT: 标准的扩散Transformer，将VL特征作为cross-attention的条件
        # - AlternateVLDiT: 交替注意力DiT，每隔N个block才做一次cross-attention，降低计算量
        # 【中文】两种DiT都是32层Transformer，用于学习从噪声到真实动作的速度场
        if config.use_alternate_vl_dit:
            self.model = AlternateVLDiT(
                **config.diffusion_model_cfg,
                cross_attention_dim=config.backbone_embedding_dim,
                attend_text_every_n_blocks=config.attend_text_every_n_blocks,
            )
            print("Using AlternateVLDiT for diffusion model")
        else:
            self.model = DiT(
                **config.diffusion_model_cfg, cross_attention_dim=config.backbone_embedding_dim
            )
            print("Using DiT for diffusion model")
        
        # ==================== 动作空间配置 ====================
        self.action_dim = config.max_action_dim  # 【中文】动作维度上限（默认29），不同具身形态padding到此维度
        self.action_horizon = config.action_horizon  # 【中文】预测未来动作步数（默认16），即一次预测未来16步动作
        self.num_inference_timesteps = config.num_inference_timesteps  # 【中文】推理时ODE积分步数（默认10），越多越精确但越慢

        # ==================== 编码器与解码器 ====================
        # 【中文】状态编码器：根据embodiment_id选择对应的MLP
        # 【作用】将不同具身形态的状态（如gr1的30维、panda的8维）统一编码为input_embedding_dim维
        # 【CategorySpecificMLP】内部维护max_num_embodiments个MLP，通过embodiment_id索引选择
        self.state_encoder = CategorySpecificMLP(
            num_categories=config.max_num_embodiments, # 最大支持的本体类型数量
            input_dim=config.max_state_dim, # 输入state的dim，应该是29
            hidden_dim=self.hidden_size, # 因为连续两次MLP，先升到1024
            output_dim=self.input_embedding_dim, # 再拉到1536
        )
        
        # 【中文】动作编码器：编码噪声动作+时间步，支持多具身形态
        # 【作用】将噪声动作轨迹[B, action_horizon, action_dim]和时间步t编码为特征
        # 【包含】时间步embedding（通过sinusoidal encoding）+ 具身形态特定的动作MLP
        self.action_encoder = MultiEmbodimentActionEncoder(
            action_dim=self.action_dim,
            hidden_size=self.input_embedding_dim,
            num_embodiments=config.max_num_embodiments,
        )
        
        # 【中文】动作解码器：从DiT输出映射回动作空间
        # 【作用】将DiT的隐藏状态[B, seq_len, hidden_size]解码为速度场[B, action_horizon, action_dim]
        self.action_decoder = CategorySpecificMLP(
            num_categories=config.max_num_embodiments,
            input_dim=self.hidden_size,
            hidden_dim=self.hidden_size,
            output_dim=self.action_dim,
        )

        # VL LayerNorm：可选的视觉-语言特征归一化
        self.vlln = (

            nn.LayerNorm(config.backbone_embedding_dim) if config.use_vlln else nn.Identity()  # 对VLM的输出进行层归一化（可选）
        )

        # 可选：位置编码（用于动作序列）
        if config.add_pos_embed:
            self.position_embedding = nn.Embedding(config.max_seq_len, self.input_embedding_dim) #可学习的Dit位置向量
            nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)

        # 状态dropout参数：训练时按样本随机“屏蔽”整段状态，用于增强对缺失/不可靠状态的鲁棒性
        # 【中文】当被drop的样本，其state特征整体替换为可训练的mask_token，表示“无状态信息”，模型被迫依赖视觉/语言也能给出动作
        self.state_dropout_prob = config.state_dropout_prob
        self.mask_token = (
            nn.Parameter(0.02 * torch.randn(1, 1, self.input_embedding_dim))
            if self.state_dropout_prob > 0
            else None
        )

        # 状态噪声注入：训练时可选的高斯噪声
        self.state_additive_noise_scale = config.state_additive_noise_scale

        # 流匹配时间采样分布：Beta分布
        self.beta_dist = Beta(config.noise_beta_alpha, config.noise_beta_beta)
        self.num_timestep_buckets = config.num_timestep_buckets
        # 设置可训练参数（支持部分冻结）
        self.set_trainable_parameters(
            config.tune_projector, config.tune_diffusion_model, config.tune_vlln
        )

    def set_trainable_parameters(
        self, tune_projector: bool, tune_diffusion_model: bool, tune_vlln: bool
    ):
        """
        控制动作头各模块的可训练性。
        tune_projector: 是否训练状态/动作编解码器
        tune_diffusion_model: 是否训练DiT模型
        tune_vlln: 是否训练VL LayerNorm
        """
        self.tune_projector = tune_projector
        self.tune_diffusion_model = tune_diffusion_model
        self.tune_vlln = tune_vlln
        # 默认所有参数可训练
        for p in self.parameters():
            p.requires_grad = True
        # 根据配置冻结特定模块
        if not tune_projector: 
            # 冻结encoder,decoder
            self.state_encoder.requires_grad_(False)
            self.action_encoder.requires_grad_(False)
            self.action_decoder.requires_grad_(False)
            # 冻结可学习位置向量
            if self.config.add_pos_embed: 
                self.position_embedding.requires_grad_(False)
            # 冻结状态dropout的mask_token
            if self.state_dropout_prob > 0:
                self.mask_token.requires_grad_(False)
        # 冻结DiT模型自身权重
        if not tune_diffusion_model:
            self.model.requires_grad_(False)
        # 冻结VLM LayerNorm adapter
        if not tune_vlln:
            self.vlln.requires_grad_(False)
        print(f"Tune action head projector: {self.tune_projector}")
        print(f"Tune action head diffusion model: {self.tune_diffusion_model}")
        print(f"Tune action head vlln: {self.tune_vlln}")
        # Check if any parameters are still trainable. If not, print a warning.


        # 如果 project, diffusion_model 和 vlln 都被冻结，Action head 内部将几乎没有可训练参数。
        # 但整个 Gr00tN1d6 模型可能仍在训练 VLM Backbone (如 LLM 或 Vision Tower)。
        # 此处仅检查并打印 Action head 内部残余的可训练参数（如 norm 层的 affine 参数等，取决于具体实现）。
        if not any(p.requires_grad for p in self.parameters()):
            print("Warning: No action head trainable parameters found.")
        else:
            if not tune_projector and not tune_diffusion_model and not tune_vlln:
                for name, p in self.named_parameters():
                    if p.requires_grad:
                        print(f"Action head internal trainable parameter (unexpected): {name}")

    def set_frozen_modules_to_eval_mode(self):
        """
        将冻结的模块设为eval模式。
        HuggingFace会在每个training_step调用model.train()，
        需要手动将冻结模块设回eval以保持dropout/batchnorm的正确行为。
        """
        if self.training:
            if not self.tune_projector:
                self.state_encoder.eval()
                self.action_encoder.eval()
                self.action_decoder.eval()
                if self.config.add_pos_embed:
                    self.position_embedding.eval()
            if not self.tune_diffusion_model:
                self.model.eval()

    def sample_time(self, batch_size, device, dtype):
        """从Beta分布采样流匹配时间步 t，范围[0, noise_s]。"""
        sample = self.beta_dist.sample([batch_size]).to(device, dtype=dtype)
        sample = (1 - sample) * self.config.noise_s
        return sample

    def process_backbone_output(self, backbone_output: BatchFeature) -> BatchFeature:
        """对backbone特征应用可选的LayerNorm。"""
        backbone_features = backbone_output["backbone_features"]

        # backbone_features 原始形状通常为 [Batch, Seq_Len, Hidden_Dim]
        # input_ids 输入形状为 [Batch, L]
        # 这里的 LayerNorm 作用在最后一个维度 Hidden_Dim 上，用于对 VLM 输出的特征进行归一化稳定训练
        backbone_features = self.vlln(backbone_features) 
        backbone_output["backbone_features"] = backbone_features
        return backbone_output

    def forward(self, backbone_output: BatchFeature, action_input: BatchFeature) -> BatchFeature:
        """
        训练前向传播：通过流匹配学习从噪声到真实动作的速度场。

        核心流程：
        1. 编码状态特征，应用dropout和噪声增强
        2. 采样时间步 t ~ Beta，构造噪声轨迹：noisy_trajectory = (1-t)*noise + t*action
        3. 目标速度场：velocity = action - noise
        4. 通过DiT预测速度场，计算MSE loss

        Args:
            backbone_output: 视觉-语言特征 [B, seq_len, D]
            action_input: 状态、动作、embodiment_id等

        Returns:
            包含loss的BatchFeature
        """
        # 将冻结模块设为eval模式（某些层是特殊一点，比如dropout，batchnorm啥的需要在推理的时候行为正确）
        self.set_frozen_modules_to_eval_mode()

        # ==================== 步骤 1: 处理 Backbone 输出 ====================
        # 【输入】backbone_output.backbone_features: [B, vl_seq_len, backbone_embedding_dim=2048]
        # 【模块】self.vlln (LayerNorm 或 Identity)
        # 【输出】backbone_output.backbone_features: [B, vl_seq_len, backbone_embedding_dim=2048]
        backbone_output = self.process_backbone_output(backbone_output)

        # 获取视觉-语言特征
        # vl_embeds: [B, vl_seq_len, backbone_embedding_dim=2048]
        vl_embeds = backbone_output.backbone_features
        device = vl_embeds.device

        # 获取具身形态ID
        # embodiment_id: [B] - 整数张量，取值范围 [0, max_num_embodiments-1]
        embodiment_id = action_input.embodiment_id

        # ==================== 步骤 2: 编码状态特征 ====================
        # 【输入】action_input.state: [B, T_state, max_state_dim=29]
        #        embodiment_id: [B]
        # 【模块】self.state_encoder (CategorySpecificMLP)
        #        - 根据 embodiment_id 选择对应的 MLP
        #        - MLP: max_state_dim=29 → hidden_size=1024 → input_embedding_dim=1536
        # 【输出】state_features: [B, T_state, input_embedding_dim=1536]
        state_features = self.state_encoder(action_input.state, embodiment_id)
        
        # ==================== 步骤 2.1: 状态特征增强 ====================
        # 【应用状态dropout】以 state_dropout_prob 的概率，将某个样本的整段状态特征替换为 mask_token
        # 【作用】显式表示"状态缺失/不可靠"，增强模型对视觉/语言信息的依赖
        # 【形状保持】state_features: [B, T_state, input_embedding_dim=1536]
        if self.state_dropout_prob > 0:
            do_dropout = (  # [B] - bool 张量
                torch.rand(state_features.shape[0], device=state_features.device) # 按batch维度丢弃state
                < self.state_dropout_prob
            )
            do_dropout = do_dropout[:, None, None].to(dtype=state_features.dtype)  # [B, 1, 1] - 增加2维，广播用
            # self.mask_token: [1, 1, input_embedding_dim=1536]
            state_features = state_features * (1 - do_dropout) + self.mask_token * do_dropout # 部分样例的state被丢弃换成了mask emb（可训练的）
        
        # 【添加高斯噪声】增强对噪声状态输入的鲁棒性
        # 【形状保持】state_features: [B, T_state, input_embedding_dim=1536]
        if self.training and self.state_additive_noise_scale > 0:
            print(
                f"Adding Gaussian noise to state features with scale {self.state_additive_noise_scale}"
            )
            noise = torch.randn_like(state_features) * self.state_additive_noise_scale 
            state_features = state_features + noise # 随机给state搞了点高斯噪音，难为一下模型

        # ==================== 步骤 3: 流匹配核心 - 构造噪声轨迹 ====================
        # 【获取真实动作】
        # actions: [B, action_horizon=16, max_action_dim=29]
        actions = action_input.action
        
        # 【采样噪声】从标准正态分布N(0,I)采样初始噪声
        # 【作用】作为流匹配的起点，代表完全随机的动作
        # noise: [B, action_horizon=16, max_action_dim=29]
        noise = torch.randn(actions.shape, device=actions.device, dtype=actions.dtype)
        
        # 【采样时间步t】从Beta分布采样，t∈[0, noise_s]，其中noise_s通常为0.999，时刻t属于0~1，符合flow matching要求
        # 【原理】t=0时完全是噪声，t=1时完全是真实动作
        # 【Beta分布作用】控制训练时更关注哪个阶段（噪声阶段或接近真实动作阶段）
        # 【模块】self.beta_dist.sample + 缩放
        # t: [B] → 广播后 [B, 1, 1]
        t = self.sample_time(actions.shape[0], device=actions.device, dtype=actions.dtype)  # [B],每个样本随机选1个时刻t
        t = t[:, None, None]  # 广播形状 [B, 1, 1] 以匹配 [B, action_horizon, action_dim]

        # 【线性插值构造噪声轨迹】
        # noisy_trajectory = (1-t)·noise + t·action
        # - 当t=0: noisy_trajectory = noise（完全噪声）
        # - 当t=1: noisy_trajectory = action（真实动作）
        # - 当t∈(0,1): 介于噪声和真实动作之间的插值
        # 【几何意义】连接噪声和真实动作的直线路径上的某个点
        # noisy_trajectory: [B, action_horizon=16, max_action_dim=29]
        noisy_trajectory = (1 - t) * noise + t * actions # 给action加噪音到t时刻，作为DiT输入， 用来预测速度场
        
        # 【计算目标速度场】
        # velocity = dx/dt = ∂[(1-t)·noise + t·action]/∂t = action - noise
        # 【物理意义】从噪声到真实动作的"方向"和"速度"
        # 【训练目标】让模型学习预测这个速度场
        # velocity: [B, action_horizon=16, max_action_dim=29]
        velocity = actions - noise # 要拟合的目标，也就是方向，用最终清晰的action减去全混乱的Noise

        # 【时间步离散化】将连续时间t∈[0,1]映射到离散bucket索引
        # 【用途】用于时间步embedding（DiT需要知道当前处于哪个时间步）
        # 【示例】t=0.5, num_timestep_buckets=1000 → t_discretized=500
        # t_discretized: [B] - 长整型张量，取值范围 [0, num_timestep_buckets-1]
        t_discretized = (t[:, 0, 0] * self.num_timestep_buckets).long() # 将时间T离散化到0~1000的整数，看看下后面怎么用？
        
        # ==================== 步骤 4: 编码噪声动作轨迹 ====================
        # 【模块】self.action_encoder (MultiEmbodimentActionEncoder)
        # 【输入】noisy_trajectory: [B, action_horizon=16, max_action_dim=29]
        #        t_discretized: [B] - 时间步索引
        #        embodiment_id: [B] - 具身形态ID
        # 【内部处理】
        #   1. 时间步 embedding: t_discretized → sinusoidal encoding → [B, hidden_size]
        #   2. 根据 embodiment_id 选择对应的 MLP
        #   3. 动作 MLP: [B, action_horizon, max_action_dim=29] → [B, action_horizon, input_embedding_dim=1536]
        #   4. 时间步特征广播并相加: [B, 1, 1536] + [B, action_horizon, 1536]
        # 【输出】action_features: [B, action_horizon=16, input_embedding_dim=1536]

        # 离散时间T进去后embedding layer得到时间特征，然后与action特征做embedding维度concat，之后再MLP出来
        action_features = self.action_encoder(noisy_trajectory, t_discretized, embodiment_id) # 执行action的编码，编码后才输入DiT

        # 【可选：添加位置编码】为动作序列的每个时间步添加位置信息
        # 【作用】让DiT区分action_horizon内的不同时间步（t=0,1,...,15）
        # 【模块】self.position_embedding (nn.Embedding)
        # 【输入】pos_ids: [action_horizon=16]
        # 【输出】pos_embs: [1, action_horizon=16, input_embedding_dim=1536]
        # 【形状保持】action_features: [B, action_horizon=16, input_embedding_dim=1536]
        if self.config.add_pos_embed:
            pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)  # [action_horizon]
            # 这里走的可训练的embedding layer作为给action horizon级的位置信息
            pos_embs = self.position_embedding(pos_ids).unsqueeze(0)  # [1, action_horizon, input_embedding_dim=1536]
            action_features = action_features + pos_embs  # 把位置embedding直接加到action embedding上

        # ==================== 步骤 5: 拼接状态和动作特征 ====================
        # 【拼接操作】sa_embs = concat([state_features, action_features], dim=1)
        # 【输入】state_features: [B, T_state, input_embedding_dim=1536]
        #        action_features: [B, action_horizon=16, input_embedding_dim=1536]
        # 【输出】sa_embs: [B, T_state+action_horizon, input_embedding_dim=1536]
        #        例如：T_state=1 时，sa_embs: [B, 17, 1536]
        # 【语义】将当前状态和噪声动作轨迹作为DiT的输入序列
        sa_embs = torch.cat((state_features, action_features), dim=1)   # state emb 和每个action horizon的emb一样宽，大家一起拼成大序列，进入Diffusion transformer
        
        # 获取VL特征的attention mask（标记哪些token是有效的）
        # vl_attn_mask: [B, vl_seq_len] - 1表示有效token，0表示padding
        vl_attn_mask = backbone_output.backbone_attention_mask

        # ==================== 步骤 6: 通过DiT预测速度场 ====================
        # 【模块】self.model (AlternateVLDiT 或 DiT)
        # 【DiT的输入】
        # - hidden_states: sa_embs [B, sa_seq_len=T_state+action_horizon, input_embedding_dim=1536]
        # - encoder_hidden_states: vl_embeds [B, vl_seq_len, backbone_embedding_dim=2048] - 作为cross-attention的条件
        # - encoder_attention_mask: vl_attn_mask [B, vl_seq_len] - VL特征的mask
        # - timestep: t_discretized [B] - 时间步索引，用于adaptive layer norm
        # - image_mask (AlternateVLDiT专用): [B, num_images] - 图像补丁的有效性mask
        # - backbone_attention_mask (AlternateVLDiT专用): [B, vl_seq_len] - 用于区分文本和图像token
        # 
        # 【DiT的工作流程】(32层Transformer)
        # 1. 时间步条件注入: 通过 AdaLN 调制每层的 LayerNorm
        # 2. Self-attention on sa_embs: 融合状态和动作特征
        # 3. Cross-attention to vl_embeds: 引入视觉语言条件信息
        # 4. FFN (2层MLP): 1536 → 6144 → 1536 的非线性变换
        # 5. 最后通过 proj_out_2 投影: 1536 → hidden_size=1024
        # 
        # 【DiT内部维度】
        # - inner_dim = num_heads × head_dim = 32 × 48 = 1536
        # - 每层保持 1536 维
        # - 最后 proj_out_2: Linear(1536 → 1024)
        # 
        # 【输出】model_output: [B, sa_seq_len=T_state+action_horizon, hidden_size=1024]
        #        例如: [B, 17, 1024]
        if self.config.use_alternate_vl_dit:
            # 使用交替注意力DiT（每隔attend_text_every_n_blocks=2个block才做一次cross-attention）
            image_mask = backbone_output.image_mask  # [B, num_images]
            backbone_attention_mask = backbone_output.backbone_attention_mask  # [B, vl_seq_len]
            model_output, _ = self.model(
                hidden_states=sa_embs,  # [B, sa_seq_len, 1536]
                encoder_hidden_states=vl_embeds,  # [B, vl_seq_len, 2048]
                encoder_attention_mask=vl_attn_mask,  # [B, vl_seq_len]
                timestep=t_discretized,  # [B]
                return_all_hidden_states=True,
                image_mask=image_mask,  # [B, num_images]
                backbone_attention_mask=backbone_attention_mask,  # [B, vl_seq_len]
            )  # → model_output: [B, sa_seq_len, 1024]
        else:
            # 使用标准DiT（每个block都做cross-attention）
            # 走Diffusion Transformer
            model_output, _ = self.model(
                hidden_states=sa_embs,  # [B, sa_seq_len, 1536] 这是state+action输入
                encoder_hidden_states=vl_embeds,  # [B, vl_seq_len, 2048], 这是VLM输出，用于cross attn
                encoder_attention_mask=vl_attn_mask,  # [B, vl_seq_len] # # 这是VLM的padding mask，做cross attn注意力打分遮掩
                timestep=t_discretized,  # [B] # 离散时间信息还要继续用到DiT内部？
                return_all_hidden_states=True,
            )  # → model_output: [B, sa_seq_len, 1024]

        # ==================== 步骤 7: 解码并计算损失 ====================
        # 【模块】self.action_decoder (CategorySpecificMLP)
        # 【输入】model_output: [B, sa_seq_len=T_state+action_horizon, hidden_size=1024]
        #        embodiment_id: [B]
        # 【内部处理】
        #   1. 根据 embodiment_id 选择对应的 MLP
        #   2. MLP: hidden_size=1024 → hidden_size=1024 → max_action_dim=29
        # 【输出】pred: [B, sa_seq_len, max_action_dim=29]
        pred = self.action_decoder(model_output, embodiment_id)  # 将DiT的输出解码回action的维度29
        
        # 【提取动作部分】只取最后action_horizon个时间步（对应动作特征部分）
        # 前面的是状态特征部分，不需要预测
        # pred_actions: [B, action_horizon=16, max_action_dim=29]
        pred_actions = pred[:, -actions.shape[1] :]  # 切片: [:, -action_horizon:, :]  DiT输出左侧的state hidden仍掉

        # 【计算MSE损失】预测的速度场 vs 真实速度场
        # 【输入】pred_actions: [B, action_horizon=16, max_action_dim=29]
        #        velocity: [B, action_horizon=16, max_action_dim=29]
        # 【输出】action_loss (before mask): [B, action_horizon=16, max_action_dim=29]
        # 【应用action_mask】只计算有效动作维度和时间步的损失
        # - action_mask[b,t,d]=1: 有效动作，计入损失
        # - action_mask[b,t,d]=0: padding，不计入损失
        # action_mask: [B, action_horizon=16, max_action_dim=29]
        action_mask = action_input.action_mask
        action_loss = F.mse_loss(pred_actions, velocity, reduction="none") * action_mask   # 计算element-wise mse损失，这时候每个action horizon内的padding位置需要*0抹掉
        # action_loss: [B, action_horizon=16, max_action_dim=29]
        
        # 【平均损失】对所有有效位置求平均
        # +1e-6防止除零（当所有mask都是0时）
        # loss: 标量 Tensor[()]
        loss = action_loss.sum() / (action_mask.sum() + 1e-6) # 求element-wise mse损失的平均值

        # 【返回】包含损失和中间特征的字典
        return {
            "loss": loss,  # 标量损失 Tensor[()]，用于反向传播
            "action_loss": action_loss,  # [B, action_horizon=16, max_action_dim=29]，详细的逐元素损失
            "action_mask": action_mask,  # [B, action_horizon=16, max_action_dim=29]，有效位置标记
            "backbone_features": vl_embeds,  # [B, vl_seq_len, backbone_embedding_dim=2048]，视觉语言特征
            "state_features": state_features,  # [B, T_state, input_embedding_dim=1536]，状态特征
        }

    def _encode_features(
        self, backbone_output: BatchFeature, action_input: BatchFeature
    ) -> BatchFeature:
        """
        编码backbone和状态特征（用于推理）。
        返回backbone_features和state_features。
        """
        backbone_output = self.process_backbone_output(backbone_output)

        # 获取VL特征和编码状态
        vl_embeds = backbone_output.backbone_features
        embodiment_id = action_input.embodiment_id

        # 编码状态
        state_features = self.state_encoder(action_input.state, embodiment_id)

        return BatchFeature(data={"backbone_features": vl_embeds, "state_features": state_features})

    @torch.no_grad()
    def get_action_with_features(
        self,
        backbone_features: torch.Tensor,
        state_features: torch.Tensor,
        embodiment_id: torch.Tensor,
        backbone_output: BatchFeature,
    ) -> BatchFeature:
        """
        推理时生成动作：通过ODE积分从噪声逐步去噪。
        
        【ODE求解原理】
        训练时学习了速度场v(x,t)，推理时通过ODE积分恢复动作：
        - ODE方程：dx/dt = v(x,t)
        - 初始条件：x(0) = noise ~ N(0,I)
        - 终点：x(1) = action（期望的真实动作）
        
        【欧拉积分法】
        采用最简单的一阶欧拉法数值求解ODE：
        x_{n+1} = x_n + dt · v(x_n, t_n)
        其中dt = 1/num_inference_timesteps
        
        【推理步数的影响】
        - num_inference_timesteps越大：积分越精确，动作质量越好，但速度越慢
        - num_inference_timesteps越小：速度快，但可能欠拟合
        - 典型值：训练时可用1000步，推理时用10-50步
        
        核心流程：
        1. 初始化动作为高斯噪声 x(0) ~ N(0,I)
        2. 迭代 num_inference_timesteps 次：
           - 编码当前动作 + 时间步
           - 通过DiT预测速度场 v(x_n, t_n)
           - 欧拉积分：x_{n+1} = x_n + dt · v(x_n, t_n)
        3. 返回最终去噪后的动作 x(1)
        """
        vl_embeds = backbone_features

        # ==================== 步骤 1: 初始化为噪声 ====================
        # 【采样初始噪声】x(0) ~ N(0, I)
        # 【形状】[B, action_horizon, action_dim]
        batch_size = vl_embeds.shape[0]
        device = vl_embeds.device
        actions = torch.randn(
            size=(batch_size, self.config.action_horizon, self.action_dim),
            dtype=vl_embeds.dtype,
            device=device,
        )

        # ==================== 步骤 2: ODE积分参数 ====================
        # 【积分步长】dt = 1 / num_inference_timesteps
        # 【示例】如果num_inference_timesteps=10，则dt=0.1
        #        从t=0经过10步到达t=1
        dt = 1.0 / self.num_inference_timesteps

        # ==================== 步骤 3: 迭代去噪 ====================
        # 【循环】对每个时间步t = 0/N, 1/N, 2/N, ..., (N-1)/N
        for t in range(self.num_inference_timesteps):
            # 【连续时间】t_cont ∈ [0, 1)
            t_cont = t / float(self.num_inference_timesteps)
            
            # 【离散化时间步】映射到bucket索引，用于时间步embedding
            # 【示例】t_cont=0.5, num_timestep_buckets=1000 → t_discretized=500
            t_discretized = int(t_cont * self.num_timestep_buckets)

            # ==================== 步骤 3.1: 编码当前动作轨迹 ====================
            # 【构造时间步张量】全batch使用相同的时间步
            timesteps_tensor = torch.full(
                size=(batch_size,), fill_value=t_discretized, device=device
            )
            # 【编码动作+时间步】action_features: [B, action_horizon, input_embedding_dim]
            action_features = self.action_encoder(actions, timesteps_tensor, embodiment_id)
            
            # 【添加位置编码】
            if self.config.add_pos_embed:
                pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
                pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
                action_features = action_features + pos_embs

            # ==================== 步骤 3.2: 拼接状态和动作特征 ====================
            # sa_embs: [B, T_state+action_horizon, input_embedding_dim]
            sa_embs = torch.cat((state_features, action_features), dim=1)

            # ==================== 步骤 3.3: 通过DiT预测速度场 ====================
            # 【输入】当前噪声动作 + 状态 + 视觉语言特征 + 时间步
            # 【输出】预测的速度场 v(x_t, t)
            if self.config.use_alternate_vl_dit:
                model_output = self.model(
                    hidden_states=sa_embs,
                    encoder_hidden_states=vl_embeds,
                    timestep=timesteps_tensor,
                    image_mask=backbone_output.image_mask,
                    backbone_attention_mask=backbone_output.backbone_attention_mask,
                )
            else:
                model_output = self.model(
                    hidden_states=sa_embs,
                    encoder_hidden_states=vl_embeds,
                    timestep=timesteps_tensor,
                )
            
            # 【解码为速度场】pred: [B, sa_seq_len, action_dim]
            pred = self.action_decoder(model_output, embodiment_id)
            # 【提取动作部分的速度场】
            pred_velocity = pred[:, -self.action_horizon :]

            # ==================== 步骤 3.4: 欧拉积分更新 ====================
            # 【更新公式】x_{t+dt} = x_t + dt · v(x_t, t)
            # 【物理意义】沿着速度场的方向移动一小步
            # 【效果】逐步从噪声"流动"到真实动作
            actions = actions + dt * pred_velocity
        
        # ==================== 步骤 4: 返回最终去噪后的动作 ====================
        # 【此时】actions已经从初始噪声经过ODE积分到达t=1附近，接近真实动作分布
        return BatchFeature(
            data={
                "action_pred": actions,  # [B, action_horizon, action_dim]
                "backbone_features": vl_embeds,
                "state_features": state_features,
            }
        )

    @torch.no_grad()
    def get_action(self, backbone_output: BatchFeature, action_input: BatchFeature) -> BatchFeature:
        """
        推理接口：封装_encode_features和get_action_with_features。
        直接调用此函数即可生成动作。
        """
        features = self._encode_features(backbone_output, action_input)
        return self.get_action_with_features(
            backbone_features=features.backbone_features,
            state_features=features.state_features,
            embodiment_id=action_input.embodiment_id,
            backbone_output=backbone_output,
        )

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype

    def prepare_input(self, batch: dict) -> BatchFeature:
        """Prepare input batch for the action head."""
        return BatchFeature(data=batch)


def get_backbone_cls(config: Gr00tN1d6Config):
    """根据模型名称选择backbone类（目前仅支持Eagle）。"""
    if "NVEagle" in config.model_name or "nvidia/Eagle" in config.model_name:
        return EagleBackbone
    else:
        raise ValueError(f"Unsupported model name: {config.model_name}")


class Gr00tN1d6(PreTrainedModel):
    """
    Gr00t N1.6 完整模型：视觉-语言-动作 (VLA) 模型。
    
    架构：
    - backbone (EagleBackbone): 视觉-语言基础模型，将图像+文本编码为特征
    - action_head (Gr00tN1d6ActionHead): 流匹配扩散动作头，预测动作序列
    - collator: 数据拼接器，处理多模态输入
    
    训练：forward() 接受包含 `vlm_content`、`state`、`action`、`embodiment_id` 等字段的字典，返回包含 loss 的 BatchFeature。
    推理：get_action() 接受不含 action 的输入字典，返回包含 `action_pred` 的 BatchFeature。

    示例（训练）::

        model = Gr00tN1d6(config)
        outputs = model.forward({
            "vlm_content": vlm_content,              # 文本+图像描述
            "state": state_tensor,                  # [B, state_dim]
            "action": action_tensor,                # [B, action_horizon, action_dim]
            "embodiment_id": embodiment_id_tensor,  # [B]
        })
        loss = outputs["loss"]                      # 标量 loss

    示例（推理）::

        actions = model.get_action({
            "vlm_content": vlm_content,              # 文本+图像描述
            "state": state_tensor,                  # [B, state_dim]
            "embodiment_id": embodiment_id_tensor,  # [B]
        })
        action_pred = actions["action_pred"]        # [B, action_horizon, action_dim]
    """

    config_class = Gr00tN1d6Config
    supports_gradient_checkpointing = True

    def __init__(
        self,
        config: Gr00tN1d6Config,
        transformers_loading_kwargs: dict = {"trust_remote_code": True},
    ):
        """
        初始化 Gr00t N1.6 模型。

        Args:
            config: 模型配置
            transformers_loading_kwargs: Transformers加载参数
        """
        super().__init__(config)
        self.config = config


        # 初始化backbone（视觉-语言编码器）
        # 【中文】EagleBackbone负责将图像序列和文本指令编码为多模态特征向量。
        # 【配置说明】
        # - model_name: 预训练模型路径（如 nvidia/Eagle-7B）
        # - tune_llm/tune_visual: 是否微调大语言模型/视觉编码器
        # - select_layer: 提取视觉特征的层索引
        # - tune_top_llm_layers: 仅微调LLM顶部的N层（节省内存）
        backbone_cls = get_backbone_cls(config)
        self.backbone = backbone_cls(
            model_name=config.model_name,
            tune_llm=config.tune_llm,
            tune_visual=config.tune_visual,
            select_layer=config.select_layer,
            reproject_vision=config.reproject_vision,
            use_flash_attention=config.use_flash_attention,
            load_bf16=config.load_bf16,
            tune_top_llm_layers=config.tune_top_llm_layers,
            trainable_params_fp32=config.backbone_trainable_params_fp32,
            transformers_loading_kwargs=transformers_loading_kwargs,
        )
        # 初始化动作头
        self.action_head = Gr00tN1d6ActionHead(config)
        from .processing_gr00t_n1d6 import Gr00tN1d6DataCollator

        # 初始化数据拼接器
        self.collator = Gr00tN1d6DataCollator(
            model_name=config.model_name,
            model_type=config.backbone_model_type,
            transformers_loading_kwargs=transformers_loading_kwargs,
        )


    def prepare_input(self, inputs: dict) -> Tuple[BatchFeature, BatchFeature]:
        """
        准备backbone和action_head的输入。
        处理vlm_content（如果存在），并将所有数据移动到正确的设备/精度。
        
        Args:
            inputs: 原始输入字典，例如：

                {
                    # --- 视觉-语言输入 (由 collator 从 vlm_content 转换而来) ---
                    "input_ids": torch.Tensor[B, L],               # 文本 token IDs
                    "attention_mask": torch.Tensor[B, L],          # 文本/图像序列 mask
                    "pixel_values": torch.Tensor[B, N, C, H, W],    # 图像像素值 (N个补丁/帧)
                    "image_mask": torch.Tensor[B, N],               # 图像补丁的有效性 mask (用于 AlternateVLDiT)
                    
                    # --- 具身智能状态与动作输入 ---
                    "state": torch.Tensor[B, state_dim],           # 机器人本体状态
                    "action": torch.Tensor[B, action_horizon, action_dim], # 真实动作序列
                    "embodiment_id": torch.Tensor[B],              # 具身形态 ID
                    ...
                }
        
        Returns:
            Tuple[BatchFeature, BatchFeature]: (backbone_inputs, action_inputs)，例如：
                backbone_inputs = BatchFeature({
                    "input_ids": torch.Tensor[B, L],
                    "attention_mask": torch.Tensor[B, L],
                    "pixel_values": torch.Tensor[B, N_img, C, H, W],
                    ...
                })
                action_inputs = BatchFeature({
                    "state": torch.Tensor[B, state_dim],
                    "action": torch.Tensor[B, action_horizon, action_dim],
                    "embodiment_id": torch.Tensor[B],
                    ...
                })
        """
        
        # 注：推理代码不使用collator，所以需要在这里处理
        if "vlm_content" in inputs: # 在推理时候，进来的是没经过collator处理的单条样本
            # 修复 n_envs > 1 的问题：处理所有环境的VLM内容
            vlm_content_list = inputs["vlm_content"]
            # Ensure vlm_content_list is always a list for consistent processing
            if not isinstance(vlm_content_list, list):
                vlm_content_list = [vlm_content_list]

            # 通过collator处理所有VLM内容，变成batch
            # 推理阶段：除了VLM输入的文字+图像，state保持不变，action由flow matching推理生成，不需要输入action 
            prep = self.collator([{"vlm_content": vlm} for vlm in vlm_content_list])["inputs"]
            inputs.pop("vlm_content")
            inputs.update(prep)


        # 这里的实现确实是为了适配 Transformers 的数据结构
        # 将输入字典直接封装为 BatchFeature，以便后续模块统一处理
        backbone_inputs = BatchFeature(data=inputs)
        action_inputs = BatchFeature(data=inputs)

        # 移动到正确的设备和精度（浮点数使bf16，整数不变）
        def to_device_with_dtype(x):
            if torch.is_floating_point(x):
                return x.to(self.device, dtype=self.dtype)
            else:
                return x.to(self.device)

        backbone_inputs = tree.map_structure(to_device_with_dtype, backbone_inputs)
        action_inputs = tree.map_structure(to_device_with_dtype, action_inputs)

        return backbone_inputs, action_inputs


    def forward(self, inputs: dict) -> BatchFeature:
        """
        训练前向传播：通过backbone编码，然后通过action_head计算loss。
        
        Args:
            inputs: 包含 Eagle 输入和动作输入的字典，通常由数据加载器生成并经过 collator 处理。
                完整样例 inputs 结构如下：
                {
                    # --- 视觉-语言输入 (由 collator 从 vlm_content 转换而来) ---
                    "input_ids": torch.Tensor[B, L],               # 文本 token IDs
                    "attention_mask": torch.Tensor[B, L],          # 文本/图像序列 mask
                    "pixel_values": torch.Tensor[B, N, C, H, W],    # 图像像素值 (N个补丁/帧)
                    "image_mask": torch.Tensor[B, N],               # 图像补丁的有效性 mask (用于 AlternateVLDiT)

                    "backbone_attention_mask": torch.Tensor[B, L], # 视觉-语言特征的注意力掩码 (取值通常与 attention_mask 相同，用于 DiT 的 cross-attention 阶段)
                    
                    # --- 具身智能状态与动作输入 ---
                    "state": torch.Tensor[B, state_dim],           # 机器人本体状态 (如关节角度、速度)
                    "action": torch.Tensor[B, T, action_dim],      # 真实动作序列 (T 为预测步长 action_horizon)
                    "action_mask": torch.Tensor[B, T, action_dim], # 动作掩码 (1表示有效动作位，0表示padding)
                    "embodiment_id": torch.Tensor[B],              # 具身形态 ID (用于区分不同机器人类型)
                    
                    # --- (可选) 原始 vlm_content，如果在 prepare_input 前未处理 ---
                    "vlm_content": list[dict],                     # 包含 "text", "images" 等原始多模态信息
                }
        Returns:
            BatchFeature: 包含损失和中间特征的 BatchFeature，例如：
                {
                    "loss": torch.Tensor[()],                      # 标量损失
                    "action_loss": torch.Tensor[B, action_horizon, action_dim],
                    "action_mask": torch.Tensor[B, action_horizon, action_dim],
                    "backbone_features": torch.Tensor[B, vl_seq_len, D],
                    "state_features": torch.Tensor[B, T_state, D],
                }
        """
        # 准备输入（将数据移动到正确的设备和精度）
        backbone_inputs, action_inputs = self.prepare_input(inputs)

        # ==================== 步骤 1: VLM Backbone 前向传播 ====================
        # 【模块】self.backbone (EagleBackbone)
        # 【输入】backbone_inputs: BatchFeature {
        #     "input_ids": [B, L],              # 文本 token IDs
        #     "attention_mask": [B, L],         # 文本/图像序列 mask
        #     "pixel_values": [B, N, C, H, W],   # 图像像素 (N个图像/帧)
        #     "image_mask": [B, N],              # 图像有效性 mask
        #     ...
        # }
        # 【内部处理】
        #   1. Vision Encoder: 将图像编码为视觉特征
        #   2. 将视觉特征插入到文本序列中（通过占位符位置）
        #   3. LLM: 自回归处理整个视觉-语言序列
        #   4. 提取 select_layer 层的隐藏状态作为特征
        # 【输出】backbone_outputs: BatchFeature {
        #     "backbone_features": [B, vl_seq_len, backbone_embedding_dim=2048],
        #     "backbone_attention_mask": [B, vl_seq_len],
        #     "image_mask": [B, N],
        # }
        backbone_outputs = self.backbone(backbone_inputs)
        
        # ==================== 步骤 2: Action Head 前向传播 ====================
        # 【模块】self.action_head (Gr00tN1d6ActionHead)
        # 【输入】backbone_outputs: 上面VLM的输出
        #        action_inputs: BatchFeature {
        #            "state": [B, T_state, max_state_dim=29],
        #            "action": [B, action_horizon=16, max_action_dim=29],
        #            "action_mask": [B, action_horizon=16, max_action_dim=29],
        #            "embodiment_id": [B],
        #        }
        # 【输出】action_outputs: BatchFeature {
        #     "loss": 标量 Tensor[()],
        #     "action_loss": [B, action_horizon=16, max_action_dim=29],
        #     "action_mask": [B, action_horizon=16, max_action_dim=29],
        #     "backbone_features": [B, vl_seq_len, backbone_embedding_dim=2048],
        #     "state_features": [B, T_state, input_embedding_dim=1536],
        # }
        action_outputs = self.action_head(backbone_outputs, action_inputs)

        return action_outputs

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
        # 准备输入
        backbone_inputs, action_inputs = self.prepare_input(inputs)

        # 通过backbone编码
        backbone_outputs = self.backbone(backbone_inputs)
        action_outputs = self.action_head.get_action(backbone_outputs, action_inputs)

        return action_outputs

    @property
    def device(self):
        """当前模型所在设备。

        返回示例：device(type="cuda", index=0)。
        """
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        """当前模型主参数的数据类型。

        返回示例：torch.bfloat16 或 torch.float16。
        """
        return next(iter(self.parameters())).dtype


# 将模型注册到HuggingFace（支持AutoModel.from_pretrained）
AutoConfig.register("Gr00tN1d6", Gr00tN1d6Config)
AutoModel.register(Gr00tN1d6Config, Gr00tN1d6)
