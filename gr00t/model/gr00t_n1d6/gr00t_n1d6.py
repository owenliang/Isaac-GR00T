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
        self.input_embedding_dim = config.input_embedding_dim  # 【中文】状态/动作编码后的特征维度（默认1024）

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
            num_categories=config.max_num_embodiments,
            input_dim=config.max_state_dim,
            hidden_dim=self.hidden_size,
            output_dim=self.input_embedding_dim,
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
            nn.LayerNorm(config.backbone_embedding_dim) if config.use_vlln else nn.Identity()
        )

        # 可选：位置编码（用于动作序列）
        if config.add_pos_embed:
            self.position_embedding = nn.Embedding(config.max_seq_len, self.input_embedding_dim)
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
            self.state_encoder.requires_grad_(False)
            self.action_encoder.requires_grad_(False)
            self.action_decoder.requires_grad_(False)
            if self.config.add_pos_embed:
                self.position_embedding.requires_grad_(False)
            if self.state_dropout_prob > 0:
                self.mask_token.requires_grad_(False)
        if not tune_diffusion_model:
            self.model.requires_grad_(False)
        if not tune_vlln:
            self.vlln.requires_grad_(False)
        print(f"Tune action head projector: {self.tune_projector}")
        print(f"Tune action head diffusion model: {self.tune_diffusion_model}")
        print(f"Tune action head vlln: {self.tune_vlln}")
        # Check if any parameters are still trainable. If not, print a warning.
        if not tune_projector and not tune_diffusion_model and not tune_vlln:
            for name, p in self.named_parameters():
                if p.requires_grad:
                    print(f"Action head trainable parameter: {name}")
        if not any(p.requires_grad for p in self.parameters()):
            print("Warning: No action head trainable parameters found.")

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
        # 将冻结模块设为eval模式
        self.set_frozen_modules_to_eval_mode()

        backbone_output = self.process_backbone_output(backbone_output)

        # 获取视觉-语言特征
        vl_embeds = backbone_output.backbone_features
        device = vl_embeds.device

        # 获取具身形态ID
        embodiment_id = action_input.embodiment_id

        # 编码状态
        state_features = self.state_encoder(action_input.state, embodiment_id)

        # 应用状态dropout：随机mask为mask_token
        # 【中文】以 state_dropout_prob 的概率，将某个样本的整段状态特征替换为同一个 mask_token 向量，显式表示“状态缺失/不可靠”
        if self.state_dropout_prob > 0:
            do_dropout = (
                torch.rand(state_features.shape[0], device=state_features.device)
                < self.state_dropout_prob
            )
            do_dropout = do_dropout[:, None, None].to(dtype=state_features.dtype)
            state_features = state_features * (1 - do_dropout) + self.mask_token * do_dropout

        # 添加高斯噪声到状态特征（增强鲁棒性）
        if self.training and self.state_additive_noise_scale > 0:
            print(
                f"Adding Gaussian noise to state features with scale {self.state_additive_noise_scale}"
            )
            noise = torch.randn_like(state_features) * self.state_additive_noise_scale
            state_features = state_features + noise

        # ==================== 步骤 3: 流匹配核心 - 构造噪声轨迹 ====================
        # 【获取真实动作】actions: [B, action_horizon, action_dim]
        actions = action_input.action
        
        # 【采样噪声】从标准正态分布N(0,I)采样初始噪声
        # 【作用】作为流匹配的起点，代表完全随机的动作
        noise = torch.randn(actions.shape, device=actions.device, dtype=actions.dtype)
        
        # 【采样时间步t】从Beta分布采样，t∈[0, noise_s]，其中noise_s通常为1
        # 【原理】t=0时完全是噪声，t=1时完全是真实动作
        # 【Beta分布作用】控制训练时更关注哪个阶段（噪声阶段或接近真实动作阶段）
        t = self.sample_time(actions.shape[0], device=actions.device, dtype=actions.dtype)
        t = t[:, None, None]  # 广播形状 (B,1,1) 以匹配 (B, action_horizon, action_dim)

        # 【线性插值构造噪声轨迹】
        # noisy_trajectory = (1-t)·noise + t·action
        # - 当t=0: noisy_trajectory = noise（完全噪声）
        # - 当t=1: noisy_trajectory = action（真实动作）
        # - 当t∈(0,1): 介于噪声和真实动作之间的插值
        # 【几何意义】连接噪声和真实动作的直线路径上的某个点
        noisy_trajectory = (1 - t) * noise + t * actions
        
        # 【计算目标速度场】
        # velocity = dx/dt = ∂[(1-t)·noise + t·action]/∂t = action - noise
        # 【物理意义】从噪声到真实动作的"方向"和"速度"
        # 【训练目标】让模型学习预测这个速度场
        velocity = actions - noise

        # 【时间步离散化】将连续时间t∈[0,1]映射到离散bucket索引
        # 【用途】用于时间步embedding（DiT需要知道当前处于哪个时间步）
        # 【示例】t=0.5, num_timestep_buckets=1000 → t_discretized=500
        t_discretized = (t[:, 0, 0] * self.num_timestep_buckets).long()
        
        # ==================== 步骤 4: 编码噪声动作轨迹 ====================
        # 【输入】noisy_trajectory + t_discretized + embodiment_id
        # 【输出】action_features: [B, action_horizon, input_embedding_dim]
        # 【包含】时间步embedding（sinusoidal）+ 具身形态特定的动作MLP
        action_features = self.action_encoder(noisy_trajectory, t_discretized, embodiment_id)

        # 【可选：添加位置编码】为动作序列的每个时间步添加位置信息
        # 【作用】让DiT区分action_horizon内的不同时间步（t=0,1,...,15）
        # 【实现】pos_embs: [1, action_horizon, input_embedding_dim]
        if self.config.add_pos_embed:
            pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
            pos_embs = self.position_embedding(pos_ids).unsqueeze(0)  # [1, T, D]
            action_features = action_features + pos_embs

        # ==================== 步骤 5: 拼接状态和动作特征 ====================
        # 【拼接】sa_embs = [state_features; action_features]
        # 【形状】state_features: [B, T_state, D], action_features: [B, action_horizon, D]
        #        → sa_embs: [B, T_state+action_horizon, D]
        # 【语义】将当前状态和噪声动作轨迹作为DiT的输入序列
        sa_embs = torch.cat((state_features, action_features), dim=1)
        
        # 获取VL特征的attention mask（标记哪些token是有效的）
        vl_attn_mask = backbone_output.backbone_attention_mask

        # ==================== 步骤 6: 通过DiT预测速度场 ====================
        # 【DiT的输入】
        # - hidden_states: sa_embs [B, sa_seq_len, D] - 状态+动作序列
        # - encoder_hidden_states: vl_embeds [B, vl_seq_len, D] - 视觉语言特征（作为cross-attention的条件）
        # - encoder_attention_mask: vl_attn_mask [B, vl_seq_len] - VL特征的mask
        # - timestep: t_discretized [B] - 时间步索引
        # 
        # 【DiT的工作流程】
        # 1. Self-attention on sa_embs: 融合状态和动作特征
        # 2. Cross-attention to vl_embeds: 引入视觉语言条件信息
        # 3. FFN: 非线性变换
        # 4. 重复32层
        # 
        # 【输出】model_output: [B, sa_seq_len, hidden_size]
        if self.config.use_alternate_vl_dit:
            # 使用交替注意力DiT（每隔N个block才做一次cross-attention）
            image_mask = backbone_output.image_mask
            backbone_attention_mask = backbone_output.backbone_attention_mask
            model_output, _ = self.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embeds,
                encoder_attention_mask=vl_attn_mask,
                timestep=t_discretized,
                return_all_hidden_states=True,
                image_mask=image_mask,
                backbone_attention_mask=backbone_attention_mask,
            )
        else:
            # 使用标准DiT（每个block都做cross-attention）
            model_output, _ = self.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embeds,
                encoder_attention_mask=vl_attn_mask,
                timestep=t_discretized,
                return_all_hidden_states=True,
            )

        # ==================== 步骤 7: 解码并计算损失 ====================
        # 【解码为动作空间】
        # 输入：model_output [B, sa_seq_len, hidden_size]
        # 输出：pred [B, sa_seq_len, action_dim]
        pred = self.action_decoder(model_output, embodiment_id)
        
        # 【提取动作部分】只取最后action_horizon个时间步（对应动作特征部分）
        # 前面的是状态特征部分，不需要预测
        pred_actions = pred[:, -actions.shape[1] :]  # [B, action_horizon, action_dim]

        # 【计算MSE损失】预测的速度场 vs 真实速度场
        # loss = ||pred_actions - velocity||²
        # 【应用action_mask】只计算有效动作维度和时间步的损失
        # - action_mask[t,d]=1: 有效动作，计入损失
        # - action_mask[t,d]=0: padding，不计入损失
        action_mask = action_input.action_mask
        action_loss = F.mse_loss(pred_actions, velocity, reduction="none") * action_mask
        
        # 【平均损失】对所有有效位置求平均
        # +1e-6防止除零（当所有mask都是0时）
        loss = action_loss.sum() / (action_mask.sum() + 1e-6)

        # 【返回】包含损失和中间特征的字典
        return {
            "loss": loss,  # 标量损失，用于反向传播
            "action_loss": action_loss,  # [B, action_horizon, action_dim]，详细的逐元素损失
            "action_mask": action_mask,  # [B, action_horizon, action_dim]，有效位置标记
            "backbone_features": vl_embeds,  # [B, vl_seq_len, D]，视觉语言特征
            "state_features": state_features,  # [B, T_state, D]，状态特征
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
    
    训练： forward() 返回 loss
    推理： get_action() 返回 action_pred
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
        """

        # 注：推理代码不使用collator，所以需要在这里处理
        if "vlm_content" in inputs:
            # 修复 n_envs > 1 的问题：处理所有环境的VLM内容
            vlm_content_list = inputs["vlm_content"]
            # Ensure vlm_content_list is always a list for consistent processing
            if not isinstance(vlm_content_list, list):
                vlm_content_list = [vlm_content_list]

            # 通过collator处理所有VLM内容
            prep = self.collator([{"vlm_content": vlm} for vlm in vlm_content_list])["inputs"]
            inputs.pop("vlm_content")
            inputs.update(prep)

        # 分别准备backbone和action_head的输入
        backbone_inputs = self.backbone.prepare_input(inputs)
        action_inputs = self.action_head.prepare_input(inputs)

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
            inputs: 包含Eagle输入和动作输入的字典

        Returns:
            包含loss的BatchFeature
        """
        # 准备输入
        backbone_inputs, action_inputs = self.prepare_input(inputs)
        backbone_outputs = self.backbone(backbone_inputs)
        action_outputs = self.action_head(backbone_outputs, action_inputs)

        return action_outputs

    def get_action(self, inputs: dict) -> BatchFeature:
        """
        推理接口：生成动作序列。
        返回action_pred: [B, action_horizon, action_dim]。
        """
        # 准备输入
        backbone_inputs, action_inputs = self.prepare_input(inputs)

        # 通过backbone编码
        backbone_outputs = self.backbone(backbone_inputs)
        action_outputs = self.action_head.get_action(backbone_outputs, action_inputs)

        return action_outputs

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype


# 将模型注册到HuggingFace（支持AutoModel.from_pretrained）
AutoConfig.register("Gr00tN1d6", Gr00tN1d6Config)
AutoModel.register(Gr00tN1d6Config, Gr00tN1d6)
