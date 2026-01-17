# Diffusion Transformer (DiT) 实现
# 用于 Gr00t N1.6 的扩散动作头，支持交叉注意力融合视觉-语言特征

from typing import Optional

from diffusers import ConfigMixin, ModelMixin
from diffusers.configuration_utils import register_to_config
from diffusers.models.attention import Attention, FeedForward
from diffusers.models.embeddings import SinusoidalPositionalEmbedding, TimestepEmbedding, Timesteps
import torch
from torch import nn
import torch.nn.functional as F


class TimestepEncoder(nn.Module):
    """
    时间步编码器：将离散时间步编码为连续嵌入向量。
    使用正弦位置编码 + MLP 投影。
    """
    def __init__(self, embedding_dim, compute_dtype=torch.float32):
        super().__init__()
        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=1)
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)

    def forward(self, timesteps):
        """
        Args:
            timesteps: 时间步索引张量，形状通常为 [B] 或 [B,]，取值范围 [0, num_timestep_buckets-1]。
        Returns:
            torch.Tensor: 时间步嵌入向量，形状 [B, embedding_dim]，其中 embedding_dim = self.timestep_embedder.time_embed_dim。
        """
        # 将整数时间步映射到正弦/余弦时间特征，再通过 MLP 投影到目标维度
        dtype = next(self.parameters()).dtype
        # time_proj: 将标量时间步编码为长度为 256 的向量 [B, 256]
        timesteps_proj = self.time_proj(timesteps).to(dtype)
        # timestep_embedder: 使用 MLP 将 256 维时间特征映射到目标维度 embedding_dim
        # timesteps_emb: [B, embedding_dim]
        timesteps_emb = self.timestep_embedder(timesteps_proj)
        return timesteps_emb

class AdaLayerNorm(nn.Module):
    """
    自适应LayerNorm：根据时间步嵌入调整归一化的scale和shift。
    用于条件扩散模型中的时间步调制。
    """
    def __init__(
        self,
        embedding_dim: int,
        norm_elementwise_affine: bool = False,
        norm_eps: float = 1e-5,
        chunk_dim: int = 0,
    ):
        super().__init__()
        self.chunk_dim = chunk_dim
        output_dim = embedding_dim * 2
        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim // 2, norm_eps, norm_elementwise_affine)

    def forward(
        self,
        x: torch.Tensor,
        temb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """将时间步嵌入 temb 映射为 scale/shift，对特征 x 做条件化 LayerNorm。

        Args:
            x: 输入特征张量，形状 [B, T, D] 或 [B, D]，其中 D = embedding_dim。
            temb: 时间步嵌入，形状 [B, embedding_dim]。
        Returns:
            torch.Tensor: 应用时间条件化后的特征，形状与 x 相同。
        """
        # 1. 通过 MLP 将时间嵌入映射到两倍通道数，用于生成 scale 和 shift
        temb = self.linear(self.silu(temb))  # [B, 2 * embedding_dim]
        scale, shift = temb.chunk(2, dim=1)  # 两个 [B, embedding_dim]
        # 2. 对 x 做 LayerNorm 后，按通道进行仿射变换：norm(x) * (1 + scale) + shift
        #    这里的 scale/shift 只依赖时间步，起到“时间条件化”的作用
        x = self.norm(x) * (1 + scale[:, None]) + shift[:, None]
        return x


class BasicTransformerBlock(nn.Module):
    """
    Transformer基础块：包含自注意力、交叉注意力（可选）、前馈网络。
    支持AdaLayerNorm用于时间步条件化。
    """
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        dropout=0.0,
        cross_attention_dim: Optional[int] = None,
        activation_fn: str = "geglu",
        attention_bias: bool = False,
        upcast_attention: bool = False,
        norm_elementwise_affine: bool = True,
        norm_type: str = "layer_norm",  # 'layer_norm', 'ada_norm', 'ada_norm_zero', 'ada_norm_single', 'ada_norm_continuous', 'layer_norm_i2vgen'
        norm_eps: float = 1e-5,
        final_dropout: bool = False,
        attention_type: str = "default",
        positional_embeddings: Optional[str] = None,
        num_positional_embeddings: Optional[int] = None,
        ff_inner_dim: Optional[int] = None,
        ff_bias: bool = True,
        attention_out_bias: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        self.dropout = dropout
        self.cross_attention_dim = cross_attention_dim
        self.activation_fn = activation_fn
        self.attention_bias = attention_bias
        self.norm_elementwise_affine = norm_elementwise_affine
        self.positional_embeddings = positional_embeddings
        self.num_positional_embeddings = num_positional_embeddings
        self.norm_type = norm_type

        if positional_embeddings and (num_positional_embeddings is None):
            raise ValueError(
                "If `positional_embedding` type is defined, `num_positition_embeddings` must also be defined."
            )

        if positional_embeddings == "sinusoidal":
            self.pos_embed = SinusoidalPositionalEmbedding(
                dim, max_seq_length=num_positional_embeddings
            )
        else:
            self.pos_embed = None

        # 定义3个主要模块：自注意力、交叉注意力（隐含）、前馈网络
        # 1. 自注意力
        if norm_type == "ada_norm":
            self.norm1 = AdaLayerNorm(dim)
        else:
            self.norm1 = nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine, eps=norm_eps)

        self.attn1 = Attention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
            cross_attention_dim=cross_attention_dim,
            upcast_attention=upcast_attention,
            out_bias=attention_out_bias,
        )

        # 3. 前馈网络
        self.norm3 = nn.LayerNorm(dim, norm_eps, norm_elementwise_affine)
        self.ff = FeedForward(
            dim,
            dropout=dropout,
            activation_fn=activation_fn,
            final_dropout=final_dropout,
            inner_dim=ff_inner_dim,
            bias=ff_bias,
        )
        if final_dropout:
            self.final_dropout = nn.Dropout(dropout)
        else:
            self.final_dropout = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        """单层 Transformer Block 的前向传播。

        Args:
            hidden_states: 主序列特征，形状 [B, T, D]。
            attention_mask: 自注意力的 mask，形状 [B, T]，1 为可见，0 为 padding。
            encoder_hidden_states: 交叉注意力的键值序列，形状 [B, S, D_enc]，如果为 None 则退化为纯自注意力。
            encoder_attention_mask: 交叉注意力的 mask，形状 [B, S]。
            temb: 时间步嵌入，形状 [B, D]，仅在 norm_type="ada_norm" 时使用。
        Returns:
            torch.Tensor: 更新后的序列特征，形状 [B, T, D]。
        """
        # 1. 归一化 + 时间步条件化（可选）
        if self.norm_type == "ada_norm":
            # AdaLayerNorm: 使用 temb 调制 LayerNorm 的 scale/shift
            norm_hidden_states = self.norm1(hidden_states, temb)
        else:
            norm_hidden_states = self.norm1(hidden_states)

        # 2. 位置编码（可选）
        if self.pos_embed is not None:
            # SinusoidalPositionalEmbedding: 为 token 添加绝对位置编码
            norm_hidden_states = self.pos_embed(norm_hidden_states)

        # 3. 自注意力或交叉注意力
        #    - 若 encoder_hidden_states 为 None: 退化为自注意力
        #    - 否则: 对 encoder_hidden_states 做 cross-attention
        attn_output = self.attn1(
            norm_hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=(
                encoder_attention_mask if encoder_hidden_states is not None else attention_mask
            ),
        )
        if self.final_dropout:
            attn_output = self.final_dropout(attn_output)

        # 残差连接
        hidden_states = attn_output + hidden_states
        if hidden_states.ndim == 4:
            # 某些 Attention 实现会临时扩展 batch 维度，需要还原回 [B, T, D]
            hidden_states = hidden_states.squeeze(1)

        # 4. 前馈网络 (FeedForward)
        norm_hidden_states = self.norm3(hidden_states)
        ff_output = self.ff(norm_hidden_states)

        # 残差连接
        hidden_states = ff_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)
        return hidden_states


class DiT(ModelMixin, ConfigMixin):
    """
    Diffusion Transformer：用于扩散模型的Transformer架构。
    
    关键特性：
    - 自注意力：处理state+action序列
    - 交叉注意力：融合vision-language特征
    - AdaLayerNorm：时间步条件化
    - 支持梯度检查点以节省显存
    """
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        num_attention_heads: int = 8,
        attention_head_dim: int = 64,
        output_dim: int = 26,
        num_layers: int = 12,
        dropout: float = 0.1,
        attention_bias: bool = True,
        activation_fn: str = "gelu-approximate",
        num_embeds_ada_norm: Optional[int] = 1000,
        upcast_attention: bool = False,
        norm_type: str = "ada_norm",
        norm_elementwise_affine: bool = False,
        norm_eps: float = 1e-5,
        max_num_positional_embeddings: int = 512,
        compute_dtype=torch.float32,
        final_dropout: bool = True,
        positional_embeddings: Optional[str] = "sinusoidal",
        interleave_self_attention=False,
        cross_attention_dim: Optional[int] = None,
    ):

        '''
        default_factory=lambda: {
            "positional_embeddings": None,
            "num_layers": 32,  # 32 layers instead of 16
            "num_attention_heads": 32, # 32头
            "attention_head_dim": 48,
            "norm_type": "ada_norm",
            "dropout": 0.2,
            "final_dropout": True,
            "output_dim": 1024,
            "interleave_self_attention": True, # 默认layer交替cross attn
        }
        '''
        super().__init__()

        # 设置注意力头维度和内部特征维度
        self.attention_head_dim = attention_head_dim
         # 多头，seq中每个hidden都是1536，标准的多头实现
        self.inner_dim = self.config.num_attention_heads * self.config.attention_head_dim
        # 默认关闭梯度检查点，可在训练时开启以节省显存
        self.gradient_checkpointing = False

        # 时间步编码器：将扩散过程的时间步索引转换为连续的嵌入向量
        self.timestep_encoder = TimestepEncoder(
            embedding_dim=self.inner_dim, compute_dtype=self.compute_dtype
        )

        # 构建 Transformer 块列表
        all_blocks = []
        for idx in range(self.config.num_layers):
            # 如果启用了 interleave_self_attention，奇数层将只使用自注意力
            use_self_attn = idx % 2 == 1 and interleave_self_attention
            # 只有在非纯自注意力层才设置交叉注意力维度
            curr_cross_attention_dim = cross_attention_dim if not use_self_attn else None

            all_blocks += [
                BasicTransformerBlock(
                    self.inner_dim,
                    self.config.num_attention_heads,
                    self.config.attention_head_dim,
                    dropout=self.config.dropout,
                    activation_fn=self.config.activation_fn,
                    attention_bias=self.config.attention_bias,
                    upcast_attention=self.config.upcast_attention,
                    norm_type=norm_type,
                    norm_elementwise_affine=self.config.norm_elementwise_affine,
                    norm_eps=self.config.norm_eps,
                    positional_embeddings=positional_embeddings,
                    num_positional_embeddings=self.config.max_num_positional_embeddings,
                    final_dropout=final_dropout,
                    cross_attention_dim=curr_cross_attention_dim,
                )
            ]
        # 使用 ModuleList 管理所有块，确保参数被正确注册
        self.transformer_blocks = nn.ModuleList(all_blocks)

        # 输出层定义
        # 1. 最后的归一化层，不使用 elementwise_affine，由后面的投影层处理
        self.norm_out = nn.LayerNorm(self.inner_dim, elementwise_affine=False, eps=1e-6)
        # 2. 投影层 1：用于处理时间步条件 (AdaLN)，生成 scale 和 shift
        self.proj_out_1 = nn.Linear(self.inner_dim, 2 * self.inner_dim)
        # 3. 投影层 2：将特征映射到最终的输出动作维度
        self.proj_out_2 = nn.Linear(self.inner_dim, self.output_dim)
        
        # 打印模型总参数量
        print(
            "Total number of DiT parameters: ",
            sum(p.numel() for p in self.parameters() if p.requires_grad), # 可训练参数
        )



    def forward(
        self,
        hidden_states: torch.Tensor,  # Shape: (B, sa_seq_len, 1536), sa_seq_len 为状态+动作序列长度
        encoder_hidden_states: torch.Tensor,  # Shape: (B, vl_seq_len, 2048), vl_seq_len 为视觉语言特征序列长度
        timestep: Optional[torch.LongTensor] = None, # Shape: (B,), 离散扩散时间步
        encoder_attention_mask: Optional[torch.Tensor] = None, # Shape: (B, vl_seq_len), 视觉语言特征的 mask
        return_all_hidden_states: bool = False,
    ):
        """DiT 主干的前向传播。

        Args:
            hidden_states: 主序列特征（state+action），形状 [B, sa_seq_len, 1536]。
            encoder_hidden_states: 条件序列（视觉-语言特征），形状 [B, vl_seq_len, 2048]。
            timestep: 离散时间步索引，形状 [B]。
            encoder_attention_mask: 条件序列的 mask，形状 [B, vl_seq_len]。
            return_all_hidden_states: 是否返回每一层的隐藏状态列表。
        Returns:
            torch.Tensor 或 (torch.Tensor, list[torch.Tensor]):
                - 输出张量形状 [B, sa_seq_len, output_dim]，通常为预测的动作或噪声。
        """
        # 1. 时间步编码：将标量 timestep 编码为 temb 向量 [B, 1536]
        temb = self.timestep_encoder(timestep)

        # 2. 准备输入张量，确保内存连续，便于后续高效计算
        # 这是state+action
        hidden_states = hidden_states.contiguous()  # state+action features, [B, sa_seq_len, 1536]
        # 这是vlm的hidden
        encoder_hidden_states = encoder_hidden_states.contiguous()  # [B, vl_seq_len, 2048]

        all_hidden_states = [hidden_states]

        # 3. 依次通过多个 Transformer Block
        for idx, block in enumerate(self.transformer_blocks):
            if idx % 2 == 1 and self.config.interleave_self_attention:
                # 只做自注意力（不看视觉-语言条件）
                hidden_states = block(  # 自注意力，所以不需要forward输入VLM hidden
                    hidden_states,
                    attention_mask=None,  # 没有自注意力的因果 mask：这里学习的是整条轨迹的速度场，每个时间步可以看见全局action序列（非自回归）
                    encoder_hidden_states=None,
                    encoder_attention_mask=None,
                    temb=temb,
                )
            else:
                # 带条件的 cross-attention（对 encoder_hidden_states 做注意力）
                # 注意：这里 encoder_attention_mask 当前仍为 None，实际的 VL mask 在上游处理
                hidden_states = block(
                    hidden_states,
                    attention_mask=None,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=None,  # 这里没有显式padding mask，依赖上游VLM将padding token的hidden置为近似0向量，相当于"看见也没信息"的近似屏蔽
                    temb=temb,
                )
            all_hidden_states.append(hidden_states)


        # 4. 输出层处理：采用 AdaLN-Zero 思想，使用时间步嵌入 temb 对最终特征进行条件化调制
        # -------------------------------------------------------------------------
        # 为什么要用 shift 和 scale 而不直接用 MLP？
        # 1) 条件化表达：扩散模型需要在不同时间步 t 下预测不同的分布。MLP 是静态映射，而 shift/scale
        #    能根据 t 动态调整特征的均值和方差，实现对特征分布的精确“引导”(Conditioning)。
        # 2) 结构化先验：LayerNorm 本身就有缩放和平移操作。AdaLN 通过预测这些参数，让模型学习如何
        #    根据时间步“归一化”特征。研究表明，这种调制比简单的特征拼接或相加在扩散模型中更有效。
        
        # [B, 1536] -> SiLU 激活 -> 线性层 [B, 3072]
        # self.proj_out_1 会生成一个两倍维度的向量，用于提取平移(shift)和缩放(scale)参数
        conditioning = self.proj_out_1(F.silu(temb)) 
        
        # .chunk(2, dim=1) 的作用：
        # 将维度为 [B, 3072] 的张量在第 1 维（特征维）切分成 2 块，每块形状为 [B, 1536]
        # 第一块作为 shift (平移量)，第二块作为 scale (缩放因子)
        shift, scale = conditioning.chunk(2, dim=1) 

        # 对 hidden_states 应用条件化归一化：
        # 1. self.norm_out(hidden_states) 将特征归一化到均值 0，方差 1
        # 2. (1 + scale[:, None])：对归一化后的特征进行缩放。使用 (1 + scale) 是为了让初始化的 scale 接近 0 时，变换接近恒等变换
        #    换句话说：一开始 scale≈0 时，这一层几乎等价于不动；训练过程中再学习在不同时间步上放大/压制哪些通道
        # 3. + shift[:, None]：对特征进行平移
        #    可以理解为：在当前时间步下，给每个通道设置一个“基线偏好”，某些通道整体抬高、某些通道整体压低
        # 注意：[:, None] 是为了将 [B, 1536] 广播(Broadcast)到 [B, sa_seq_len, 1536]，语法等价于 [:, None, :]，省略的维度默认就是 : 全取
        hidden_states = self.norm_out(hidden_states) * (1 + scale[:, None]) + shift[:, None]

        # proj_out_2: [B, sa_seq_len, 1536] → [B, sa_seq_len, output_dim]
        if return_all_hidden_states:
            return self.proj_out_2(hidden_states), all_hidden_states
        else:
            return self.proj_out_2(hidden_states)


class AlternateVLDiT(DiT):
    """
    Alternate Vision-Language DiT that separates image and non-image tokens
    during cross-attention processing.
    """

    def __init__(self, *args, attend_text_every_n_blocks: int = 2, **kwargs):
        super().__init__(*args, **kwargs)
        self.attend_text_every_n_blocks = attend_text_every_n_blocks

    def forward(
        self,
        hidden_states: torch.Tensor,  # Shape: (B, T, D)
        encoder_hidden_states: torch.Tensor,  # Shape: (B, S, D)
        timestep: Optional[torch.LongTensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        return_all_hidden_states: bool = False,
        image_mask: Optional[torch.Tensor] = None,
        backbone_attention_mask: Optional[torch.Tensor] = None,
    ):
        """交替关注图像 token 与非图像 token 的 DiT 变体。

        Args:
            hidden_states: 主序列特征（state+action 序列），形状 [B, T, D]。
            encoder_hidden_states: VLM 输出的序列特征，形状 [B, S, D_enc]。
            timestep: 离散时间步索引 [B]。
            encoder_attention_mask: 未直接使用（具体的图像/非图像 mask 通过 image_mask + backbone_attention_mask 组合）。
            image_mask: 标记哪些 encoder token 是图像 token，形状 [B, S]，bool 张量。
            backbone_attention_mask: 标记哪些 encoder token 有效（文本+图像），形状 [B, S]。
            return_all_hidden_states: 是否返回每层 hidden_states。
        Returns:
            torch.Tensor 或 (torch.Tensor, list[torch.Tensor]):
                - 输出形状 [B, T, output_dim]。
        """
        assert image_mask is not None, "Image mask is required"

        # 1. 时间步编码
        temb = self.timestep_encoder(timestep)

        # 2. 准备输入
        hidden_states = hidden_states.contiguous()
        encoder_hidden_states = encoder_hidden_states.contiguous()

        # 3. 构造图像/非图像的 attention mask
        # image_mask: True 表示该位置是图像 token
        # backbone_attention_mask: True 表示该位置是有效 token
        image_attention_mask = image_mask & backbone_attention_mask           # 只保留图像 token
        non_image_attention_mask = (~image_mask) & backbone_attention_mask    # 只保留非图像 token（文本等）

        all_hidden_states = [hidden_states]
        assert self.config.interleave_self_attention, "Interleave self attention must be enabled"

        # 4. 依次通过多个 Transformer Block
        for idx, block in enumerate(self.transformer_blocks):
            if idx % 2 == 1:
                # Self-attention blocks：只在 state+action 序列内部做自注意力
                hidden_states = block(
                    hidden_states,
                    attention_mask=None,
                    encoder_hidden_states=None,
                    encoder_attention_mask=None,
                    temb=temb,
                )
            else:
                # Cross-attention blocks：交替关注非图像 token 与图像 token
                if idx % (2 * self.attend_text_every_n_blocks) == 0:
                    # Attend to non-image tokens（文本等）
                    curr_encoder_attention_mask = non_image_attention_mask
                else:
                    # Attend to image tokens（视觉 patch）
                    curr_encoder_attention_mask = image_attention_mask

                hidden_states = block(
                    hidden_states,
                    attention_mask=None,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=curr_encoder_attention_mask,
                    temb=temb,
                )
            all_hidden_states.append(hidden_states)

        # 5. 输出层：与 DiT 相同，使用时间步 temb 作为条件做 AdaLN + MLP 投影
        conditioning = temb
        shift, scale = self.proj_out_1(F.silu(conditioning)).chunk(2, dim=1)
        hidden_states = self.norm_out(hidden_states) * (1 + scale[:, None]) + shift[:, None]
        if return_all_hidden_states:
            return self.proj_out_2(hidden_states), all_hidden_states
        else:
            return self.proj_out_2(hidden_states)


class SelfAttentionTransformer(ModelMixin, ConfigMixin):
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        num_attention_heads: int = 8,
        attention_head_dim: int = 64,
        output_dim: int = 26,
        num_layers: int = 12,
        dropout: float = 0.1,
        attention_bias: bool = True,
        activation_fn: str = "gelu-approximate",
        num_embeds_ada_norm: Optional[int] = 1000,
        upcast_attention: bool = False,
        max_num_positional_embeddings: int = 512,
        compute_dtype=torch.float32,
        final_dropout: bool = True,
        positional_embeddings: Optional[str] = "sinusoidal",
        interleave_self_attention=False,
    ):
        super().__init__()

        self.attention_head_dim = attention_head_dim
        self.inner_dim = self.config.num_attention_heads * self.config.attention_head_dim
        self.gradient_checkpointing = False

        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    self.inner_dim,
                    self.config.num_attention_heads,
                    self.config.attention_head_dim,
                    dropout=self.config.dropout,
                    activation_fn=self.config.activation_fn,
                    attention_bias=self.config.attention_bias,
                    upcast_attention=self.config.upcast_attention,
                    positional_embeddings=positional_embeddings,
                    num_positional_embeddings=self.config.max_num_positional_embeddings,
                    final_dropout=final_dropout,
                )
                for _ in range(self.config.num_layers)
            ]
        )
        print(
            "Total number of SelfAttentionTransformer parameters: ",
            sum(p.numel() for p in self.parameters() if p.requires_grad),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,  # Shape: (B, T, D)
        return_all_hidden_states: bool = False,
    ):
        """仅包含自注意力的 Transformer。

        Args:
            hidden_states: 输入序列特征，形状 [B, T, D]，D = inner_dim。
            return_all_hidden_states: 是否返回每一层的隐藏状态列表。
        Returns:
            torch.Tensor 或 (torch.Tensor, list[torch.Tensor]):
                - 输出 shape: [B, T, D]。
        """
        # 1. 确保张量在内存中连续，便于高效计算
        hidden_states = hidden_states.contiguous()
        all_hidden_states = [hidden_states]

        # 2. 依次通过若干个 BasicTransformerBlock（只做自注意力 + FFN）
        for idx, block in enumerate(self.transformer_blocks):
            # 这里不传 encoder_hidden_states，因此 BasicTransformerBlock 退化为纯自注意力
            hidden_states = block(hidden_states)
            all_hidden_states.append(hidden_states)

        if return_all_hidden_states:
            return hidden_states, all_hidden_states
        else:
            return hidden_states
