import torch
from torch import nn
import torch.nn.functional as F


def swish(x):
    """Swish activation function."""
    return x * torch.sigmoid(x)



class SinusoidalPositionalEncoding(nn.Module):
    """
    正弦位置编码模块。
    根据形状为 (B, T) 的时间步，生成形状为 (B, T, w) 的正弦编码。
    通常用于扩散模型中对时间步（diffusion step）或 Transformer 中的位置信息进行编码。
    """

    def __init__(self, embedding_dim):
        """
        初始化。
        Args:
            embedding_dim: 编码后的维度大小 (w)。
        """
        super().__init__()
        self.embedding_dim = embedding_dim


    def forward(self, timesteps):
        """
        前向传播。
        Args:
            timesteps: [B, T] 的张量，表示时间步。
        Returns:
            [B, T, embedding_dim] 的编码张量。
        """
        # 确保时间步是浮点数类型
        timesteps = timesteps.float()  # (B, T)

        B, T = timesteps.shape
        device = timesteps.device

        # 计算编码维度的一半，因为后面会将 sin 和 cos 拼接
        half_dim = self.embedding_dim // 2
        
        # 计算对数空间内的频率指数 (log space frequencies)
        # 公式参考: exp(-i * log(10000) / half_dim)
        exponent = -torch.arange(half_dim, dtype=torch.float, device=device) * (
            torch.log(torch.tensor(10000.0)) / half_dim
        )
        
        # 将 timesteps 扩展为 (B, T, 1)，并与频率指数相乘
        # (B, T, 1) * (half_dim,) -> (B, T, half_dim)
        freqs = timesteps.unsqueeze(-1) * exponent.exp() 

        # 分别计算正弦和余弦值
        sin = torch.sin(freqs)
        cos = torch.cos(freqs)
        
        # 在最后一个维度上拼接，得到最终的编码 (B, T, embedding_dim)
        enc = torch.cat([sin, cos], dim=-1)

        return enc

# 自定义了一个Linear层，为不同的embody各自准备了一个W,B矩阵
class CategorySpecificLinear(nn.Module):
    """Linear layer with category-specific weights and biases for multi-embodiment support."""

    def __init__(self, num_categories, input_dim, hidden_dim):
        super().__init__()
        self.num_categories = num_categories
        # For each category, we have separate weights and biases.
        self.W = nn.Parameter(0.02 * torch.randn(num_categories, input_dim, hidden_dim)) # 一共有num_categories个类别，每个类别都有一个W矩阵
        self.b = nn.Parameter(torch.zeros(num_categories, hidden_dim)) # 一共有num_categories个类别，每个类别都有一个b矩阵

    def forward(self, x, cat_ids):
        """
        Args:
            x: [B, T, input_dim] input tensor
            cat_ids: [B] category/embodiment IDs
        Returns:
            [B, T, hidden_dim] output tensor
        """
        # 先取embody下标得到本体对应的W,B矩阵，再做linear
        selected_W = self.W[cat_ids]
        selected_b = self.b[cat_ids]
        return torch.bmm(x, selected_W) + selected_b.unsqueeze(1)

    def expand_action_dimension(
        self, old_action_dim, new_action_dim, expand_input=False, expand_output=False
    ):
        """
        Safely expand action dimension with explicit targeting.

        Args:
            old_action_dim: Original action dimension
            new_action_dim: New (larger) action dimension
            expand_input: Whether to expand input dimension (dim=1)
            expand_output: Whether to expand output dimension (dim=2)
        """
        if new_action_dim <= old_action_dim:
            raise ValueError(
                f"New action dim {new_action_dim} must be larger than old action dim {old_action_dim}"
            )

        # Expand input dimension (dim=1) only if explicitly requested AND dimensions match
        if expand_input and self.W.shape[1] == old_action_dim:
            repeat_times = new_action_dim // old_action_dim
            remainder = new_action_dim % old_action_dim

            new_W_parts = [self.W] * repeat_times
            if remainder > 0:
                new_W_parts.append(self.W[:, :remainder, :])

            new_W = torch.cat(new_W_parts, dim=1)
            self.W = nn.Parameter(new_W)

        # Expand output dimension (dim=2) only if explicitly requested AND dimensions match
        if expand_output and self.W.shape[2] == old_action_dim:
            repeat_times = new_action_dim // old_action_dim
            remainder = new_action_dim % old_action_dim

            new_W_parts = [self.W] * repeat_times
            if remainder > 0:
                new_W_parts.append(self.W[:, :, :remainder])

            new_W = torch.cat(new_W_parts, dim=2)
            self.W = nn.Parameter(new_W)

            # Expand bias for output dimension
            if self.b.shape[1] == old_action_dim:
                new_b_parts = [self.b] * repeat_times
                if remainder > 0:
                    new_b_parts.append(self.b[:, :remainder])

                new_b = torch.cat(new_b_parts, dim=1)
                self.b = nn.Parameter(new_b)


class SmallMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        hidden = F.relu(self.layer1(x))
        return self.layer2(hidden)



class CategorySpecificMLP(nn.Module):
    """Two-layer MLP with category-specific weights for multi-embodiment support."""
    """具有类别特定权重的两层 MLP，用于支持多具身。"""

    def __init__(self, num_categories, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.num_categories = num_categories
        self.layer1 = CategorySpecificLinear(num_categories, input_dim, hidden_dim)
        self.layer2 = CategorySpecificLinear(num_categories, hidden_dim, output_dim)

    def forward(self, x, cat_ids):
        """
        Args:
            x: [B, T, input_dim] input tensor
            cat_ids: [B] category/embodiment IDs
        Returns:
            [B, T, output_dim] output tensor
        """
        """
        参数:
            x: [B, T, input_dim] 输入张量
            cat_ids: [B] 类别/具身 ID
        返回:
            [B, T, output_dim] 输出张量
        """
        hidden = F.relu(self.layer1(x, cat_ids))
        return self.layer2(hidden, cat_ids)

    def expand_action_dimension(self, old_action_dim, new_action_dim):
        """
        Expand action dimension by copying weights from existing dimensions.

        Args:
            old_action_dim: Original action dimension
            new_action_dim: New (larger) action dimension
        """
        """
        通过复制现有维度的权重来扩展动作维度。

        参数:
            old_action_dim: 原始动作维度
            new_action_dim: 新的（较大的）动作维度
        """
        # self.layer1 does not take action_dim as input, so no expansion needed
        # self.layer1 不以 action_dim 作为输入，因此不需要扩展
        self.layer2.expand_action_dimension(
            old_action_dim, new_action_dim, expand_input=False, expand_output=True
        )



class MultiEmbodimentActionEncoder(nn.Module):
    """
    Action encoder with multi-embodiment support and sinusoidal positional encoding.
    支持多具身的动作编码器，集成了正弦位置编码。
    """

    def __init__(self, action_dim, hidden_size, num_embodiments):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_embodiments = num_embodiments

        # W1: R^{w x d}, 将动作维度映射到隐藏层维度 (action_dim -> hidden_size)
        self.W1 = CategorySpecificLinear(num_embodiments, action_dim, hidden_size)  
        # W2: R^{w x 2w}, 将拼接后的动作和时间特征映射回隐藏层维度 (2 * hidden_size -> hidden_size)
        self.W2 = CategorySpecificLinear(num_embodiments, 2 * hidden_size, hidden_size)  
        # W3: R^{w x w}, 最终的线性变换层 (hidden_size -> hidden_size)
        self.W3 = CategorySpecificLinear(num_embodiments, hidden_size, hidden_size)  
        # 正弦位置编码器，用于编码扩散步数或时间步
        self.pos_encoding = SinusoidalPositionalEncoding(hidden_size)

    def forward(self, actions, timesteps, cat_ids):
        """
        前向传播
        Args:
            actions: [B, T, action_dim] 动作张量，T 为动作序列长度或 horizon
            timesteps: [B,] 时间步/扩散步张量 - 每个 batch 一个标量
            cat_ids: [B,] 类别/具身 ID 索引
        Returns:
            [B, T, hidden_size] 编码后的动作特征
        """
        B, T, _ = actions.shape

        # 1) 将每个 batch 的单一时间标量 'tau' 扩展到所有 T 个时间步，以便与动作序列对齐
        # shape (B,) => (B, T)
        if timesteps.dim() == 1 and timesteps.shape[0] == B:
            timesteps = timesteps.unsqueeze(1).expand(-1, T)
        else:
            raise ValueError(
                f"Expected `timesteps` to have shape ({B},), but got {timesteps.shape}."
            )

        # 2) 使用具身特定的线性层 W1 对动作进行初步特征提取
        # (B, T, action_dim) -> (B, T, hidden_size)
        a_emb = self.W1(actions, cat_ids)

        # 3) 获取时间步的正弦位置编码，并确保数据类型一致
        # (B, T) -> (B, T, hidden_size)
        tau_emb = self.pos_encoding(timesteps).to(dtype=a_emb.dtype)

        # 4) 在最后一个维度拼接动作特征和时间特征，通过 W2 并应用 Swish 激活函数
        # 拼接后的维度为 (B, T, 2 * hidden_size)，注入扩散步信息到每个 action horizon
        x = torch.cat([a_emb, tau_emb], dim=-1) 
        x = swish(self.W2(x, cat_ids)) # (B, T, 2 * hidden_size) -> (B, T, hidden_size)

        # 5) 通过最终的具身特定线性层 W3 得到最终编码特征
        # (B, T, hidden_size) -> (B, T, hidden_size)
        x = self.W3(x, cat_ids) 
        return x

    def expand_action_dimension(self, old_action_dim, new_action_dim):
        """
        Expand action dimension by copying weights from existing dimensions.

        Args:
            old_action_dim: Original action dimension
            new_action_dim: New (larger) action dimension
        """
        # Only W1 takes action_dim as input, so only expand its input dimension
        self.W1.expand_action_dimension(
            old_action_dim, new_action_dim, expand_input=True, expand_output=False
        )
