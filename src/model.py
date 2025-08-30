import torch
import torch.nn as nn
from torch.nn import functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # shape: (1, max_len, d_model)

    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        # 确保位置编码的维度与输入匹配
        return x + self.pe[:, :x.size(1), :]


class CausalSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.embed_dim = embed_dim

        # 使用单个线性层然后分割，而不是多个线性层
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.proj = nn.Linear(embed_dim, embed_dim)

        # 动态创建因果掩码
        self.register_buffer('tril', None)

    def forward(self, x):
        B, T, C = x.shape

        # 动态创建因果掩码
        if self.tril is None or self.tril.shape[2] < T:
            self.tril = torch.tril(torch.ones(T, T)).view(1, 1, T, T).to(x.device)

        # 计算查询、键、值
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # 形状: (B, num_heads, T, head_dim)

        # 计算注意力权重
        attn_weights = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn_weights = attn_weights.masked_fill(self.tril[:, :, :T, :T] == 0, float('-inf'))
        attn_weights = F.softmax(attn_weights, dim=-1)

        # 应用注意力
        out = attn_weights @ v  # (B, num_heads, T, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, T, C)  # 重新组合头部

        # 投影输出
        out = self.proj(out)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.attn = CausalSelfAttention(embed_dim, num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.ReLU(),
            nn.Linear(4 * embed_dim, embed_dim)
        )
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class MyLLM(nn.Module):
    def __init__(self, vocab_size, embed_dim, block_size, num_heads, num_layers):
        super().__init__()
        self.block_size = block_size
        self.embed_dim = embed_dim
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = PositionalEncoding(embed_dim, max_len=block_size)

        self.blocks = nn.Sequential(*[
            TransformerBlock(embed_dim, num_heads) for _ in range(num_layers)
        ])
        self.final_layer_norm = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)

    def forward(self, idx):
        # 确保输入序列长度不超过 block_size
        if idx.size(1) > self.block_size:
            idx = idx[:, :self.block_size]

        token_embeddings = self.token_embedding(idx)
        # 直接使用位置编码，不需要再加 token_embeddings
        x = self.pos_embedding(token_embeddings)
        x = self.blocks(x)
        x = self.final_layer_norm(x)
        logits = self.lm_head(x)
        return logits
