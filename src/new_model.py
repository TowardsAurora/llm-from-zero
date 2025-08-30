# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import math
#
#
# class RotaryPositionEmbedding(nn.Module):
#     """旋转位置编码 (RoPE)"""
#
#     def __init__(self, dim, max_seq_len=512):
#         super().__init__()
#         self.dim = dim
#         self.max_seq_len = max_seq_len
#
#         # 预计算正弦和余弦值
#         inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
#         t = torch.arange(max_seq_len).type_as(inv_freq)
#         freqs = torch.einsum("i,j->ij", t, inv_freq)
#         emb = torch.cat((freqs, freqs), dim=-1)
#         self.register_buffer("cos_cached", emb.cos()[None, None, :, :])
#         self.register_buffer("sin_cached", emb.sin()[None, None, :, :])
#
#     def forward(self, x, seq_len=None):
#         if seq_len > self.max_seq_len:
#             # 动态扩展位置编码
#             self._extend_rotary_emb(seq_len)
#
#         return (
#             self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
#             self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype)
#         )
#
#     def _extend_rotary_emb(self, seq_len):
#         inv_freq = 1.0 / (10000 ** (torch.arange(0, self.dim, 2).float() / self.dim))
#         t = torch.arange(seq_len).type_as(inv_freq)
#         freqs = torch.einsum("i,j->ij", t, inv_freq)
#         emb = torch.cat((freqs, freqs), dim=-1)
#
#         self.cos_cached = emb.cos()[None, None, :, :]
#         self.sin_cached = emb.sin()[None, None, :, :]
#         self.max_seq_len = seq_len
#
#
# def rotate_half(x):
#     """将张量的后一半旋转"""
#     x1, x2 = x.chunk(2, dim=-1)
#     return torch.cat((-x2, x1), dim=-1)
#
#
# def apply_rotary_pos_emb(q, k, cos, sin):
#     """应用旋转位置编码到查询和键"""
#     cos = cos[:, :, :q.shape[2], :]
#     sin = sin[:, :, :q.shape[2], :]
#
#     q_embed = (q * cos) + (rotate_half(q) * sin)
#     k_embed = (k * cos) + (rotate_half(k) * sin)
#     return q_embed, k_embed
#
#
# class RMSNorm(nn.Module):
#     """RMS归一化 (LLaMA使用)"""
#
#     def __init__(self, dim, eps=1e-8):
#         super().__init__()
#         self.scale = dim ** -0.5
#         self.eps = eps
#         self.weight = nn.Parameter(torch.ones(dim))
#
#     def forward(self, x):
#         norm = torch.norm(x, dim=-1, keepdim=True) * self.scale
#         return x / norm.clamp(min=self.eps) * self.weight
#
#
# class SwiGLU(nn.Module):
#     """SwiGLU激活函数 (LLaMA使用)"""
#
#     def forward(self, x):
#         x, gate = x.chunk(2, dim=-1)
#         return F.silu(gate) * x
#
#
# class CausalSelfAttention(nn.Module):
#     """带RoPE的因果自注意力机制"""
#
#     def __init__(self, embed_dim, num_heads):
#         super().__init__()
#         assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
#
#         self.num_heads = num_heads
#         self.head_dim = embed_dim // num_heads
#         self.embed_dim = embed_dim
#
#         # 查询、键、值投影
#         self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
#         self.output_proj = nn.Linear(embed_dim, embed_dim, bias=False)
#
#         # 旋转位置编码
#         self.rotary_emb = RotaryPositionEmbedding(self.head_dim)
#
#         # 注册因果掩码
#         self.register_buffer("causal_mask", None)
#
#     def forward(self, x):
#         batch_size, seq_len, _ = x.shape
#
#         # 生成查询、键、值
#         qkv = self.qkv_proj(x)
#         q, k, v = qkv.chunk(3, dim=-1)
#
#         # 重塑为多头格式
#         q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
#         k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
#         v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
#
#         # 应用旋转位置编码
#         cos, sin = self.rotary_emb(q, seq_len=seq_len)
#         q, k = apply_rotary_pos_emb(q, k, cos, sin)
#
#         # 计算注意力分数
#         attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
#
#         # 应用因果掩码
#         if self.causal_mask is None or self.causal_mask.shape[2] < seq_len:
#             self.causal_mask = torch.triu(
#                 torch.full((seq_len, seq_len), float('-inf')), diagonal=1
#             ).to(x.device)
#
#         attn_scores = attn_scores + self.causal_mask[None, None, :seq_len, :seq_len]
#
#         # 计算注意力权重
#         attn_weights = F.softmax(attn_scores, dim=-1)
#
#         # 应用注意力到值
#         attn_output = torch.matmul(attn_weights, v)
#
#         # 重塑并投影输出
#         attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
#         attn_output = self.output_proj(attn_output)
#
#         return attn_output
#
#
# class FeedForward(nn.Module):
#     """前馈网络 (使用SwiGLU)"""
#
#     def __init__(self, embed_dim, hidden_dim_multiplier=4):
#         super().__init__()
#         hidden_dim = int(embed_dim * hidden_dim_multiplier * 2 / 3)  # LLaMA的比例
#
#         self.w1 = nn.Linear(embed_dim, hidden_dim, bias=False)
#         self.w2 = nn.Linear(hidden_dim, embed_dim, bias=False)
#         self.w3 = nn.Linear(embed_dim, hidden_dim, bias=False)
#         self.activation = SwiGLU()
#
#     def forward(self, x):
#         return self.w2(self.activation(torch.cat([self.w1(x), self.w3(x)], dim=-1)))
#
#
# class TransformerBlock(nn.Module):
#     """Transformer块 (使用RMSNorm)"""
#
#     def __init__(self, embed_dim, num_heads):
#         super().__init__()
#         self.attn_norm = RMSNorm(embed_dim)
#         self.attn = CausalSelfAttention(embed_dim, num_heads)
#
#         self.ffn_norm = RMSNorm(embed_dim)
#         self.ffn = FeedForward(embed_dim)
#
#     def forward(self, x):
#         # 注意力部分
#         attn_output = self.attn(self.attn_norm(x))
#         x = x + attn_output
#
#         # 前馈部分
#         ffn_output = self.ffn(self.ffn_norm(x))
#         x = x + ffn_output
#
#         return x
#
#
# class MyLLM(nn.Module):
#     """改进的大语言模型 (基于LLaMA架构)"""
#
#     def __init__(self, vocab_size, embed_dim, block_size, num_heads, num_layers):
#         super().__init__()
#         self.block_size = block_size
#         self.embed_dim = embed_dim
#
#         # 词嵌入
#         self.token_embedding = nn.Embedding(vocab_size, embed_dim)
#
#         # Transformer块
#         self.blocks = nn.Sequential(*[
#             TransformerBlock(embed_dim, num_heads) for _ in range(num_layers)
#         ])
#
#         # 输出层
#         self.output_norm = RMSNorm(embed_dim)
#         self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)
#
#         # 权重绑定
#         self.lm_head.weight = self.token_embedding.weight
#
#         # 初始化权重
#         self.apply(self._init_weights)
#
#     def _init_weights(self, module):
#         if isinstance(module, nn.Linear):
#             torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
#             if module.bias is not None:
#                 torch.nn.init.zeros_(module.bias)
#         elif isinstance(module, nn.Embedding):
#             torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
#
#     def forward(self, idx):
#         batch_size, seq_len = idx.shape
#
#         # 确保输入序列长度不超过block_size
#         if seq_len > self.block_size:
#             idx = idx[:, -self.block_size:]
#             seq_len = self.block_size
#
#         # 词嵌入
#         token_embeddings = self.token_embedding(idx)
#
#         # 通过Transformer块
#         x = self.blocks(token_embeddings)
#
#         # 输出层
#         x = self.output_norm(x)
#         logits = self.lm_head(x)
#
#         return logits
#
#     # def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
#     #     """生成文本"""
#     #     for _ in range(max_new_tokens):
#     #         # 如果序列太长，截断到block_size
#     #         idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]
#     #
#     #         # 获取预测
#     #         with torch.no_grad():
#     #             logits = self(idx_cond)
#     #             logits = logits[:, -1, :] / temperature
#     #
#     #             # 可选: 应用top-k过滤
#     #             if top_k is not None:
#     #                 v, _ = torch.topk(logits, top_k)
#     #                 logits[logits < v[:, [-1]]] = -float('Inf')
#     #
#     #             # 采样下一个token
#     #             probs = F.softmax(logits, dim=-1)
#     #             idx_next = torch.multinomial(probs, num_samples=1)
#     #
#     #             # 添加到序列
#     #             idx = torch.cat((idx, idx_next), dim=1)
#     #
#     #     return idx


import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class RotaryPositionEmbedding(nn.Module):
    """旋转位置编码 (RoPE)"""

    def __init__(self, dim, max_seq_len=512):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len

        # 预计算正弦和余弦值
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_seq_len).type_as(inv_freq)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :])
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :])

    def forward(self, x, seq_len=None):
        if seq_len > self.max_seq_len:
            # 动态扩展位置编码
            self._extend_rotary_emb(seq_len)

        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype)
        )

    def _extend_rotary_emb(self, seq_len):
        inv_freq = 1.0 / (10000 ** (torch.arange(0, self.dim, 2).float() / self.dim))
        t = torch.arange(seq_len).type_as(inv_freq)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)

        self.cos_cached = emb.cos()[None, None, :, :]
        self.sin_cached = emb.sin()[None, None, :, :]
        self.max_seq_len = seq_len


def rotate_half(x):
    """将张量的后一半旋转"""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    """应用旋转位置编码到查询和键"""
    cos = cos[:, :, :q.shape[2], :]
    sin = sin[:, :, :q.shape[2], :]

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class RMSNorm(nn.Module):
    """RMS归一化 (LLaMA使用)"""

    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.scale = dim ** -0.5
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = torch.norm(x, dim=-1, keepdim=True) * self.scale
        return x / norm.clamp(min=self.eps) * self.weight


class SwiGLU(nn.Module):
    """SwiGLU激活函数 (LLaMA使用)"""

    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x


class CausalSelfAttention(nn.Module):
    """带RoPE的因果自注意力机制"""

    def __init__(self, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.embed_dim = embed_dim

        # 查询、键、值投影
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.output_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        # 旋转位置编码
        self.rotary_emb = RotaryPositionEmbedding(self.head_dim)

        # 不注册因果掩码为缓冲区，而是在需要时动态创建
        self.causal_mask = None
        self.current_seq_len = 0

    def forward(self, x):
        batch_size, seq_len, _ = x.shape

        # 生成查询、键、值
        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)

        # 重塑为多头格式
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # 应用旋转位置编码
        cos, sin = self.rotary_emb(q, seq_len=seq_len)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # 计算注意力分数
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # 动态创建因果掩码（如果需要）
        if self.causal_mask is None or self.current_seq_len < seq_len:
            self.causal_mask = torch.triu(
                torch.full((seq_len, seq_len), float('-inf')), diagonal=1
            ).to(x.device)
            self.current_seq_len = seq_len

        # 应用因果掩码
        attn_scores = attn_scores + self.causal_mask[:seq_len, :seq_len][None, None, :, :]

        # 计算注意力权重
        attn_weights = F.softmax(attn_scores, dim=-1)

        # 应用注意力到值
        attn_output = torch.matmul(attn_weights, v)

        # 重塑并投影输出
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        attn_output = self.output_proj(attn_output)

        return attn_output


class FeedForward(nn.Module):
    """前馈网络 (使用SwiGLU)"""

    def __init__(self, embed_dim, hidden_dim_multiplier=4):
        super().__init__()
        hidden_dim = int(embed_dim * hidden_dim_multiplier * 2 / 3)  # LLaMA的比例

        self.w1 = nn.Linear(embed_dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, embed_dim, bias=False)
        self.w3 = nn.Linear(embed_dim, hidden_dim, bias=False)
        self.activation = SwiGLU()

    def forward(self, x):
        return self.w2(self.activation(torch.cat([self.w1(x), self.w3(x)], dim=-1)))


class TransformerBlock(nn.Module):
    """Transformer块 (使用RMSNorm)"""

    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.attn_norm = RMSNorm(embed_dim)
        self.attn = CausalSelfAttention(embed_dim, num_heads)

        self.ffn_norm = RMSNorm(embed_dim)
        self.ffn = FeedForward(embed_dim)

    def forward(self, x):
        # 注意力部分
        attn_output = self.attn(self.attn_norm(x))
        x = x + attn_output

        # 前馈部分
        ffn_output = self.ffn(self.ffn_norm(x))
        x = x + ffn_output

        return x


class MyLLM(nn.Module):
    """改进的大语言模型 (基于LLaMA架构)"""

    def __init__(self, vocab_size, embed_dim, block_size, num_heads, num_layers):
        super().__init__()
        self.block_size = block_size
        self.embed_dim = embed_dim

        # 词嵌入
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)

        # Transformer块
        self.blocks = nn.Sequential(*[
            TransformerBlock(embed_dim, num_heads) for _ in range(num_layers)
        ])

        # 输出层
        self.output_norm = RMSNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)

        # 权重绑定
        self.lm_head.weight = self.token_embedding.weight

        # 初始化权重
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx):
        batch_size, seq_len = idx.shape

        # 确保输入序列长度不超过block_size
        if seq_len > self.block_size:
            idx = idx[:, -self.block_size:]
            seq_len = self.block_size

        # 词嵌入
        token_embeddings = self.token_embedding(idx)

        # 通过Transformer块
        x = self.blocks(token_embeddings)

        # 输出层
        x = self.output_norm(x)
        logits = self.lm_head(x)

        return logits

    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """生成文本"""
        for _ in range(max_new_tokens):
            # 如果序列太长，截断到block_size
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]

            # 获取预测
            with torch.no_grad():
                logits = self(idx_cond)
                logits = logits[:, -1, :] / temperature

                # 可选: 应用top-k过滤
                if top_k is not None:
                    v, _ = torch.topk(logits, top_k)
                    logits[logits < v[:, [-1]]] = -float('Inf')

                # 采样下一个token
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)

                # 添加到序列
                idx = torch.cat((idx, idx_next), dim=1)

        return idx