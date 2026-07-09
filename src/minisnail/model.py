import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor
from einops import rearrange, einsum

from minisnail.config import SnailConfig

def init_model(config: SnailConfig, from_weight: str | os.PathLike | None=None):
    '''Load a model using the options in config. \n
    It is optional to load the model from a certain weight file.
    '''
    model = SnailModel(config)
    if from_weight:
        model.load_state_dict(torch.load(from_weight))
    return model

class PWFFN(nn.Module):
    '''
        PWFFN --- Position-Wise Feed-Forward Network
        A SiLU-based SwiGLU network
    '''
    def __init__(self, d_ff: int, d_model: int, device=None, weights: list[Tensor] | None = None):
        super().__init__()
        self.W1: Float[Tensor, " d_ff d_model"] = nn.Parameter(
            torch.empty(d_ff, d_model, device=device) if weights is None else weights[0], requires_grad=True)
        self.W2: Float[Tensor, " d_model d_ff"] = nn.Parameter(
            torch.empty(d_model, d_ff, device=device) if weights is None else weights[1], requires_grad=True)
        self.W3: Float[Tensor, " d_ff d_model"] = nn.Parameter(
            torch.empty(d_ff, d_model, device=device) if weights is None else weights[2], requires_grad=True)
    
    def forward(self, x: Float[Tensor, " ... d_model"]) -> Float[Tensor, " ... d_model"]:
        '''
            FFN(x) = SwiGLU(x, w1, w2, w3) = w2( SiLU(w1 * x) ⊙ (w3 * x) )
        '''
        w1x = einsum(self.W1, x, "... d_ff d_model, ... d_model -> ... d_ff")    # Shape: [... d_ff]
        w3x = einsum(self.W3, x, "... d_ff d_model, ... d_model -> ... d_ff")    # Shape: [... d_ff]
        silu_result = F.silu(w1x)                         # SiLU(w1 * x)  Shape: [... d_ff]
        FFNx = einsum(self.W2, silu_result.mul(w3x), "... d_model d_ff, ... d_ff -> ... d_model")

        return FFNx
        
class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        '''
            构建RoPE模块, 并根据需要创建缓冲区
            theta: float,       RoPE 的 theta 值
            d_k: int,           query 向量和 key 向量的维度
            max_seq_len: int,   输入的最大序列长度
            device: torch.device | None = None Device to store the buffer on
        '''
        super().__init__()
        
        self.register_buffer(
            "angle_cache",
            RotaryPositionalEmbedding.init_cache(max_seq_len, d_k, theta), persistent=False
        )

    @staticmethod
    def init_cache(max_seq_len: int, d_k: int, theta: float) -> tuple[Float[torch.Tensor, "half_dim"], Float[torch.Tensor, "half_dim"]]:
        '''
            初始化 RoPE 缓冲区
            max_seq_len: int,   输入的最大序列长度
            d_k: int,           query 向量和 key 向量的维度
            theta: float,       RoPE 的 theta 值
            device: torch.device | None = None Device to store the buffer on
        '''
        # 计算 theta 值的幂次
        # theta_pow: (d_k,)
        theta_pow = theta ** (-torch.arange(0, d_k, 2) / d_k)

        # 生成 i_range: (max_seq_len, 1)
        i_range = torch.arange(max_seq_len).unsqueeze(-1)

        # 计算 freqs: (max_seq_len, d_k)
        freqs = torch.mul(theta_pow, i_range)       # freqs = theta^( -(2k-2) / d_k)

        cos, sin = torch.cos(freqs), torch.sin(freqs)
        return torch.stack((cos, sin))

    def forward(self, x: Float[Tensor, " ... seq_len d_k"]) -> torch.Tensor:
        '''
            处理一个形状为 (..., seq_len, d_k) 的输入张量，并返回一个相同形状的张量。
            请注意，你应该能够处理具有任意数量的批量维度的 x。
            你应该假设 token 位置是一个形状为 (..., seq_len) 的 Tensor，用于指定 x 在序列维度上的标记位置。
        '''
        seq_len = x.shape[-2]
        # 动态生成位置索引
        token_positions = torch.arange(seq_len, device=x.device)

        # 将输入按照奇偶位置切片
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        # 按照 token_positions 获取相应 cos sin
        cos, sin = self.angle_cache[:, token_positions, :]

        # 将旋转应用在 x pair 上
        x1_rot = cos * x1 - sin * x2
        x2_rot = sin * x1 + cos * x2
        result = torch.stack((x1_rot, x2_rot), dim=-1).flatten(-2)
        return result

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, rope_embedding=None, device=None):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads

        self.d_k: int = d_model // num_heads
        self.d_v: int = self.d_k

        # 构造多头 Q K V 矩阵, 这里使用一个大矩阵进行矩阵乘法, 效率优于分为多头小矩阵
        self.W_Q = nn.Linear(d_model, d_model, device=device)
        self.W_K = nn.Linear(d_model, d_model, device=device)
        self.W_V = nn.Linear(d_model, d_model, device=device)
        self.W_O = nn.Linear(d_model, d_model, device=device)

        self.rope_embedding = rope_embedding

    def forward(self, X: Float[Tensor, " ... sequence_length d_in"]) -> Float[Tensor, " ... sequence_length d_out"]:
        seq_len = X.shape[-2]

        # 1. 线性投影 得到 Q K V (所有头在一起)
        Q = self.W_Q(X)
        K = self.W_K(X)
        V = self.W_V(X)

        # 2. 变换为多头形式 (batch_size, seq_len, d_model) -> (batch_size, seq_len, num_heads, d_k)
        Q = rearrange(Q, "... seq_len (num_heads d_k) -> ... num_heads seq_len d_k", num_heads=self.num_heads)
        K = rearrange(K, "... seq_len (num_heads d_k) -> ... num_heads seq_len d_k", num_heads=self.num_heads)
        V = rearrange(V, "... seq_len (num_heads d_v) -> ... num_heads seq_len d_v", num_heads=self.num_heads)
        
        # 2.5 对 Q K 应用 RoPE (如果构建类时给出了 RotaryPositionalEmbedding)
        if self.rope_embedding:
            Q = self.rope_embedding(Q)
            K = self.rope_embedding(K)

        # 3. 使用因果编码计算缩放点积注意力
        mask = torch.tril(torch.ones(seq_len, seq_len, device=X.device))

        multi_head_output: Float[Tensor, " ... queries d_v"] = F.scaled_dot_product_attention(Q, K, V, mask)
        multi_head_output = rearrange(multi_head_output, "... num_heads seq_len d_v -> ... seq_len (num_heads d_v)")

        output = self.W_O(multi_head_output)

        return output

class SnailBlock(nn.Module):
    def __init__(self, config: SnailConfig, rope_embedding=None, device=None) -> None:
        super().__init__()
        self.config = config
        d_model: int = config.hidden_size
        num_heads: int = config.num_attention_heads
        d_ff: int = config.intermediate_size
        
        self.multihead_attention = MultiHeadSelfAttention(d_model, num_heads, rope_embedding=rope_embedding, device=device)
        self.ffn = PWFFN(d_ff, d_model, device=device)
        self.norm1 = nn.RMSNorm(d_model, eps=config.rms_norm_eps, device=device)
        self.norm2 = nn.RMSNorm(d_model, eps=config.rms_norm_eps, device=device)
        
    def forward(self, X: Float[Tensor, "... seq_len d_model"]) -> Float[Tensor, "... seq_len d_model"]:
        # 1. Pre-norm
        _X = self.norm1(X)
        # 2. Causal Multi-Head Self-Attention
        _X = self.multihead_attention(_X)
        # 3. X1 = X + multi_head_output
        X1 = X + _X
        # 4. Pre-norm
        __X = self.norm2(X1)
        # 5. Position-Wise Feed-Forward
        __X = self.ffn(__X)
        # 6. Output = X1 + PWFFN(X1)
        output = X1 + __X
        return output

class SnailModel(nn.Module):
    def __init__(self, config: SnailConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.num_hidden_layers = config.num_hidden_layers

        # Components
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.rope = RotaryPositionalEmbedding(
            config.rope_theta,
            config.hidden_size // config.num_attention_heads,
            config.max_seq_len,
        )
        self.blocks = nn.ModuleList([SnailBlock(config, self.rope)])
        self.final_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.output = nn.Linear(config.hidden_size, config.vocab_size)

    def forward(self, X: Float[Tensor, "... seq_len"]) -> Float[Tensor, "... seq_len vocab_size"]:
        # 1. Token Embedding
        X = self.embedding(X)
        # 2. Transformer Blocks
        for block in self.blocks:
            X = block(X)
        # 3. Final Norm
        X = self.final_norm(X)
        # 4. Output Embedding
        output = self.output(X)
        return output
    
    def generate(self, X: torch.Tensor,
        attention_mask=None,
        max_new_tokens=8192,
        temperature=0.85,
        top_p=0.85,
        top_k=50,
        eos_token_id=2,
        streamer=None,
        use_cache=True,
        num_return_sequences=1,
        do_sample=True,
        repetition_penalty=1.0,
        **kwargs
    ):
        if X.dim() == 1:
            X = X.unsqueeze(0)
        
        original_sequence_length = X.size(-1)

        for _ in range(max_new_tokens):
            # If the prompt exceeds the context_length, truncate the prompt.
            X = X[:, -self.config.max_generate_token:] if X.size(1) > self.config.max_generate_token else X
            # Get logits
            logits = self.forward(X)
            # Get the logits for the next token
            next_token_logits = logits[:, -1]
            # Temperature scale
            temperature_scaled_next_token_logits = next_token_logits / temperature
            # If top-k is provided, only the top-k tokens will be considered
            if top_k:
                topk_values, _ = torch.topk(
                    temperature_scaled_next_token_logits,
                    min(top_k, temperature_scaled_next_token_logits.size(-1)),
                )
                # 获取 top-k 个 token 中分数最高的 token 的分数
                threshold = topk_values[:, -1]
                top_k_mask = temperature_scaled_next_token_logits < threshold
                temperature_scaled_next_token_logits.masked_fill(top_k_mask, float("-inf"))
            # Top-p sampling
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(temperature_scaled_next_token_logits, descending=True)
                mask = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1) > top_p
                mask[..., 1:], mask[..., 0] = mask[..., :-1].clone(), 0
                temperature_scaled_next_token_logits[mask.scatter(1, sorted_indices, mask)] = -float('inf')
            
            next_token_probabilities = F.softmax(temperature_scaled_next_token_logits, dim=-1)
            next_token_id = torch.multinomial(next_token_probabilities, 1)

            # Upon encountering an EOS token, stop generating
            if eos_token_id is not None and next_token_id.item() == eos_token_id:
                break
            X = torch.cat((X, next_token_id), dim=-1)
        new_token_ids = X[:, original_sequence_length:]
        return new_token_ids
