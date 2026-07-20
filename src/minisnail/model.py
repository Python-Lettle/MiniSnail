import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor
from einops import rearrange, einsum

from minisnail.config import SnailConfig
from minisnail.debug import console

def init_model(config: SnailConfig, device=None, dtype=None):
    '''Load a model using the options in config. \n
    It is optional to load the model from a certain weight file.
    '''
    model = SnailModel(config, device=device, dtype=dtype)
    return model.to(device)

class PWFFN(nn.Module):
    '''
        PWFFN --- Position-Wise Feed-Forward Network
        A SiLU-based SwiGLU network
    '''
    def __init__(self, d_ff: int, d_model: int, device=None, dtype=None):
        super().__init__()
        self.W1: Float[Tensor, " d_ff d_model"] = nn.Parameter(
            torch.empty(d_ff, d_model, device=device, dtype=dtype), requires_grad=True)
        self.W2: Float[Tensor, " d_model d_ff"] = nn.Parameter(
            torch.empty(d_model, d_ff, device=device, dtype=dtype), requires_grad=True) 
        self.W3: Float[Tensor, " d_ff d_model"] = nn.Parameter(
            torch.empty(d_ff, d_model, device=device, dtype=dtype), requires_grad=True)
    
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
            Build RoPE module
            theta: float,       RoPE's theta value
            d_k: int,           query and key dimension
            max_seq_len: int,   Input sequence length
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
            Initialize RoPE buffer
            max_seq_len: int,   Input sequence length
            d_k: int,           query and key dimension
            theta: float,       RoPE's theta value
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
            Apply RoPE to input tensor x
            x: Float[Tensor, " ... seq_len d_k"] Input tensor
            Returns:
                Float[Tensor, " ... seq_len d_k"] Rotated tensor
        '''
        seq_len = x.shape[-2]
        # Dynamically generate position indices
        token_positions = torch.arange(seq_len, device=x.device)

        # Slice input tensor by odd-even positions
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        # Get corresponding cos sin values according to token positions
        cos, sin = self.angle_cache[:, token_positions, :]

        # Apply rotation to each x pair
        x1_rot = cos * x1 - sin * x2
        x2_rot = sin * x1 + cos * x2
        result = torch.stack((x1_rot, x2_rot), dim=-1).flatten(-2)
        return result

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, rope_embedding=None, device=None, dtype=None):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads

        self.d_k: int = d_model // num_heads
        self.d_v: int = self.d_k

        # Construct multi-head Q K V matrices
        self.W_Q = nn.Linear(d_model, d_model, device=device, dtype=dtype)
        self.W_K = nn.Linear(d_model, d_model, device=device, dtype=dtype)
        self.W_V = nn.Linear(d_model, d_model, device=device, dtype=dtype)
        self.W_O = nn.Linear(d_model, d_model, device=device, dtype=dtype)

        self.rope_embedding = rope_embedding

    def forward(self, X: Float[Tensor, " ... sequence_length d_in"]) -> Float[Tensor, " ... sequence_length d_out"]:
        # 1. Linear projection to get Q K V (all heads together)
        Q = self.W_Q(X)
        K = self.W_K(X)
        V = self.W_V(X)

        # 2. Transform to multi-head form (batch_size, seq_len, d_model) -> (batch_size, seq_len, num_heads, d_k)
        Q = rearrange(Q, "... seq_len (num_heads d_k) -> ... num_heads seq_len d_k", num_heads=self.num_heads)
        K = rearrange(K, "... seq_len (num_heads d_k) -> ... num_heads seq_len d_k", num_heads=self.num_heads)
        V = rearrange(V, "... seq_len (num_heads d_v) -> ... num_heads seq_len d_v", num_heads=self.num_heads)
        
        # 2.5 Apply RoPE to Q K (if provided)
        if self.rope_embedding:
            Q = self.rope_embedding(Q)
            K = self.rope_embedding(K)

        # 3. Use causal encoding to calculate scaled dot-product attention
        multi_head_output: Float[Tensor, " ... queries d_v"] = F.scaled_dot_product_attention(Q, K, V, is_causal=True)
        multi_head_output = rearrange(multi_head_output, "... num_heads seq_len d_v -> ... seq_len (num_heads d_v)")

        output = self.W_O(multi_head_output)

        return output

class SnailBlock(nn.Module):
    def __init__(self, config: SnailConfig, rope_embedding=None, device=None, dtype=None) -> None:
        super().__init__()
        self.config = config
        d_model: int = config.model.d_model
        num_heads: int = config.model.num_heads
        d_ff: int = config.model.d_ff
        
        self.multihead_attention = MultiHeadSelfAttention(d_model, num_heads, rope_embedding=rope_embedding, device=device, dtype=dtype)
        self.ffn = PWFFN(d_ff, d_model, device=device, dtype=dtype)
        self.norm1 = nn.RMSNorm(d_model, eps=config.model.rms_norm_eps, device=device, dtype=dtype)
        self.norm2 = nn.RMSNorm(d_model, eps=config.model.rms_norm_eps, device=device, dtype=dtype)
        
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
    def __init__(
        self,
        config: SnailConfig,
        device: torch.device = None,
        dtype: torch.dtype = None,
    ) -> None:
        '''Constructor for SnailModel'''
        super().__init__()
        self.config = config
        self.device = device
        self.dtype = dtype

        # 1. Token Embedding 
        self.embedding = nn.Embedding(config.model.vocab_size, config.model.d_model, device=device, dtype=dtype)

        # 2. Rotary Positional Embedding Layer for Transformer Blocks
        self.d_k = config.model.d_model // config.model.num_heads
        self.rope = RotaryPositionalEmbedding(config.model.rope_theta, self.d_k, config.model.context_length, device=device)

        # 3. SnailModel Blocks
        self.blocks = nn.ModuleList([SnailBlock(config, rope_embedding=self.rope, device=device, dtype=dtype) for _ in range(config.model.num_layers)])

        # 4. Final Norm
        self.norm = nn.RMSNorm(config.model.d_model, eps=config.model.rms_norm_eps, device=device, dtype=dtype)
        
        # 5. Output Linear Layer
        self.output = nn.Linear(config.model.d_model, config.model.vocab_size, device=device, dtype=dtype)

    def forward(self, X: Float[Tensor, "... seq_len"]) -> Float[Tensor, "... seq_len vocab_size"]:
        # 1. Token Embedding
        X = self.embedding(X)
        # 2. Transformer Blocks
        for block in self.blocks:
            X = block(X)
        # 3. Final Norm
        X = self.norm(X)
        # 4. Output Embedding
        output = self.output(X)
        return output

    @torch.no_grad()
    def generate(self, X: torch.Tensor,
                max_tokens=8192,
                temperature=0.85,
                repetition_penalty=1.2,
                top_k=50,
                eos_token_id=2,
                do_sample=True,
                skip_prompt=True,        # ← 新增：默认只返回新生成的 token
                ):
        if X.dim() == 1:
            X = X.unsqueeze(0)
        X = X.long()
        original_length = X.size(-1)

        for _ in range(max_tokens):
            X = X[:, -self.config.model.context_length:]

            logits = self.forward(X)
            next_token_logits = logits[:, -1] / temperature

            if do_sample:
                # 在 temperature 缩放之后、top-k 之前加
                if repetition_penalty > 1.0:
                    # 对已生成的 token 的 logits 打折
                    for token_id in X[0].tolist():
                        next_token_logits[:, token_id] /= repetition_penalty
                if top_k:
                    topk_values, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                    threshold = topk_values[:, -1]
                    next_token_logits = next_token_logits.masked_fill(
                        next_token_logits < threshold, float("-inf")
                    )

                probs = F.softmax(next_token_logits, dim=-1)
                next_token_id = torch.multinomial(probs, 1)
            else:
                next_token_id = next_token_logits.argmax(dim=-1, keepdim=True)

            # 遇到 EOS 停止
            if eos_token_id is not None and next_token_id.item() == eos_token_id:
                break

            X = torch.cat((X, next_token_id), dim=-1)

        if skip_prompt:
            return X[:, original_length:]   # 只返回新生成的部分
        return X                            # 返回完整序列


    def chat(self, message: str, tokenizer, history=None, **kwargs):
        """
        SFT 对话生成入口。

        用法：
            response = model.chat("你好", tokenizer)
            print(response)
        """
        # 1. 构建对话历史
        messages = history or []
        messages.append({"role": "user", "content": message})

        # 2. 用 ChatML 模板 + 添加生成提示
        input_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,   # ← 自动追加 "<|im_start|>assistant\n"
            return_tensors="pt",
        )["input_ids"].to(self.config.system.device)   # 或直接 .to(next(self.parameters()).device)

        # 3. 生成
        output_ids = self.generate(
            input_ids,
            eos_token_id=tokenizer.eos_token_id,  # =2
            **kwargs
        )

        # 4. 解码为文本（跳过特殊 token）
        response = tokenizer.decode(
            output_ids[0],
            skip_special_tokens=True,
        )
        return response
