import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor
from einops import rearrange, einsum

from minisnail.config import SnailConfig
from minisnail.debug import console

def init_model(config: SnailConfig, model_path: str = None, device=None, dtype=None):
    '''使用 config 初始化模型, 可以从 model_path 加载模型参数'''
    model = SnailModel(config, device=device, dtype=dtype)
    if model_path is not None:
        model.load_state_dict(
            torch.load(model_path, map_location=model.device, weights_only=True)
        )
    return model

def top_p_filtering(logits, top_p):
    """
    Nucleus sampling:
    保留累计概率达到 top_p 的 token
    """

    if top_p >= 1.0:
        return logits

    # 按概率从大到小排序
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)

    # 转换成概率
    sorted_probs = torch.softmax(sorted_logits, dim=-1)

    # 累计概率
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # 删除累计概率超过 top_p 之后的 token
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = (sorted_indices_to_remove[..., :-1].clone())
    # 保留至少一个 token
    sorted_indices_to_remove[..., 0] = False

    # 映射回原始token位置
    indices_to_remove = torch.zeros_like(sorted_indices_to_remove)

    indices_to_remove.scatter_(
        dim=-1,
        index=sorted_indices,
        src=sorted_indices_to_remove
    )

    logits = logits.masked_fill(
        indices_to_remove,
        float("-inf")
    )

    return logits

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
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """初始化权重 (Kaiming uniform, 与 nn.Linear 默认一致)。
        torch.empty 不会初始化内存, 不调用 init 会得到垃圾值 (初始 loss 异常大甚至 NaN)
        """
        nn.init.kaiming_uniform_(self.W1, a=5 ** 0.5)
        nn.init.kaiming_uniform_(self.W2, a=5 ** 0.5)
        nn.init.kaiming_uniform_(self.W3, a=5 ** 0.5)
    
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
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None, dtype=None):
        '''
            Build RoPE module
            theta: float,       RoPE's theta value
            d_k: int,           query and key dimension
            max_seq_len: int,   Input sequence length
            device: torch.device | None = None Device to store the buffer on
            dtype: torch.dtype | None = None Data type for the angle cache
        '''
        super().__init__()

        angle_cache = RotaryPositionalEmbedding.init_cache(max_seq_len, d_k, theta)
        if device is not None or dtype is not None:
            angle_cache = angle_cache.to(device=device, dtype=dtype)
        self.register_buffer(
            "angle_cache",
            angle_cache, persistent=False
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

    def forward(self, x: Float[Tensor, " ... seq_len d_k"], start_pos: int = 0) -> torch.Tensor:
        '''
            Apply RoPE to input tensor x
            x: Float[Tensor, " ... seq_len d_k"] Input tensor
            Returns:
                Float[Tensor, " ... seq_len d_k"] Rotated tensor
        '''
        seq_len = x.shape[-2]
        # Dynamically generate position indices
        token_positions = torch.arange(start_pos, start_pos + seq_len, device=x.device)

        # Slice input tensor by odd-even positions
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        # Get corresponding cos sin values according to token positions
        # buffer 保持 fp32 保证角度精度, 计算时对齐输入 dtype (AMP 下 Q/K 为低精度)
        cos, sin = self.angle_cache[:, token_positions, :].to(dtype=x.dtype)

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
        self.W_Q = nn.Linear(d_model, d_model, device=device, dtype=dtype, bias=False)
        self.W_K = nn.Linear(d_model, d_model, device=device, dtype=dtype, bias=False)
        self.W_V = nn.Linear(d_model, d_model, device=device, dtype=dtype, bias=False)
        self.W_O = nn.Linear(d_model, d_model, device=device, dtype=dtype, bias=False)

        self.rope_embedding = rope_embedding

    def forward(self, X: Float[Tensor, " ... sequence_length d_in"], past_kv: tuple[torch.Tensor, torch.Tensor] | None = None, use_cache: bool = False, start_pos: int = 0) -> tuple[Float[Tensor, " ... sequence_length d_out"], tuple[torch.Tensor, torch.Tensor]]:
        # 1. Linear projection to get Q K V (all heads together)
        Q = self.W_Q(X)
        K = self.W_K(X)
        V = self.W_V(X)

        # 2.1 Transform to multi-head form (batch_size, seq_len, d_model) -> (batch_size, seq_len, num_heads, d_k)
        Q = rearrange(Q, "... seq_len (num_heads d_k) -> ... num_heads seq_len d_k", num_heads=self.num_heads)
        K = rearrange(K, "... seq_len (num_heads d_k) -> ... num_heads seq_len d_k", num_heads=self.num_heads)
        V = rearrange(V, "... seq_len (num_heads d_v) -> ... num_heads seq_len d_v", num_heads=self.num_heads)
        
        # 2.2 Apply RoPE to Q K (if provided)
        if self.rope_embedding:
            Q = self.rope_embedding(Q, start_pos=start_pos)
            K = self.rope_embedding(K, start_pos=start_pos)

        # 2.3 Cache KV if required
        past_len = 0
        if past_kv is not None:
            past_k, past_v = past_kv
            past_len = past_k.shape[-2]
            K = torch.cat((past_k, K), dim=-2)
            V = torch.cat((past_v, V), dim=-2)

        if use_cache:
            present_kv=(K,V)
        else:
            present_kv=None

        # 3. Attention
        if past_len > 0:
            # 当前 Q 的长度
            query_len = Q.shape[-2]
            # 当前 K/V 的总长度
            key_len = K.shape[-2]

            # Query 的绝对位置
            query_positions = torch.arange(past_len, past_len + query_len, device=X.device).unsqueeze(-1)

            # Key 的位置
            key_positions = torch.arange(key_len,device=X.device).unsqueeze(0)

            # causal mask
            causal_mask = (key_positions <= query_positions)

            multi_head_output: Float[Tensor, " ... queries d_v"] = F.scaled_dot_product_attention(Q, K, V, attn_mask=causal_mask, is_causal=False)
        else:
            multi_head_output: Float[Tensor, " ... queries d_v"] = F.scaled_dot_product_attention(Q, K, V, is_causal=True)

        multi_head_output = rearrange(multi_head_output, "... num_heads seq_len d_v -> ... seq_len (num_heads d_v)")

        output = self.W_O(multi_head_output)
        
        if use_cache:
            return output, present_kv
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
        
    def forward(self,
            X: Float[Tensor, "... seq_len d_model"],
            past_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
            use_cache: bool = False,
            start_pos: int = 0
        ) -> tuple[Float[Tensor, "... seq_len d_model"], tuple[torch.Tensor, torch.Tensor]]:
        # 1. Pre-norm
        _X = self.norm1(X)
        # 2. Causal Multi-Head Self-Attention
        if use_cache:
            _X, layer_present_kv = self.multihead_attention(_X, past_kv=past_kv, use_cache=use_cache, start_pos=start_pos)
        else:
            _X = self.multihead_attention(_X, start_pos=start_pos)
            layer_present_kv = None

        # 3. X1 = X + multi_head_output
        X1 = X + _X
        # 4. Pre-norm
        __X = self.norm2(X1)
        # 5. Position-Wise Feed-Forward
        __X = self.ffn(__X)
        # 6. Output = X1 + PWFFN(X1)
        output = X1 + __X

        if use_cache:
            return output, layer_present_kv
        return output


def _no_repeat_ngram_mask(next_token_logits, seq, no_repeat_ngram_size):
    """把会复现历史 n-gram 的候选 token 置为 -inf, 抑制"答完停不下来/循环"的退化。

    args:
        next_token_logits: shape (1, vocab_size)
        seq: list[int] 当前完整 token 序列 (prompt + 已生成)
        no_repeat_ngram_size: n-gram 长度, <=1 表示关闭
    """
    n = no_repeat_ngram_size
    if not n or n < 2 or len(seq) < n:
        return next_token_logits
    prefix = tuple(seq[-(n - 1):])
    banned = {
        seq[i + n - 1]
        for i in range(len(seq) - n + 1)
        if tuple(seq[i:i + n - 1]) == prefix
    }
    for t in banned:
        next_token_logits[:, t] = float("-inf")
    return next_token_logits


def _apply_repetition_penalty(next_token_logits, seen_token_ids, repetition_penalty):
    """对已出现过的 token 各应用一次 repetition penalty。

    正 logit 除以 penalty，负 logit 乘以 penalty，保证两种情况都会降低
    token 的选中概率。seen_token_ids 使用集合去重，避免按出现次数重复惩罚。
    """
    if repetition_penalty <= 1.0 or not seen_token_ids:
        return next_token_logits

    token_ids = list(seen_token_ids)
    token_logits = next_token_logits[:, token_ids]
    next_token_logits[:, token_ids] = torch.where(
        token_logits < 0,
        token_logits * repetition_penalty,
        token_logits / repetition_penalty,
    )
    return next_token_logits


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
        self.device = device if device else torch.device(config.system.device)
        self.dtype = dtype if dtype else None
        if self.dtype is None:
            # get_torch_dtype 返回 (model_dtype, amp_dtype), 这里只取 model_dtype
            model_dtype, _ = config.get_torch_dtype()
            self.dtype = model_dtype

        # 1. Token Embedding
        self.embedding = nn.Embedding(config.model.vocab_size, config.model.d_model, device=self.device, dtype=self.dtype)
        # nn.Embedding 默认 N(0,1) 方差偏大; LLM 标准做法 std=0.02 (与 MiniMind 一致)
        nn.init.normal_(self.embedding.weight, std=0.02)

        # 2. Rotary Positional Embedding Layer for Transformer Blocks
        self.d_k = config.model.d_model // config.model.num_heads
        self.rope = RotaryPositionalEmbedding(config.model.rope_theta, self.d_k, config.model.context_length, device=self.device, dtype=self.dtype)

        # 3. SnailModel Blocks
        self.blocks = nn.ModuleList([SnailBlock(config, rope_embedding=self.rope, device=self.device, dtype=self.dtype) for _ in range(config.model.num_layers)])

        # 4. Final Norm
        self.norm = nn.RMSNorm(config.model.d_model, eps=config.model.rms_norm_eps, device=self.device, dtype=self.dtype)
        
        # 5. Output Linear Layer
        self.output = nn.Linear(config.model.d_model, config.model.vocab_size, device=self.device, dtype=self.dtype, bias=False)

        # weight tying
        self.output.weight = self.embedding.weight

    def forward(self,
            X: Float[Tensor, "... seq_len"],
            past_kv: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
            use_cache: bool = False,
            start_pos: int = 0
        ) -> tuple[Float[Tensor, "... seq_len vocab_size"], list[tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass.

        use_cache=False:
            普通训练模式。

        use_cache=True:
            generation KV Cache 模式。

        past_kv:
            [
                (K_layer_0, V_layer_0),
                (K_layer_1, V_layer_1),
                ...
            ]
        """
        # 1. Token Embedding
        X = self.embedding(X)
        
        # 2. Transformer Blocks
        # Blocks KV Cache
        present_kv: list[tuple[torch.Tensor, torch.Tensor]] = [] if use_cache else None

        for layer_idx, block in enumerate(self.blocks):
            if use_cache:
                layer_past_kv = None
                if past_kv is not None:
                    layer_past_kv = past_kv[layer_idx]
                X, present_key_value = block(X, past_kv=layer_past_kv, use_cache=True, start_pos=start_pos)
                present_kv.append(present_key_value)
            else: 
                X = block(X, start_pos=start_pos)

        # 3. Final Norm
        X = self.norm(X)
        # 4. Output Embedding
        output = self.output(X)

        if use_cache:
            return output, present_kv
        return output

    @torch.no_grad()
    def generate(self, 
            X: torch.Tensor,
            max_tokens=512,
            temperature=0.85,
            repetition_penalty=1.2,
            top_k=50,
            top_p=0.9,
            eos_token_id=2,
            do_sample=True,
            skip_prompt=True,
            no_repeat_ngram_size=0,
        ):
        if X.dim() == 1:
            X = X.unsqueeze(0)
        X = X.long()
        original_length = X.size(-1)

        # 1. Prompt 长度超过 context length
        if X.size(-1) > self.config.model.context_length:
            X = X[:, -self.config.model.context_length:]
        
        # 2. Prefill
        # 第一次把整个 prompt 输入模型
        # 得到：logits + 每一层的 KV Cache
        logits, past_kv = self.forward(X, use_cache=True, start_pos=0)
        
        generated_ids = []
        seen_token_ids = set(X[0].tolist())

        # 当前 cache 中有多少 token
        cache_len = X.size(-1)

        # 3. Autoregressive Generation
        for _ in range(max_tokens):
            # Sampling
            if do_sample:
                next_token_logits = logits[:, -1] / temperature
                # Repetition Penalty
                next_token_logits = _apply_repetition_penalty(
                    next_token_logits, seen_token_ids, repetition_penalty
                )
                # n-gram 抑制放在 top_k/top_p 之前, 保证 top_k 始终保留候选, 避免 logits 全 -inf
                next_token_logits = _no_repeat_ngram_mask(next_token_logits, X[0].tolist(), no_repeat_ngram_size)
                # Top-K
                if top_k:
                    topk_values, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                    threshold = topk_values[:, -1].unsqueeze(-1)
                    next_token_logits = next_token_logits.masked_fill(
                        next_token_logits < threshold, float("-inf")
                    )
                # Top-P
                if top_p < 1.0:
                    next_token_logits = top_p_filtering(next_token_logits, top_p)
                # Sample
                probs = F.softmax(next_token_logits, dim=-1)
                next_token_id = torch.multinomial(probs, 1)
            else:
                # Greedy Decoding
                next_token_logits = logits[:, -1]
                next_token_logits = _apply_repetition_penalty(
                    next_token_logits, seen_token_ids, repetition_penalty
                )
                next_token_logits = _no_repeat_ngram_mask(next_token_logits, X[0].tolist(), no_repeat_ngram_size)
                next_token_id = next_token_logits.argmax(dim=-1, keepdim=True)

            # 遇到 EOS 停止
            if eos_token_id is not None and next_token_id.item() == eos_token_id:
                break

            generated_ids.append(next_token_id.item())
            seen_token_ids.add(next_token_id.item())

            # 4. 将新 token 添加到序列
            X = torch.cat((X, next_token_id), dim=-1)

            # 5. Context Window 已经满了
            # KV Cache 无法无限增长
            # 当达到 context_length 后：
            # 重新截取最近 context_length tokens 并重新建立 cache
            if X.size(-1) > self.config.model.context_length:
                X = X[:, -self.config.model.context_length:]
                logits, past_kv = self.forward(X, use_cache=True, start_pos=0)
                cache_len = self.config.model.context_length
                continue
            
            # 6. KV Cache Decode
            logits, past_kv = (
                self.forward(
                    next_token_id,
                    past_kv=past_kv,
                    use_cache=True,
                    start_pos=cache_len
                )
            )
            cache_len += 1
        # 7. Construct Output
        output_ids = torch.tensor([generated_ids], dtype=X.dtype, device=X.device)

        if skip_prompt:
            return output_ids

        return torch.cat((X[:, :original_length], output_ids), dim=-1)

    @torch.no_grad()
    def streaming_generate(self, 
            X: torch.Tensor,
            max_tokens=512,
            temperature=0.85,
            repetition_penalty=1.2,
            top_k=50,
            top_p=0.9,
            eos_token_id=2,
            do_sample=True,
            no_repeat_ngram_size=0,
        ):
        if X.dim() == 1:
            X = X.unsqueeze(0)
        X = X.long()

        # 1. Prompt 长度超过 context length
        if X.size(-1) > self.config.model.context_length:
            X = X[:, -self.config.model.context_length:]
        
        # 2. Prefill
        # 第一次把整个 prompt 输入模型
        # 得到：logits + 每一层的 KV Cache
        logits, past_kv = self.forward(X, use_cache=True, start_pos=0)
        
        seen_token_ids = set(X[0].tolist())

        # 当前 cache 中有多少 token
        cache_len = X.size(-1)

        # 3. Autoregressive Generation
        for _ in range(max_tokens):
            # Sampling
            if do_sample:
                next_token_logits = logits[:, -1] / temperature
                # Repetition Penalty
                next_token_logits = _apply_repetition_penalty(
                    next_token_logits, seen_token_ids, repetition_penalty
                )
                # n-gram 抑制放在 top_k/top_p 之前, 保证 top_k 始终保留候选, 避免 logits 全 -inf
                next_token_logits = _no_repeat_ngram_mask(next_token_logits, X[0].tolist(), no_repeat_ngram_size)
                # Top-K
                if top_k:
                    topk_values, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                    threshold = topk_values[:, -1].unsqueeze(-1)
                    next_token_logits = next_token_logits.masked_fill(
                        next_token_logits < threshold, float("-inf")
                    )
                # Top-P
                if top_p < 1.0:
                    next_token_logits = top_p_filtering(next_token_logits, top_p)
                # Sample
                probs = F.softmax(next_token_logits, dim=-1)
                next_token_id = torch.multinomial(probs, 1)
            else:
                # Greedy Decoding
                next_token_logits = logits[:, -1]
                next_token_logits = _apply_repetition_penalty(
                    next_token_logits, seen_token_ids, repetition_penalty
                )
                next_token_logits = _no_repeat_ngram_mask(next_token_logits, X[0].tolist(), no_repeat_ngram_size)
                next_token_id = next_token_logits.argmax(dim=-1, keepdim=True)

            # 遇到 EOS 停止
            if eos_token_id is not None and next_token_id.item() == eos_token_id:
                break

            seen_token_ids.add(next_token_id.item())

            # 4. 将新 token 添加到序列
            X = torch.cat((X, next_token_id), dim=-1)

            # 生成后立即输出 token。缓存重建属于下一轮生成的准备工作，
            # 不应该跳过当前 token 的 yield。
            yield next_token_id.item()

            # 5. Context Window 已经满了
            # KV Cache 无法无限增长
            # 当达到 context_length 后：
            # 重新截取最近 context_length tokens 并重新建立 cache
            if X.size(-1) > self.config.model.context_length:
                X = X[:, -self.config.model.context_length:]
                logits, past_kv = self.forward(X, use_cache=True, start_pos=0)
                cache_len = self.config.model.context_length
                continue
            
            # 6. KV Cache Decode
            logits, past_kv = (
                self.forward(
                    next_token_id,
                    past_kv=past_kv,
                    use_cache=True,
                    start_pos=cache_len
                )
            )
            cache_len += 1

    def chat(self, message, tokenizer, history=None, **kwargs):
        messages = history or []
        messages.append({"role": "user", "content": message})

        # 手动构造 prompt：用模板渲染对话，但不加 generation_prompt
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        # 只加 assistant 标记，不加 <think> 标签
        prompt += "<|im_start|>assistant\n"

        # 以模型参数的实际设备为准；调用方可能已通过 model.to(...) 将模型移到
        # generation.device，不能继续使用训练阶段的 system.device。
        model_device = self.embedding.weight.device
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model_device)

        # generate() 内部现在自动使用 KV Cache
        output_ids = self.generate(input_ids,eos_token_id=tokenizer.eos_token_id,**kwargs)
        response = tokenizer.decode(output_ids[0], skip_special_tokens=True,)
        return response
