import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor
from einops import rearrange, einsum

from minisnail.config import SnailConfig
from minisnail.debug import console
from minisnail.chat_protocol import encode_chat_prompt, fit_chat_messages

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
    # n-gram 大小
    n = no_repeat_ngram_size
    # 如果 n-gram 大小无效, 不做处理, 直接返回
    # 如果 seq 长度不足 n, 不做处理, 也直接返回
    if not n or n < 2 or len(seq) < n:
        return next_token_logits
    
    # 最后 n-1 个 token 作为前缀, 查找所有与前缀匹配的 token
    # 把这些 token 的 logit 置为 -inf

    # 取出当前序列最后的 n - 1 个 token
    prefix = tuple(seq[-(n - 1):])
    # 查找历史上相同前缀后面接过哪些 token
    banned = {
        seq[i + n - 1]
        for i in range(len(seq) - n + 1)
        if tuple(seq[i:i + n - 1]) == prefix
    }
    # 把被禁止 token 的分数设为负无穷
    for t in banned:
        next_token_logits[:, t] = float("-inf")
    # 返回处理后的 logits
    return next_token_logits


def _apply_repetition_penalty(next_token_logits, seen_token_ids, repetition_penalty):
    """对已出现过的 token 各应用一次 repetition penalty。

    正 logit 除以 penalty，负 logit 乘以 penalty，保证两种情况都会降低
    token 的选中概率。seen_token_ids 使用集合去重，避免按出现次数重复惩罚。
    """
    # 如果 penalty 无效, 不做处理, 直接返回
    # 如果 seen_token_ids 为空, 不做处理, 直接返回
    if repetition_penalty <= 1.0 or not seen_token_ids:
        return next_token_logits

    # 取出所有已出现过的 token 的 logit
    token_ids = list(seen_token_ids)
    token_logits = next_token_logits[:, token_ids]
    
    # 对每个 token 的 logit 应用 penalty
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

    def _generate_tokens(self, X: torch.Tensor, max_tokens: int = 512, temperature: float = 0.85,
                            repetition_penalty: float = 1.2, top_k: int = 50, top_p: float = 0.9,
                            eos_token_id: int = 2, do_sample: bool = True, no_repeat_ngram_size: int = 0,
                            suppress_token_ids=(), allow_context_rollover: bool = True):
        # 简写参数
        context_length = self.config.model.context_length
        # 检查输入参数
        if X.dim() == 1:
            X = X.unsqueeze(0)
        if X.dim() != 2 or X.shape[0] != 1 or X.shape[1] == 0:
            raise ValueError("生成接口需要一条非空输入，batch size 必须为 1")
        if X.shape[1] > context_length:
            raise ValueError("输入超过上下文窗口；请先按完整消息裁剪或缩短预训练提示")
        if not isinstance(max_tokens, int) or max_tokens < 1:
            raise ValueError("max_tokens 必须为正整数")
        if not math.isfinite(temperature) or (do_sample and temperature <= 0):
            raise ValueError("采样 temperature 必须为有限正数；贪婪解码请使用 do_sample=False")
        if not isinstance(top_k, int) or top_k < 0 or not 0 < top_p <= 1:
            raise ValueError("top_k 必须为非负整数，top_p 必须在 (0, 1] 内")
        if not math.isfinite(repetition_penalty) or repetition_penalty < 1:
            raise ValueError("repetition_penalty 必须为不小于 1 的有限值")
        if not isinstance(no_repeat_ngram_size, int) or no_repeat_ngram_size < 0:
            raise ValueError("no_repeat_ngram_size 必须为非负整数")
        
        # forbidden 保存明确禁止输出的 token ID, 例如某些特殊控制符
        forbidden = set(suppress_token_ids)
        # 将 EOS 移除, 避免禁止列表把 EOS 也屏蔽掉
        forbidden.discard(eos_token_id)
        
        # 检查禁止列表中的 token ID 是否超出词表范围
        if any(token_id < 0 or token_id >= self.config.model.vocab_size for token_id in forbidden):
            raise ValueError("suppress_token_ids 超出词表范围")

        # 确保 X 的类型和设备与模型参数一致
        X = X.to(device=self.embedding.weight.device, dtype=torch.long)

        # 初始化生成记录
        generated = []
        self.last_generation_info = {"stop_reason": "length", "generated_ids": generated}
        # 完整提示词送进模型
        with torch.no_grad():
            # 第一次把整个 prompt 输入模型
            # 得到：
            #   logits: 各个位置对下一个 token 的预测分数
            #   past_kv: 已处理 token 的注意力 Key、Value 缓存
            logits, past_kv = self.forward(X, use_cache=True, start_pos=0)
            cache_len = X.shape[1]

            for step in range(max_tokens):
                # 为了预测提示词后面的第一个新 token，只需要最后一个位置的输出
                # .float() 转成 float32
                # .clone() 创建独立副本，方便后面修改分数。
                scores = logits[:, -1].float().clone()
                # 检查 scores, 允许负无穷, -inf 可以表示“这个 token 不可选”
                if torch.isnan(scores).any() or torch.isposinf(scores).any():
                    raise ValueError("模型产生非有限 logits")

                # 已经生成过的普通 token 集合, 用于重复惩罚
                seen = set(generated) - {eos_token_id} - forbidden

                # 对 scores 施加重复惩罚
                scores = _apply_repetition_penalty(scores, seen, repetition_penalty)
                scores = _no_repeat_ngram_mask(scores, generated, no_repeat_ngram_size)

                # 对 scores 应用禁止列表
                if forbidden: scores[:, list(forbidden)] = -float("inf")

                # 检查是否所有候选 token 都被屏蔽了
                if not torch.isfinite(scores).any(): raise ValueError("生成规则屏蔽了所有候选 token")

                # 采样下一个 token
                if do_sample:
                    # 对 scores 应用温度
                    scores /= temperature
                    # Top-K 过滤
                    if top_k:
                        threshold = torch.topk(scores, min(top_k, scores.shape[-1])).values[:, -1:]
                        scores = scores.masked_fill(scores < threshold, -float("inf"))
                    # Top-P 过滤
                    scores = top_p_filtering(scores, top_p)
                    # 计算概率
                    probabilities = F.softmax(scores, dim=-1)

                    if not torch.isfinite(probabilities).all(): raise ValueError("模型产生非有限采样概率")

                    next_id = torch.multinomial(probabilities, 1)
                else:
                    next_id = scores.argmax(dim=-1, keepdim=True)

                token = next_id.item()

                if eos_token_id is not None and token == eos_token_id:
                    self.last_generation_info["stop_reason"] = "eos"
                    return
                # 记录当前生成的 token
                generated.append(token)
                # 返回当前生成的 token
                yield token
                # 检查是否超过最大 token 数, 如果超过则停止生成
                if step + 1 == max_tokens: return
                # 更新上下文, 为下一轮预测做准备
                X = torch.cat((X, next_id), dim=-1)

                if X.shape[1] > context_length:
                    # 如果 prompt 长度超过 context length, 则检查是否允许 rolloverver
                    if not allow_context_rollover:
                        self.last_generation_info["stop_reason"] = "context_length"
                        return
                    # 如果允许 rolloverver, 则截断上下文, 保持 context length
                    X = X[:, -context_length:]
                    logits, past_kv = self.forward(X, use_cache=True, start_pos=0)
                    cache_len = context_length
                else:
                    logits, past_kv = self.forward(next_id, past_kv=past_kv,
                                                  use_cache=True, start_pos=cache_len)
                    cache_len += 1

    @torch.no_grad()
    def generate(self,
            X: torch.Tensor, max_tokens=512, temperature=0.85,
            repetition_penalty=1.2, top_k=50, top_p=0.9,
            eos_token_id=2, do_sample=True, skip_prompt=True,
            no_repeat_ngram_size=0, suppress_token_ids=(),
            allow_context_rollover=True,
        ):
        original: torch.Tensor = X.unsqueeze(0) if X.dim() == 1 else X
        ids = list(self._generate_tokens(
            X, max_tokens=max_tokens, temperature=temperature,
            repetition_penalty=repetition_penalty, top_k=top_k, top_p=top_p,
            eos_token_id=eos_token_id, do_sample=do_sample,
            no_repeat_ngram_size=no_repeat_ngram_size,
            suppress_token_ids=suppress_token_ids,
            allow_context_rollover=allow_context_rollover,
        ))
        output = torch.tensor([ids], device=self.embedding.weight.device, dtype=torch.long)
        return output if skip_prompt else torch.cat((original.to(output.device).long(), output), dim=1)

    @torch.no_grad()
    def streaming_generate(self, X: torch.Tensor, max_tokens=512, temperature=0.85,
                           repetition_penalty=1.2, top_k=50, top_p=0.9,
                           eos_token_id=2, do_sample=True, no_repeat_ngram_size=0,
                           suppress_token_ids=(), allow_context_rollover=True):
        yield from self._generate_tokens(
            X, max_tokens=max_tokens, temperature=temperature,
            repetition_penalty=repetition_penalty, top_k=top_k, top_p=top_p,
            eos_token_id=eos_token_id, do_sample=do_sample,
            no_repeat_ngram_size=no_repeat_ngram_size,
            suppress_token_ids=suppress_token_ids,
            allow_context_rollover=allow_context_rollover,
        )

    def chat(self, message, tokenizer, history=None, **kwargs):
        # 合并历史和当前问题
        messages = list(history or []) + [{"role": "user", "content": message}]
        # 用于裁剪
        budget = kwargs.setdefault("max_tokens", self.config.model.context_length // 2)
        messages = fit_chat_messages(
            tokenizer, messages, self.config.model.context_length,
            reserve_tokens=budget,
        )
        kwargs["allow_context_rollover"] = False

        # 以模型参数的实际设备为准；调用方可能已通过 model.to(...) 将模型移到
        # generation.device，不能继续使用训练阶段的 system.device。
        model_device = self.embedding.weight.device
        input_ids = torch.tensor(
            [encode_chat_prompt(tokenizer, messages)],
            dtype=torch.long,
            device=model_device,
        )

        # generate() 内部现在自动使用 KV Cache
        output_ids = self.generate(input_ids,eos_token_id=tokenizer.eos_token_id,**kwargs)
        response = tokenizer.decode(output_ids[0], skip_special_tokens=True,)
        return response
