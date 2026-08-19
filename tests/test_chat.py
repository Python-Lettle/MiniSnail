import argparse

import torch
import torch.nn.functional as F
from transformers import PreTrainedTokenizer
from minisnail.config import SnailConfig
from minisnail.model import init_model, SnailModel, top_p_filtering
from minisnail.tokenizer import get_tokenizer
from minisnail.debug import console

KVCache = list[tuple[torch.Tensor, torch.Tensor]]


class ChatSession:
    """A single-user chat session with history and reusable KV Cache."""

    def __init__(
        self,
        model: SnailModel,
        tokenizer: PreTrainedTokenizer,
        config: SnailConfig,
        use_kv_cache: bool = True,
    ):
        self.model, self.tokenizer, self.config = model, tokenizer, config
        self.device = torch.device(config.system.device)
        self.use_kv_cache = use_kv_cache
        self.history: list[dict[str, str]] = []
        self.cache_ids: list[int] = []
        self.past_kv: KVCache | None = None

    def reset(self) -> None:
        self.history.clear()
        self.cache_ids.clear()
        self.past_kv = None

    def _prompt_ids(self) -> list[int]:
        """Render all dialogue turns and the header for the next assistant turn."""
        prompt = self.tokenizer.apply_chat_template(
            self.history, tokenize=False, add_generation_prompt=True,
        )
        # The template already contains <|im_start|> / <|im_end|>; avoid adding
        # another BOS/EOS pair. ``tokenize=False`` is deliberate: this project's
        # tokenizer returns a string even when its ``tokenize=True`` is requested.
        ids = self.tokenizer(prompt, add_special_tokens=False)['input_ids']
        if isinstance(ids, torch.Tensor):
            return ids.flatten().tolist()
        if ids and isinstance(ids[0], list):
            return ids[0]
        return list(ids)

    @staticmethod
    def _common_prefix_length(left: list[int], right: list[int]) -> int:
        '''Find the length of the common prefix between two lists of token IDs.'''
        for index, (left_id, right_id) in enumerate(zip(left, right)):
            if left_id != right_id:
                return index
        return min(len(left), len(right))

    def _truncate_cache(self, length: int) -> None:
        '''Truncate the cache to the specified length.'''
        self.cache_ids = self.cache_ids[:length]
        if self.past_kv is not None:
            self.past_kv = [(key[..., :length, :], value[..., :length, :]) for key, value in self.past_kv]

    def _prefill(self, prompt_ids: list[int]) -> torch.Tensor:
        """Run only the prompt suffix which is not already in the KV Cache."""
        prompt_ids = prompt_ids[-self.config.model.context_length:]
        if not self.use_kv_cache:
            self.cache_ids = prompt_ids
            self.past_kv = None
            return self.model.forward(
                torch.tensor([prompt_ids], dtype=torch.long, device=self.device),
                use_cache=False,
            )

        prefix_length = self._common_prefix_length(self.cache_ids, prompt_ids)
        self._truncate_cache(prefix_length)
        suffix_ids: list[int] = prompt_ids[prefix_length:]
        if not suffix_ids:  # Correct fallback for an identical prompt.
            self.past_kv, self.cache_ids = None, []
            suffix_ids, prefix_length = prompt_ids, 0

        suffix = torch.tensor([suffix_ids], dtype=torch.long, device=self.device)
        logits, self.past_kv = self.model.forward(
            suffix, past_kv=self.past_kv, use_cache=True, start_pos=prefix_length,
        )
        self.cache_ids = prompt_ids
        return logits

    def _sample(self, logits: torch.Tensor) -> torch.Tensor:
        generation = self.config.generation
        next_logits = logits[:, -1].clone()
        if generation.repetition_penalty > 1.0 and self.cache_ids:
            next_logits[:, self.cache_ids] /= generation.repetition_penalty
        if generation.greedy:
            return next_logits.argmax(dim=-1, keepdim=True)

        next_logits /= generation.temperature
        if generation.top_k:
            values, _ = torch.topk(next_logits, min(generation.top_k, next_logits.size(-1)))
            next_logits = next_logits.masked_fill(next_logits < values[:, -1:], float('-inf'))
        if generation.top_p < 1.0:
            next_logits = top_p_filtering(next_logits, generation.top_p)
        return torch.multinomial(F.softmax(next_logits, dim=-1), 1)

    def stream_reply(self, message: str):
        self.history.append({'role': 'user', 'content': message})
        logits = self._prefill(self._prompt_ids())
        generated_ids: list[int] = []
        limit = self.config.model.context_length

        for _ in range(self.config.generation.max_tokens):
            next_token = self._sample(logits)
            token_id = next_token.item()
            if token_id == self.tokenizer.eos_token_id:
                break
            generated_ids.append(token_id)
            yield self.tokenizer.decode([token_id], skip_special_tokens=True)

            if not self.use_kv_cache:
                # No KV Cache: recompute the entire active context for every
                # token. This trades substantially more compute for lower VRAM.
                self.cache_ids = (self.cache_ids + [token_id])[-limit:]
                logits = self.model.forward(
                    torch.tensor([self.cache_ids], dtype=torch.long, device=self.device),
                    use_cache=False,
                )
                continue

            if len(self.cache_ids) >= limit:
                # Sliding the context changes RoPE positions: rebuild the cache.
                window_ids = (self.cache_ids + [token_id])[-limit:]
                logits, self.past_kv = self.model.forward(
                    torch.tensor([window_ids], dtype=torch.long, device=self.device),
                    use_cache=True, start_pos=0,
                )
                self.cache_ids = window_ids
            else:
                logits, self.past_kv = self.model.forward(
                    next_token, past_kv=self.past_kv, use_cache=True, start_pos=len(self.cache_ids),
                )
                self.cache_ids.append(token_id)

        response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        self.history.append({'role': 'assistant', 'content': response})

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Interactive MiniSnail chat')
    parser.add_argument(
        '--no-kv-cache', action='store_true',
        help='Disable KV Cache to reduce VRAM usage (slower generation).',
    )
    args = parser.parse_args()

    config = SnailConfig.from_json("./config.json")
    device = torch.device(config.system.device)
    console.print("[yellow]Using device:", device)
    
    tokenizer: PreTrainedTokenizer = get_tokenizer(config)
    # The model will load the weight from config.training.from_weight
    model: SnailModel = init_model(config)

    # Load the model weight
    model_dir = "./model/local_dpo_model_epo2/dpo_new.pt"
    model.load_state_dict(torch.load(model_dir, weights_only=False))
    
    console.print("[yellow]Loading model from weight:", model_dir)
    
    model.eval()
    model.to(device=device)

    session = ChatSession(model, tokenizer, config, use_kv_cache=not args.no_kv_cache)
    cache_status = 'enabled' if session.use_kv_cache else 'disabled (slower, lower VRAM)'
    console.print(f'[yellow]KV Cache: {cache_status}[/yellow]')
    console.print('[dim]Commands: /reset clears the conversation; /exit quits.[/dim]')

    while True:
        prompt = input("👤: ").strip()
        if prompt == '/exit':
            break
        elif prompt == '/reset':
            session.reset()
            print('Conversation cleared.')
            continue
        elif not prompt:
            continue

        print("🤖: ", end='', flush=True)
        for text in session.stream_reply(prompt):
            print(text, end='', flush=True)
        print()
