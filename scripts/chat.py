import argparse
import os
import sys
import time

import torch
from rich.console import Console
from transformers import PreTrainedTokenizer, TextStreamer

from minisnail.chat_protocol import encode_chat_prompt
from minisnail.config import SnailConfig
from minisnail.model import SnailModel, init_model
from minisnail.tokenizer import get_tokenizer
from minisnail.util import setup_seed


console = Console()


def _build_input_ids(model, tokenizer, message, history=None):
    messages = list(history or [])
    messages.append({"role": "user", "content": message})
    return torch.tensor(
        [encode_chat_prompt(tokenizer, messages)],
        dtype=torch.long,
        device=model.embedding.weight.device,
    )


def chat_generate(model: SnailModel, tokenizer: PreTrainedTokenizer, message: str,
                  config: SnailConfig, max_tokens: int | None = None,
                  do_sample: bool | None = None, history=None):
    """SFT 对话生成，可选传入多轮对话历史。

    SFT 模型应学会: 1) 按对话模板回答; 2) 回答结束后主动输出 <|im_end|> 停止
    Returns:
        (answer, n_tokens, stopped_early, generation_seconds)
    """
    input_ids = _build_input_ids(model, tokenizer, message, history)

    max_tokens = config.generation.max_tokens if max_tokens is None else max_tokens
    if do_sample is None:
        do_sample = not config.generation.greedy

    start_time = time.time()
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_tokens=max_tokens,
            temperature=config.generation.temperature,
            top_k=config.generation.top_k,
            top_p=config.generation.top_p,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=config.generation.repetition_penalty,
            do_sample=do_sample,
        )
    gen_time = time.time() - start_time

    n_tokens = output_ids.shape[-1]
    answer = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
    # 提前停止 = 生成的 token 数未达到上限, 即模型主动输出了 <|im_end|>
    stopped_early = n_tokens < max_tokens
    return answer, n_tokens, stopped_early, gen_time


def chat_generate_streaming(
    model: SnailModel,
    tokenizer: PreTrainedTokenizer,
    message: str,
    config: SnailConfig,
    max_tokens: int | None = None,
    do_sample: bool | None = None,
    history=None,
):
    """边生成边写入终端，并返回完整回答及统计信息。"""
    input_ids = _build_input_ids(model, tokenizer, message, history)

    max_tokens = config.generation.max_tokens if max_tokens is None else max_tokens
    if do_sample is None:
        do_sample = not config.generation.greedy

    # TextStreamer 会缓冲子词，避免逐 token decode 造成乱码、丢空格或
    # SentencePiece/BPE 的半个字符被提前打印。
    streamer = TextStreamer(
        tokenizer,
        skip_prompt=False,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    generated_ids: list[int] = []
    start_time = time.perf_counter()
    try:
        with torch.no_grad():
            for output_id in model.streaming_generate(
                input_ids,
                max_tokens=max_tokens,
                temperature=config.generation.temperature,
                top_k=config.generation.top_k,
                top_p=config.generation.top_p,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=config.generation.repetition_penalty,
                do_sample=do_sample,
            ):
                token_id = int(output_id)
                generated_ids.append(token_id)
                streamer.put(torch.tensor([token_id]))
    finally:
        streamer.end()

    gen_time = time.perf_counter() - start_time
    answer = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    ).strip()
    n_tokens = len(generated_ids)
    stopped_early = n_tokens < max_tokens
    return answer, n_tokens, stopped_early, gen_time


def parse_args():
    parser = argparse.ArgumentParser(description='Chat with MiniSnail in the terminal.')
    parser.add_argument('--model', type=str, default=None,
                        help='Path to model weights (default: config.generation.model_path).')
    parser.add_argument('--config', type=str, default='./config.json', help='Path to config JSON file.')
    parser.add_argument('--greedy', action='store_true', help='强制贪婪解码 (结果更稳定, 便于对比)')
    parser.add_argument('--max_tokens', type=int, default=None,
                        help='每轮最大生成 token 数 (默认使用 config.generation.max_tokens)')
    parser.add_argument('--single-turn', action='store_true',
                        help='不保留多轮对话上下文')
    args = parser.parse_args()
    if args.max_tokens is not None and args.max_tokens <= 0:
        parser.error('--max_tokens 必须是正整数')
    return args


if __name__ == '__main__':
    # Windows GBK 控制台无法编码 emoji: 保留终端编码, 将无法编码的字符降级为 ? (保证中文正常显示)
    try:
        sys.stdout.reconfigure(errors='replace')
        sys.stderr.reconfigure(errors='replace')
    except AttributeError:
        pass

    args = parse_args()
    config = SnailConfig.from_json(args.config)
    tokenizer: PreTrainedTokenizer = get_tokenizer(config)

    # 固定随机种子, 保证采样结果可复现
    setup_seed(config.system.seed)

    model_path: str = args.model or config.generation.model_path
    if not os.path.exists(model_path):
        console.print("Model file not found:", model_path, style="red")
        raise SystemExit(1)

    model: SnailModel = init_model(
        config,
        model_path=model_path,
        device=torch.device(config.generation.device),
    )
    console.print("[yellow]Loaded model weights:[/yellow]", model_path)

    model.eval()

    do_sample = False if args.greedy else None

    history: list[dict[str, str]] = []
    console.print("[dim]输入 /clear 清空上下文，输入 /exit 退出。[/dim]")

    while True:
        try:
            prompt = input("👩：")
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]已退出。[/dim]")
            break

        command = prompt.strip().lower()
        if command in {"exit", "quit", "/exit", "/quit"}:
            break
        if command == "/clear":
            history.clear()
            console.print("[dim]上下文已清空。[/dim]")
            continue
        if not prompt.strip():
            continue

        console.print("[bold cyan]🤖：[/bold cyan]", end="")
        try:
            answer, n_tokens, stopped_early, gen_time = chat_generate_streaming(
                model,
                tokenizer,
                prompt,
                config,
                max_tokens=args.max_tokens,
                do_sample=do_sample,
                history=None if args.single_turn else history,
            )
        except KeyboardInterrupt:
            console.print("[dim]已中止本轮生成。[/dim]")
            continue

        if not args.single_turn:
            history.extend([
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": answer},
            ])
        stop_reason = "eos" if stopped_early else "length"
        console.print(
            f"[dim]tokens={n_tokens}, time={gen_time:.2f}s, stop={stop_reason}[/dim]"
        )

