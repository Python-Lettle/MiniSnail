import os
import glob
import time
import argparse
import torch
from typing import IO, BinaryIO
from transformers import PreTrainedTokenizer
from minisnail.config import SnailConfig
from minisnail.model import init_model, SnailModel
from minisnail.tokenizer import get_tokenizer
from minisnail.debug import console
from minisnail.chat_protocol import encode_chat_prompt

def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
) -> dict[str, any]:
    """
    Given a serialized checkpoint (path or file-like object), restore the
    serialized state to the given model and optimizer.
    Return the checkpoint state.

    Args:
        src (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialized checkpoint.
    Returns:
        dict[str, any]: A dictionary of the checkpoint state.
    """
    # Load the checkpoint from the file or object
    if isinstance(src, str) or isinstance(src, os.PathLike):
        src = open(src, 'rb')
    # Load the model state from the checkpoint
    checkpoint = torch.load(src, map_location="cpu", weights_only=False)

    return checkpoint

def chat_generate(model: SnailModel, tokenizer: PreTrainedTokenizer, message: str,
                  config: SnailConfig, max_tokens: int | None = None,
                  do_sample: bool | None = None):
    """SFT 对话生成 (单轮, 不保留上下文, 与 model.chat 相同的 prompt 构造, 额外返回 token 数和是否提前停止)

    SFT 模型应学会: 1) 按对话模板回答; 2) 回答结束后主动输出 <|im_end|> 停止
    Returns:
        (answer, n_tokens, stopped_early)
    """
    messages = [{"role": "user", "content": message}]
    input_ids = torch.tensor(
        [encode_chat_prompt(tokenizer, messages)],
        dtype=torch.long,
        device=model.embedding.weight.device,
    )

    max_tokens = max_tokens or config.generation.max_tokens
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

# SFT 评测题: 覆盖事实问答 / 常识 / 简单解释 (英文为主, 对齐 SFT 数据域, 辅以中文)
TEST_PROMPTS = [
    "What is the primary function of the Domain Name System (DNS) in the internet?",
    "What is the capital of France?",
    "Explain the difference between HTTP and HTTPS in one sentence.",
    "How many days are there in a leap year?",
    "一年有多少个月？",
    "请用一句话解释什么是机器学习。",
]

def auto_test(model: SnailModel, tokenizer: PreTrainedTokenizer, config: SnailConfig,
              prompts: list[str], max_tokens: int = 128, do_sample: bool | None = None,
              emit=None):
    """自动测试: 对固定测试题逐一生成回答, 统计 <|im_end|> 停止率

    emit 为 None 时直接打印 (写终端), 否则打印并写入结果列表
    Returns:
        (n_stopped, lines): 停止数, 输出文本行列表
    """
    if emit is None:
        out_lines: list[str] = []
        emit = lambda text: (out_lines.append(text), console.print(text))
        owns_lines = True
    else:
        out_lines = None
        owns_lines = False

    n_stopped = 0
    with torch.no_grad():
        for prompt_index, p in enumerate(prompts):
            torch.manual_seed(config.system.seed + prompt_index)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(config.system.seed + prompt_index)
            answer, n_tokens, stopped_early, gen_time = chat_generate(
                model, tokenizer, p, config, max_tokens=max_tokens, do_sample=do_sample)
            n_stopped += int(stopped_early)
            emit(f"  Q: {p}")
            emit(f"  A: {answer}")
            emit(f"  [tokens={n_tokens}, stopped_early={stopped_early}, time={gen_time:.2f}s]")
            emit("")
    emit(f"停止率 (提前输出 <|im_end|>): {n_stopped}/{len(prompts)}")

    if owns_lines:
        return n_stopped, out_lines
    return n_stopped, None

def batch_eval(models: list[str], config: SnailConfig, tokenizer: PreTrainedTokenizer,
               prompts: list[str] | None = None, max_tokens: int = 128, do_sample: bool | None = None):
    """批量评测: 遍历模型列表逐一自动测试, 结果写入文件 (与预训练测试方案一致)"""
    prompts = prompts if prompts is not None else TEST_PROMPTS

    # 结果同时打印到终端并写入 UTF-8 文件, 避免 Windows GBK 控制台的中文乱码
    out_lines: list[str] = []
    def emit(text: str):
        out_lines.append(text)
        console.print(text)

    emit(f"Evaluating {len(models)} SFT model(s) with {len(prompts)} prompts each")
    device = torch.device(config.generation.device)
    for mf in models:
        emit("")
        emit(f"{'=' * 60}")
        emit(f"模型: {os.path.basename(mf)}")
        emit(f"{'=' * 60}")

        model = init_model(config, model_path=mf, device=device)
        model.eval()

        auto_test(model, tokenizer, config, prompts, max_tokens=max_tokens,
                  do_sample=do_sample, emit=emit)

        del model
        torch.cuda.empty_cache()

    # 写入结果文件 (与第一个模型同目录)
    result_path = os.path.join(os.path.dirname(os.path.abspath(models[0])), "sft_eval_result.txt")
    with open(result_path, "w", encoding="utf-8") as f:
        f.write("\n".join(out_lines))
    console.print(f"[bold green]Results written to {result_path}[/bold green]")

if __name__ == '__main__':
    import sys
    # Windows GBK 控制台无法编码 emoji: 保留终端编码, 将无法编码的字符降级为 ? (保证中文正常显示)
    try:
        sys.stdout.reconfigure(errors='replace')
        sys.stderr.reconfigure(errors='replace')
    except AttributeError:
        pass

    parser = argparse.ArgumentParser(description='Test model with chat template.')
    parser.add_argument('--model', type=str, default=None,
                        help='Path to model weights (default: config.generation.model_path).')
    parser.add_argument('--eval_dir', type=str, default=None,
                        help='批量评测目录: 遍历 <dir>/sft_*.pt 并逐一对话评测对比.')
    parser.add_argument('--config', type=str, default='./config.json', help='Path to config JSON file.')
    parser.add_argument('--greedy', action='store_true', help='强制贪婪解码 (结果更稳定, 便于对比)')
    parser.add_argument('--max_tokens', type=int, default=128, help='批量评测时每题最大生成 token 数')
    args = parser.parse_args()

    config = SnailConfig.from_json(args.config)
    tokenizer: PreTrainedTokenizer = get_tokenizer(config)

    # 固定随机种子, 保证采样结果可复现
    torch.manual_seed(config.system.seed)

    if args.eval_dir is not None:
        model_files = sorted(glob.glob(os.path.join(args.eval_dir, "sft_*.pt")))
        if not model_files:
            console.print(f"[red]No sft_*.pt found in {args.eval_dir}")
            raise SystemExit(1)
        console.print(f"Found {len(model_files)} model(s) in {args.eval_dir}:")
        for mf in model_files:
            console.print(f"  - {os.path.basename(mf)}")
        batch_eval(model_files, config, tokenizer, max_tokens=args.max_tokens,
                   do_sample=False if args.greedy else None)
        raise SystemExit(0)

    model_path: str = args.model or config.generation.model_path
    if not os.path.exists(model_path):
        console.print(f"[red]Model file not found: {model_path}")
        raise SystemExit(1)

    model: SnailModel = init_model(
        config,
        model_path=model_path,
        device=torch.device(config.generation.device),
    )
    console.print("[yellow]Loading model from weight:", model_path)

    model.eval()

    # 用户选择测试模式 (与预训练测试方案一致: --eval_dir 走批量自动评测, 单模型可选自动/手动)
    console.print("[bold cyan]测试模式:")
    console.print("  [1] 自动测试 (固定测试题, 统计 <|im_end|> 停止率, 结果写入 sft_eval_result.txt)")
    console.print("  [2] 手动测试 (交互式单轮对话, 不保留上下文)")
    choice = input("请选择 [1/2] (默认 1): ").strip()
    do_sample = False if args.greedy else None

    if choice == "2":
        # 手动测试: 交互式单轮对话 (不保留上下文, 每段对话独立)
        while True:
            prompt = input("👩：")
            if prompt == "exit":
                break
            answer, n_tokens, _, gen_time = chat_generate(model, tokenizer, prompt, config)
            console.print(f"🤖: {answer}")
            console.print(f"[dim][tokens={n_tokens}, time={gen_time:.2f}s][/dim]")
    else:
        # 自动测试: 固定测试题 + 停止率统计, 结果写入模型同目录的 sft_eval_result.txt
        _, lines = auto_test(model, tokenizer, config, TEST_PROMPTS,
                             max_tokens=args.max_tokens, do_sample=do_sample)
        result_path = os.path.join(os.path.dirname(os.path.abspath(model_path)), "sft_eval_result.txt")
        with open(result_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        console.print(f"[bold green]Results written to {result_path}[/bold green]")
