import os
import glob
import argparse
import torch
import time
from typing import IO, BinaryIO
from transformers import BatchEncoding, PreTrainedTokenizer
from minisnail.config import SnailConfig
from minisnail.model import init_model, SnailModel
from minisnail.tokenizer import get_tokenizer
from minisnail.debug import console

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
    checkpoint = torch.load(src, weights_only=False)
    
    return checkpoint

def generate_text(model: SnailModel, tokenizer: PreTrainedTokenizer, prompt: str, max_tokens: int = 512, config: SnailConfig = None, device: torch.device = None):
    '''Generate text output by the model.
    '''
    device = torch.device(config.generation.device) if device is None else device
    
    model.to(device)
    model.eval()

    # 预训练序列以 bos_token (<|im_start|>) 开头, 生成时也必须加上, 否则输出退化为空/循环
    input_ids = [tokenizer.bos_token_id] + tokenizer(prompt)['input_ids']

    prompt_ids: list[int] = input_ids
    prompt_tensor: torch.Tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    start_time = time.time()
    with torch.no_grad():
        output_ids_tensor = model.generate(
            prompt_tensor,
            max_tokens=max_tokens,
            temperature=config.generation.temperature,
            top_k=config.generation.top_k,
            top_p=config.generation.top_p,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=config.generation.repetition_penalty,
            do_sample=not config.generation.greedy,
        )

        # console.print("Output token ids:")
        # console.print(output_ids_tensor)

        output_ids: list[int] = output_ids_tensor[0].cpu().numpy().tolist()

        # ==== Merge the original prompt and the generated content ====
        full_ids = prompt_ids + output_ids
        text = tokenizer.decode(full_ids, skip_special_tokens=True)
        console.print("🤖:")
        console.print(text)
    end_time = time.time()
    console.print(f"Generation time: {end_time - start_time:.2f} seconds")

def streaming_generate_text(model: SnailModel, tokenizer: PreTrainedTokenizer, prompt: str, max_tokens: int = 512, config: SnailConfig = None, device: torch.device = None):
    '''Generate text output by the model.
    '''
    device = torch.device(config.generation.device) if device is None else device

    prompt_ids: list[int] = [tokenizer.bos_token_id] + tokenizer(prompt)['input_ids']
    prompt_tensor: torch.Tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    start_time = time.time()
    with torch.no_grad():
        for next_token_id in model.streaming_generate(
            prompt_tensor,
            max_tokens=max_tokens,
            temperature=config.generation.temperature,
            top_k=config.generation.top_k,
            top_p=config.generation.top_p,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=config.generation.repetition_penalty,
            do_sample=not config.generation.greedy,
        ):
            yield tokenizer.decode([next_token_id], skip_special_tokens=True)
    console.print("\n")
    end_time = time.time()
    console.print(f"Generation time: {end_time - start_time:.2f} seconds")

# 批量评测用测试题：覆盖事实记忆、常识与简单算术
# (预训练语料为通用中文问答/百科文本, 这些题目能直接反映模型学到的知识)
TEST_PROMPTS = [
    "请回答以下问题：中国的首都是哪座城市？",
    "请计算：24 乘以 8 等于多少？",
    "一年共有多少个月？",
    "水的化学式是什么？",
    "太阳是从哪个方向升起的？",
    "请回答：唐朝的开国皇帝是谁？",
    "地球上最大的动物是什么？",
    "一天有多少个小时？",
]

def extract_step(model_path: str) -> int:
    """从文件名 pretrain_lm_68000.pt 中提取训练步数"""
    name = os.path.basename(model_path)
    digits = name.replace("pretrain_lm_", "").replace(".pt", "")
    return int(digits)

def batch_eval(models_dir: str, config: SnailConfig, tokenizer: PreTrainedTokenizer, prompts: list[str] | None = None, greedy: bool | None = None):
    """遍历模型目录下所有 pretrain_lm_*.pt, 按训练步数升序逐个评测并生成文本。

    默认沿用 config.generation.greedy 的采样设置 (与脚本原单模型测试一致)。
    """
    prompts = prompts if prompts is not None else TEST_PROMPTS
    if greedy is None:
        greedy = config.generation.greedy
    model_files = glob.glob(os.path.join(models_dir, "pretrain_lm_*.pt"))
    model_files = sorted(model_files, key=extract_step)

    if not model_files:
        console.print(f"[red]No pretrain_lm_*.pt found in {models_dir}")
        return

    # 结果同时打印到终端并写入 UTF-8 文件, 避免 Windows GBK 控制台的中文乱码
    out_lines: list[str] = []
    def emit(text: str):
        out_lines.append(text)
        console.print(text)

    emit(f"Found {len(model_files)} models in {models_dir}:")
    for mf in model_files:
        emit(f"  - {os.path.basename(mf)} (step {extract_step(mf)})")

    device = torch.device(config.generation.device)

    for mf in model_files:
        step = extract_step(mf)
        emit("")
        emit(f"{'=' * 60}")
        emit(f"模型 step={step}  ({os.path.basename(mf)})")
        emit(f"{'=' * 60}")

        model = init_model(config, model_path=mf, device=device)
        model.eval()

        with torch.no_grad():
            for prompt_index, p in enumerate(prompts):
                # 每个 prompt 独立重置 RNG，防止前一模型提前停止后影响后续比较。
                torch.manual_seed(config.system.seed + prompt_index)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(config.system.seed + prompt_index)
                input_ids: list[int] = [tokenizer.bos_token_id] + tokenizer(p)['input_ids']
                prompt_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
                output_ids_tensor = model.generate(
                    prompt_tensor,
                    max_tokens=128,
                    temperature=config.generation.temperature,
                    top_k=config.generation.top_k,
                    top_p=config.generation.top_p,
                    eos_token_id=tokenizer.eos_token_id,
                    repetition_penalty=config.generation.repetition_penalty,
                    do_sample=not greedy,
                )
                gen_ids: list[int] = output_ids_tensor[0].cpu().numpy().tolist()
                answer = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
                emit(f"  Q: {p}")
                emit(f"  A: {answer}")

        del model
        torch.cuda.empty_cache()

    result_path = os.path.join(models_dir, "batch_eval_result.txt")
    with open(result_path, "w", encoding="utf-8") as f:
        f.write("\n".join(out_lines))
    console.print(f"[bold green]Results written to {result_path}[/bold green]")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Load pretrain model from checkpoint.')
    parser.add_argument('--model', type=str, default=None,
                        help='Path to the model file (default: config.generation.model_path).')
    parser.add_argument('--eval_dir', type=str, default=None, help='批量评测目录: 遍历 <dir>/pretrain_lm_*.pt 并逐一生成文本对比.')
    args = parser.parse_args()
    
    config = SnailConfig.from_json("./config.json")
    tokenizer: PreTrainedTokenizer = get_tokenizer(config)

    if args.eval_dir is not None:
        batch_eval(args.eval_dir, config, tokenizer)
        raise SystemExit(0)
    # The model will load the weight from config.training.from_weight
    model_path: str = args.model or config.generation.model_path
    device = torch.device(config.generation.device)
    model: SnailModel = init_model(config, model_path=model_path, device=device)
    
    console.print("[yellow]Loading model from weight:", model_path)
    
    model.eval()
    prompt: str = ""

    # Pre-Training Test
    while True:
        prompt = input("👩：")
        if prompt == "exit":
            break
        # for token_str in streaming_generate_text(model, tokenizer, prompt, max_tokens=512, config=config):
        #     console.print(token_str, end="")
        generate_text(model, tokenizer, prompt, max_tokens=512, config=config)

