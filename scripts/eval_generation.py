import os
import re
import json
import glob
import argparse
import datetime
import unicodedata

import torch
from tqdm import tqdm

from minisnail.config import SnailConfig
from minisnail.tokenizer import get_tokenizer
from minisnail.model import init_model
from minisnail.chat_protocol import encode_chat_prompt


# 自建的生成式测试集: 覆盖事实知识 / 算术 / 常识 / 语言 / 文化 / 推理 / 科学。
# gold 为参考答案, 用于人工判读; 开放题 gold 置为 null。
TEST_CASES = [
    {"id": 1,  "category": "fact",        "prompt": "中国的首都是哪座城市？", "gold": "北京"},
    {"id": 2,  "category": "fact",        "prompt": "一年有多少个月？", "gold": "12个月"},
    {"id": 3,  "category": "fact",        "prompt": "水的化学式是什么？", "gold": "H2O"},
    {"id": 4,  "category": "fact",        "prompt": "太阳从哪个方向升起？", "gold": "东方"},
    {"id": 5,  "category": "fact",        "prompt": "中国最长的河流是哪条？", "gold": "长江"},
    {"id": 6,  "category": "fact",        "prompt": "世界上面积最大的海洋是哪个？", "gold": "太平洋"},
    {"id": 7,  "category": "math",        "prompt": "计算：3 + 5 等于多少？", "gold": "8"},
    {"id": 8,  "category": "math",        "prompt": "计算：12 乘以 12 等于多少？", "gold": "144"},
    {"id": 9,  "category": "math",        "prompt": "计算：100 减去 37 等于多少？", "gold": "63"},
    {"id": 10, "category": "math",        "prompt": "计算：7 乘以 8 等于多少？", "gold": "56"},
    {"id": 11, "category": "math",        "prompt": "计算：2 的 10 次方等于多少？", "gold": "1024"},
    {"id": 12, "category": "commonsense", "prompt": "一天有多少个小时？", "gold": "24小时"},
    {"id": 13, "category": "commonsense", "prompt": "一个月最多有多少天？", "gold": "31天"},
    {"id": 14, "category": "commonsense", "prompt": "如果外面正在下雨，出门时应该带什么？", "gold": "雨伞"},
    {"id": 15, "category": "commonsense", "prompt": "汽车快没油了，应该怎么办？", "gold": "去加油站加油"},
    {"id": 16, "category": "language",    "prompt": "“高兴”的反义词是什么？", "gold": "伤心"},
    {"id": 17, "category": "language",    "prompt": "请用“美丽”这个词语造一个句子。", "gold": None},
    {"id": 18, "category": "culture",     "prompt": "中国的国庆节是几月几日？", "gold": "10月1日"},
    {"id": 19, "category": "culture",     "prompt": "“床前明月光”的下一句是什么？", "gold": "疑是地上霜"},
    {"id": 20, "category": "culture",     "prompt": "《静夜思》这首诗的作者是谁？", "gold": "李白"},
    {"id": 21, "category": "reasoning",   "prompt": "小明有 5 个苹果，送给小红 2 个，小明还剩几个苹果？", "gold": "3个"},
    {"id": 22, "category": "reasoning",   "prompt": "一个篮球 20 元，买两个篮球一共需要多少钱？", "gold": "40元"},
    {"id": 23, "category": "science",     "prompt": "地球绕着哪颗星星转？", "gold": "太阳"},
    {"id": 24, "category": "science",     "prompt": "一年有几个季节？", "gold": "四个"},
]


def extract_step(model_path: str) -> int:
    m = re.search(r"(\d+)\.pt$", os.path.basename(model_path))
    return int(m.group(1)) if m else 0


def resolve_model_files(configured_model_path, model_path=None, models_dir=None,
                        pattern="pretrain_lm_*.pt") -> list[str]:
    """解析评测权重；显式参数优先，否则使用 generation.model_path。"""
    if model_path is not None and models_dir is not None:
        raise ValueError("--model_path 与 --models_dir 不能同时使用")
    if models_dir is not None:
        return sorted(
            glob.glob(os.path.join(models_dir, pattern)),
            key=extract_step,
        )
    selected_path = model_path or configured_model_path
    return [selected_path] if selected_path else []


def seed_generation(seed: int) -> None:
    """为每个 prompt 独立重置 CPU/CUDA RNG，确保模型横向比较公平。"""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_prompt_ids(tokenizer, prompt: str, prompt_format: str) -> list[int]:
    """Build an explicit pretrain or chat prompt; never mix the two protocols."""
    if prompt_format == "pretrain":
        encoded = tokenizer(prompt, add_special_tokens=False)
        prompt_ids = encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids
        return [tokenizer.bos_token_id] + list(prompt_ids)
    if prompt_format == "chat":
        return encode_chat_prompt(tokenizer, [{"role": "user", "content": prompt}])
    raise ValueError(f"未知 prompt_format: {prompt_format}")


def normalize_answer(text: str) -> str:
    """Normalize a short answer for a conservative reference-in-output check."""
    text = unicodedata.normalize("NFKC", text).casefold()
    return "".join(char for char in text if char.isalnum())


def reference_hit(gold: str | None, generated: str) -> bool | None:
    """Return whether a non-empty normalized reference occurs in the output.

    This is a smoke-test metric, not benchmark accuracy: open-ended questions
    deliberately return ``None`` and remain available for manual review.
    """
    if gold is None:
        return None
    normalized_gold = normalize_answer(gold)
    return bool(normalized_gold) and normalized_gold in normalize_answer(generated)


def generate_answer(
    model, tokenizer, prompt, config, device, max_tokens, do_sample, prompt_format
):
    input_ids = build_prompt_ids(tokenizer, prompt, prompt_format)
    prompt_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
    with torch.no_grad():
        out = model.generate(
            prompt_tensor,
            max_tokens=max_tokens,
            temperature=config.generation.temperature,
            top_k=config.generation.top_k,
            top_p=config.generation.top_p,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=config.generation.repetition_penalty,
            do_sample=do_sample,
        )
    gen_ids = out[0].cpu().tolist()
    answer = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    return answer, len(gen_ids), len(gen_ids) < max_tokens


def main():
    parser = argparse.ArgumentParser(description="自建测试集上的生成式评测 (按提示词生成并保存答案, 供人工判读)")
    parser.add_argument("--config", type=str, default="./config.json")
    parser.add_argument("--model_path", type=str, default=None,
                        help="评测单个权重；默认使用 config.generation.model_path")
    parser.add_argument("--models_dir", type=str, default=None,
                        help="评测目录中所有匹配 --pattern 的权重")
    parser.add_argument("--pattern", type=str, default="pretrain_lm_*.pt")
    parser.add_argument(
        "--prompt_format",
        choices=["pretrain", "chat"],
        required=True,
        help="pretrain=原始文本+BOS；chat=MiniSnail 对话协议。必须显式指定，防止评测错协议",
    )
    parser.add_argument(
        "--greedy",
        action="store_true",
        help="强制贪婪解码；模型横向比较时应启用",
    )
    parser.add_argument("--max_tokens", type=int, default=None,
                        help="每题最大生成 token 数；默认使用 config.generation.max_tokens")
    parser.add_argument("--seed", type=int, default=None, help="随机种子 (默认取 config.system.seed)")
    parser.add_argument("--device", type=str, default=None,
                        help="评测设备；默认使用 config.generation.device")
    parser.add_argument("--output_path", type=str, default="./eval/history/eval_generation/eval_generation.json")
    args = parser.parse_args()

    config = SnailConfig.from_json(args.config)
    tokenizer = get_tokenizer(config)
    seed = args.seed if args.seed is not None else config.system.seed
    device = torch.device(args.device or config.generation.device)
    max_tokens = (
        args.max_tokens
        if args.max_tokens is not None
        else config.generation.max_tokens
    )
    do_sample = not (args.greedy or config.generation.greedy)

    try:
        model_files = resolve_model_files(
            config.generation.model_path,
            model_path=args.model_path,
            models_dir=args.models_dir,
            pattern=args.pattern,
        )
    except ValueError as e:
        parser.error(str(e))
    missing_files = [path for path in model_files if not os.path.isfile(path)]
    if not model_files or missing_files:
        if args.models_dir is not None:
            raise SystemExit(
                f"[error] 在 {args.models_dir} 下未找到匹配 {args.pattern} 的模型"
            )
        missing_path = missing_files[0] if missing_files else config.generation.model_path
        raise SystemExit(f"[error] 模型权重不存在: {missing_path}")

    print(f"Generating answers for {len(model_files)} model(s) x {len(TEST_CASES)} prompts:")
    for mf in model_files:
        print(f"  - {os.path.basename(mf)} (step {extract_step(mf)})")

    results = []
    for mf in tqdm(model_files, desc="Evaluating models", unit="model"):
        model = init_model(config, model_path=mf, device=device)
        model.eval()
        answers = []
        for tc in TEST_CASES:
            # 每道题使用仅由全局 seed 和题号决定的 RNG。不同模型即使在前一道题
            # 提前生成 EOS，也不会让后续题目拿到不同的采样序列。
            seed_generation(seed + tc["id"])
            ans, num_generated_tokens, stopped_on_eos = generate_answer(
                model,
                tokenizer,
                tc["prompt"],
                config,
                device,
                max_tokens,
                do_sample,
                args.prompt_format,
            )
            answers.append({
                "id": tc["id"],
                "category": tc["category"],
                "prompt": tc["prompt"],
                "gold": tc["gold"],
                "generated": ans,
                "reference_hit": reference_hit(tc["gold"], ans),
                "num_generated_tokens": num_generated_tokens,
                "stopped_on_eos": stopped_on_eos,
            })
        scored = [answer for answer in answers if answer["reference_hit"] is not None]
        results.append({
            "step": extract_step(mf),
            "model_path": os.path.normpath(mf),
            "summary": {
                "reference_hits": sum(answer["reference_hit"] for answer in scored),
                "reference_total": len(scored),
                "reference_hit_rate": (
                    sum(answer["reference_hit"] for answer in scored) / len(scored)
                    if scored
                    else None
                ),
                "eos_stop_rate": sum(answer["stopped_on_eos"] for answer in answers)
                / len(answers),
            },
            "answers": answers,
        })
        del model
        torch.cuda.empty_cache()

    output = {
        "eval_type": "generation_on_curated_prompts",
        "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "params": {
            "models_dir": args.models_dir,
            "model_path": args.model_path or config.generation.model_path,
            "pattern": args.pattern,
            "prompt_format": args.prompt_format,
            "max_tokens": max_tokens,
            "seed": seed,
            "do_sample": do_sample,
            "temperature": config.generation.temperature,
            "top_k": config.generation.top_k,
            "top_p": config.generation.top_p,
            "repetition_penalty": config.generation.repetition_penalty,
            "device": str(device),
            "dtype": config.system.dtype,
        },
        "test_cases": TEST_CASES,
        "results": results,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    # 另存一份按模型分组的可读文本
    txt_path = os.path.splitext(args.output_path)[0] + ".txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(f"\n{'=' * 70}\n模型 step={r['step']}  ({os.path.basename(r['model_path'])})\n{'=' * 70}\n")
            f.write(
                f"reference_hit={r['summary']['reference_hits']}/"
                f"{r['summary']['reference_total']}, "
                f"eos_stop_rate={r['summary']['eos_stop_rate']:.2%}\n"
            )
            for a in r["answers"]:
                f.write(f"[{a['category']}] Q: {a['prompt']}\n")
                f.write(f"         gold: {a['gold']}\n")
                f.write(f"         A: {a['generated']}\n")
                f.write(
                    f"         hit={a['reference_hit']}, eos={a['stopped_on_eos']}, "
                    f"tokens={a['num_generated_tokens']}\n"
                )

    print(f"\nResults written to {args.output_path}")
    print(f"Readable text written to {txt_path}")


if __name__ == "__main__":
    main()
