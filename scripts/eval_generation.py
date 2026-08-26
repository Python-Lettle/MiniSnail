import os
import re
import json
import glob
import argparse
import datetime

import torch
from tqdm import tqdm

from minisnail.config import SnailConfig
from minisnail.tokenizer import get_tokenizer
from minisnail.model import init_model


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


def generate_answer(model, tokenizer, prompt, config, device, max_tokens, do_sample):
    # 预训练时每个序列都以 bos_token (<|im_start|>) 开头; 生成时也必须加, 否则输出退化为空/循环
    input_ids = [tokenizer.bos_token_id] + tokenizer(prompt)["input_ids"]
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
    return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()


def main():
    parser = argparse.ArgumentParser(description="自建测试集上的生成式评测 (按提示词生成并保存答案, 供人工判读)")
    parser.add_argument("--config", type=str, default="./config.json")
    parser.add_argument("--models_dir", type=str, default="./model/new_pretrain")
    parser.add_argument("--pattern", type=str, default="pretrain_lm_*.pt")
    parser.add_argument("--max_tokens", type=int, default=80)
    parser.add_argument("--seed", type=int, default=None, help="随机种子 (默认取 config.system.seed)")
    parser.add_argument("--output_path", type=str, default="./eval/history/eval_generation/eval_generation.json")
    args = parser.parse_args()

    config = SnailConfig.from_json(args.config)
    tokenizer = get_tokenizer(config)
    seed = args.seed if args.seed is not None else config.system.seed
    device = torch.device(config.system.device)
    do_sample = not config.generation.greedy

    model_files = sorted(glob.glob(os.path.join(args.models_dir, args.pattern)), key=extract_step)
    if not model_files:
        raise SystemExit(f"[error] 在 {args.models_dir} 下未找到匹配 {args.pattern} 的模型")

    print(f"Generating answers for {len(model_files)} model(s) x {len(TEST_CASES)} prompts:")
    for mf in model_files:
        print(f"  - {os.path.basename(mf)} (step {extract_step(mf)})")

    torch.manual_seed(seed)
    results = []
    for mf in tqdm(model_files, desc="Evaluating models", unit="model"):
        model = init_model(config, model_path=mf)
        model.eval()
        model.to(device)
        answers = []
        for tc in TEST_CASES:
            ans = generate_answer(model, tokenizer, tc["prompt"], config, device, args.max_tokens, do_sample)
            answers.append({
                "id": tc["id"],
                "category": tc["category"],
                "prompt": tc["prompt"],
                "gold": tc["gold"],
                "generated": ans,
            })
        results.append({
            "step": extract_step(mf),
            "model_path": os.path.normpath(mf),
            "answers": answers,
        })
        del model
        torch.cuda.empty_cache()

    output = {
        "eval_type": "generation_on_curated_prompts",
        "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "params": {
            "models_dir": args.models_dir,
            "pattern": args.pattern,
            "max_tokens": args.max_tokens,
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

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    # 另存一份按模型分组的可读文本
    txt_path = os.path.splitext(args.output_path)[0] + ".txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(f"\n{'=' * 70}\n模型 step={r['step']}  ({os.path.basename(r['model_path'])})\n{'=' * 70}\n")
            for a in r["answers"]:
                f.write(f"[{a['category']}] Q: {a['prompt']}\n")
                f.write(f"         gold: {a['gold']}\n")
                f.write(f"         A: {a['generated']}\n")

    print(f"\nResults written to {args.output_path}")
    print(f"Readable text written to {txt_path}")


if __name__ == "__main__":
    main()