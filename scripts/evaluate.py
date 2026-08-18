import os
import time
import wandb
import random
import argparse
import numpy as np
import torch
import json
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import DataLoader, Subset
import torch.nn.functional as F
from typing import IO, BinaryIO
from transformers import PreTrainedTokenizer

from minisnail.util import setup_seed
from minisnail.tokenizer import get_tokenizer
from minisnail.config import SnailConfig, DEFAULT_CONFIG
from minisnail.model import init_model
from minisnail.debug import console

# 评测时使用的生成参数（会影响模型输出，统一在此配置并随结果保存）
GEN_PARAMS = {
    "max_tokens": 512,
    "temperature": 0.85,
    "top_k": 50,
    "top_p": 0.9,
    "repetition_penalty": 1.2,
    "do_sample": True,
}
# 采样种子：do_sample=True 时影响输出，记录以保证可复现
SEED = 42

def load_jsonl(path):
    data = []
    with open(path,"r",encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data

def score_response(response, keywords):
    """计算单条得分：返回 (命中数, 关键词总数, 命中的关键词列表)"""
    if not keywords:
        return 0, 0, []
    hits = [kw for kw in keywords if kw in response]
    return len(hits), len(keywords), hits

def summarize_scores(results):
    """按类别统计正确率（x/y 形式）、归一化得分（1分满分），
    并汇总总分数和（每题满分1分）"""
    per_category = {}
    for r in results:
        if r.get("score") is None:
            continue
        stat = per_category.setdefault(r["category"], {"hit": 0, "total": 0, "score_sum": 0.0, "num_items": 0})
        stat["hit"] += r["hit"]
        stat["total"] += r["n_keywords"]
        stat["score_sum"] += r["score"]
        stat["num_items"] += 1

    category_view = {
        cat: {
            "accuracy": f"{s['hit']}/{s['total']}",
            "score": round(s["hit"] / s["total"], 4) if s["total"] else None,
            "score_sum": round(s["score_sum"], 2),
            "max_score": s["num_items"],
        }
        for cat, s in per_category.items()
    }

    total_hit = sum(s["hit"] for s in per_category.values())
    total_kw = sum(s["total"] for s in per_category.values())
    score_sum = sum(s["score_sum"] for s in per_category.values())
    overall = {
        "accuracy": f"{total_hit}/{total_kw}",
        "score": round(total_hit / total_kw, 4) if total_kw else None,
        "score_sum": round(score_sum, 2),
        "max_score": len(results),
    }
    return {"per_category": category_view, "overall": overall}

def evaluate(model, tokenizer, dataset):
    setup_seed(SEED)

    results = []
    for idx, item in enumerate(tqdm(dataset, desc="Evaluating", unit="item")):
        prompt = item["prompt"]
        response = model.chat(prompt, tokenizer, **GEN_PARAMS)

        hit, n_kw, hits = score_response(response, item.get("keywords"))
        score = hit / n_kw if n_kw else None
        result = {
            "id":item.get("id", idx),
            "category":item["category"],
            "prompt":prompt,
            "response":response,
            "score_display":f"{hit}/{n_kw}",
            "keywords_hit":hits,
            "hit":hit,
            "n_keywords":n_kw,
            "score":score,
            "gen_params":{**GEN_PARAMS, "seed":SEED},
        }
        results.append(result)

    summary = summarize_scores(results)
    return results, summary

def save_results(results,path):
    with open(path, "w", encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False)+"\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniSnail SFT")
    parser.add_argument("--config", type=str, help="Path to config JSON file")
    args = parser.parse_args()

    # 1. Load configuration
    if args.config:
        config = SnailConfig.from_json(args.config)
        console.print(f"Loaded config from {args.config}")
    elif os.path.exists("config.json"):
        config = SnailConfig.from_json("config.json")
        console.print("Loaded config from default config.json")
    else:
        config = DEFAULT_CONFIG
        console.print("Loaded default config")

    tokenizer: PreTrainedTokenizer = get_tokenizer(config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dtype = torch.bfloat16
    model = init_model(config, device=device, dtype=model_dtype)

    # Load the model weight
    model_dir = "./model/local_dpo_model_epo2/dpo_new.pt"
    model.load_state_dict(torch.load(model_dir, weights_only=False))

    # Load the dataset
    dataset = load_jsonl("eval/base.jsonl")

    results, summary = evaluate(model, tokenizer, dataset)

    # 将本次评测的生成参数写入汇总
    summary["generation_params"] = {**GEN_PARAMS, "seed":SEED}

    save_results(results, "eval_result.jsonl")
    with open("eval_score.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # 终端只输出总结性结果
    print("="*50)
    print("Per-category scores:")
    for cat, s in summary["per_category"].items():
        print(f"  [{cat}] {s['accuracy']} -> {s['score']}  score_sum: {s['score_sum']}/{s['max_score']}")
    o = summary["overall"]
    print("Overall:")
    print(f"  accuracy: {o['accuracy']} -> {o['score']}")
    print(f"  score_sum: {o['score_sum']}/{o['max_score']}")
    print("="*50)