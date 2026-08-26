import os
import re
import json
import glob
import math
import argparse
import datetime

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from minisnail.config import SnailConfig
from minisnail.dataset import LazyPretrainDataset, load_line_offsets
from minisnail.tokenizer import get_tokenizer
from minisnail.model import init_model


def extract_step(model_path: str) -> int:
    """从权重文件名 pretrain_lm_68000.pt 中提取训练步数, 失败返回 0"""
    m = re.search(r"(\d+)\.pt$", os.path.basename(model_path))
    return int(m.group(1)) if m else 0


def build_val_indices(data_path: str, train_ratio: float, seed: int) -> np.ndarray:
    """与 train_pretrain.py 完全一致的 train/val 划分, 取验证集行号。

    使用固定 seed 做 permutation 后按 train_ratio 切分, 保证与训练时验证集一致。
    """
    num_lines = len(load_line_offsets(data_path))
    split_index = int(num_lines * train_ratio)
    perm = np.random.default_rng(seed).permutation(num_lines)
    return perm[split_index:]


def evaluate_model(model, dataloader, vocab_size, device) -> np.ndarray:
    """在验证集上计算每个样本 (每个文档) 的平均交叉熵 loss, 返回 shape=(N,) 数组。

    与训练一致地做 shift (预测 token[t+1]), pad 位置 (label -100) 不计入。
    """
    model.eval()
    sample_losses: list[float] = []
    with torch.no_grad():
        for input_ids, labels in dataloader:
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            logits = model(input_ids)
            logits = logits[:, :-1].contiguous()  # (B, S-1, V)
            targets = labels[:, 1:].contiguous()  # (B, S-1)
            loss = F.cross_entropy(
                logits.view(-1, vocab_size),
                targets.view(-1),
                reduction="none",
                ignore_index=-100,
            ).view(targets.shape)  # (B, S-1)
            mask = (targets != -100).float()
            valid_count = mask.sum(dim=1).clamp(min=1)
            per_sample = (loss * mask).sum(dim=1) / valid_count
            sample_losses.extend(per_sample.cpu().tolist())
    return np.asarray(sample_losses, dtype=np.float64)


def summarize(losses: np.ndarray) -> dict:
    """聚合单样本 loss 数组, 输出均值 / 样本标准差 / 均值标准误 / 95% 置信区间 / perplexity"""
    n = len(losses)
    mean = float(losses.mean())
    std = float(losses.std(ddof=1)) if n > 1 else 0.0
    sem = std / math.sqrt(n)
    ci95 = 1.96 * sem
    return {
        "n": n,
        "avg_loss": round(mean, 4),
        "std_loss": round(std, 4),
        "sem": round(sem, 4),
        "ci95": round(ci95, 4),
        "loss_ci95": [round(mean - ci95, 4), round(mean + ci95, 4)],
        "perplexity": round(math.exp(mean), 2),
    }


def main():
    parser = argparse.ArgumentParser(description="预训练模型 perplexity 评测 (验证集上对比多个 checkpoint, 含置信区间)")
    parser.add_argument("--config", type=str, default="./config.json", help="配置文件路径")
    parser.add_argument("--data_path", type=str, default="./dataset/full/pretrain_t2t.jsonl", help="预训练 jsonl 数据路径")
    parser.add_argument("--models_dir", type=str, default="./model/new_pretrain", help="模型目录 (遍历 <dir>/<pattern>)")
    parser.add_argument("--pattern", type=str, default="pretrain_lm_*.pt", help="模型权重文件名匹配模式")
    parser.add_argument("--train_ratio", type=float, default=0.95, help="训练集占比 (与训练脚本一致)")
    parser.add_argument("--num_samples", type=int, default=2000, help="评测使用的验证样本数 (越多置信区间越窄)")
    parser.add_argument("--batch_size", type=int, default=8, help="评测 batch size")
    parser.add_argument("--seed", type=int, default=None, help="随机种子 (默认取 config.system.seed)")
    parser.add_argument("--output_path", type=str, default="./eval/history/eval_ppl/eval_ppl_score.json", help="结果 JSON 输出路径")
    args = parser.parse_args()

    config = SnailConfig.from_json(args.config)
    tokenizer = get_tokenizer(config)
    seed = args.seed if args.seed is not None else config.system.seed
    device = torch.device(config.system.device)

    # 1. 取验证集前 num_samples 个样本 (行号与训练时验证集一致, 可复现)
    val_indices = build_val_indices(args.data_path, args.train_ratio, seed)[: args.num_samples]
    val_dataset = LazyPretrainDataset(
        data_path=args.data_path,
        tokenizer=tokenizer,
        max_length=config.model.context_length,
        indices=val_indices,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )

    # 2. 收集待评测模型, 按步数升序
    model_files = sorted(
        glob.glob(os.path.join(args.models_dir, args.pattern)),
        key=extract_step,
    )
    if not model_files:
        raise SystemExit(f"[error] 在 {args.models_dir} 下未找到匹配 {args.pattern} 的模型")

    print(f"Evaluating {len(model_files)} model(s) on {len(val_dataset)} validation samples:")
    for mf in model_files:
        print(f"  - {os.path.basename(mf)} (step {extract_step(mf)})")

    # 3. 逐个模型评测, 保存每样本 loss (所有模型用同一批样本且顺序一致, 便于配对比较)
    per_model_losses: dict[int, np.ndarray] = {}
    for mf in tqdm(model_files, desc="Evaluating models", unit="model"):
        model = init_model(config, model_path=mf)
        model.to(device)
        losses = evaluate_model(model, val_dataloader, config.model.vocab_size, device)
        per_model_losses[extract_step(mf)] = losses
        del model
        torch.cuda.empty_cache()

    # 4. 汇总每个模型
    results = []
    for mf in model_files:
        step = extract_step(mf)
        summary = summarize(per_model_losses[step])
        results.append({"model_path": os.path.normpath(mf), "step": step, **summary})

    results_sorted = sorted(results, key=lambda r: r["avg_loss"])
    best = results_sorted[0]

    print("\n" + "=" * 78)
    print(f"{'step':>8}  {'avg_loss':>9}  {'mean±SEM':>14}  {'95% CI':>22}  {'ppl':>7}")
    print("=" * 78)
    for r in sorted(results, key=lambda r: r["step"]):
        lo, hi = r["loss_ci95"]
        print(f"{r['step']:>8}  {r['avg_loss']:>9.4f}  "
              f"{r['avg_loss']:>7.4f}±{r['sem']:.4f}  "
              f"[{lo:.4f}, {hi:.4f}]  {r['perplexity']:>7.2f}")
    print("=" * 78)

    # 5. 与最佳模型做配对比较 (同一批样本, 配对差异的显著性)
    best_losses = per_model_losses[best["step"]]
    print(f"\n[best] step={best['step']}  perplexity={best['perplexity']}")
    print(f"配对比较 (其余模型 vs {best['step']}, 差值=该模型loss-best loss, 正值表示更差):")
    paired_comparisons = []
    for r in sorted(results, key=lambda r: r["step"]):
        if r["step"] == best["step"]:
            continue
        diffs = per_model_losses[r["step"]] - best_losses
        n = len(diffs)
        mean_diff = float(diffs.mean())
        std_diff = float(diffs.std(ddof=1)) if n > 1 else 0.0
        sem_diff = std_diff / math.sqrt(n)
        z = mean_diff / sem_diff if sem_diff > 0 else 0.0
        significant = z > 1.96  # 近似双尾 95% 显著性 (大样本下 t≈z)
        paired_comparisons.append({
            "step": r["step"],
            "mean_loss_diff": round(mean_diff, 4),
            "sem_diff": round(sem_diff, 4),
            "z": round(z, 2),
            "significant_better_than_this": bool(significant),
        })
        flag = " [显著更差]" if significant else " [差异不显著]"
        print(f"  step={r['step']:>7}:  avg_loss_diff = {mean_diff:+.4f} ± {sem_diff:.4f} "
              f"(z={z:.2f}){flag}")

    # 6. 保存结果
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    output = {
        "eval_type": "pretrain_perplexity",
        "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "params": {
            "data_path": args.data_path,
            "models_dir": args.models_dir,
            "pattern": args.pattern,
            "train_ratio": args.train_ratio,
            "num_samples": len(val_dataset),
            "batch_size": args.batch_size,
            "seed": seed,
            "device": str(device),
            "dtype": config.system.dtype,
        },
        "results": results_sorted,
        "best": best,
        "paired_vs_best": paired_comparisons,
    }
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nResults written to {args.output_path}")


if __name__ == "__main__":
    main()