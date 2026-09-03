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
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio 必须在 0 和 1 之间")
    num_lines = len(load_line_offsets(data_path))
    split_index = int(num_lines * train_ratio)
    perm = np.random.default_rng(seed).permutation(num_lines)
    return perm[split_index:]


def evaluate_model(model, dataloader, vocab_size, device) -> tuple[np.ndarray, np.ndarray]:
    """返回每个文档的 NLL 总和与有效 token 数。

    与训练一致地做 shift (预测 token[t+1]), pad 位置 (label -100) 不计入。
    汇总阶段用 ``sum(NLL) / sum(tokens)`` 计算标准语料级 perplexity，同时
    保留逐文档 loss，供置信区间和配对比较使用。
    """
    model.eval()
    sample_nlls: list[float] = []
    sample_token_counts: list[int] = []
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
            mask = targets != -100
            nll_sum = (loss * mask).sum(dim=1)
            valid_count = mask.sum(dim=1)
            valid_samples = valid_count > 0
            sample_nlls.extend(nll_sum[valid_samples].cpu().tolist())
            sample_token_counts.extend(valid_count[valid_samples].cpu().tolist())
    return (
        np.asarray(sample_nlls, dtype=np.float64),
        np.asarray(sample_token_counts, dtype=np.int64),
    )


def summarize(nll_sums: np.ndarray, token_counts: np.ndarray) -> dict:
    """计算语料级 loss/PPL，并保留逐文档统计量用于不确定性分析。"""
    if len(nll_sums) == 0 or len(nll_sums) != len(token_counts):
        raise ValueError("评测结果为空，或 NLL 与 token 计数长度不一致")
    total_tokens = int(token_counts.sum())
    if total_tokens <= 0:
        raise ValueError("评测集中没有可用于计算 perplexity 的有效 token")

    losses = nll_sums / token_counts
    n = len(losses)
    corpus_mean = float(nll_sums.sum() / total_tokens)
    document_mean = float(losses.mean())
    std = float(losses.std(ddof=1)) if n > 1 else 0.0
    sem = std / math.sqrt(n)
    ci95 = 1.96 * sem
    return {
        "n": n,
        "num_tokens": total_tokens,
        "avg_loss": corpus_mean,
        "document_avg_loss": document_mean,
        "std_loss": std,
        "sem": sem,
        "ci95": ci95,
        "loss_ci95": [document_mean - ci95, document_mean + ci95],
        "perplexity": math.exp(corpus_mean),
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
    parser.add_argument("--device", type=str, default=None, help="评测设备；默认使用 config.system.device")
    parser.add_argument("--output_path", type=str, default="./eval/history/eval_ppl/eval_ppl_score.json", help="结果 JSON 输出路径")
    args = parser.parse_args()

    config = SnailConfig.from_json(args.config)
    tokenizer = get_tokenizer(config)
    seed = args.seed if args.seed is not None else config.system.seed
    device = torch.device(args.device or config.system.device)

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
    # Use paths as keys: model_best.pt and model_final.pt both have step=0 and
    # previously overwrote each other silently.
    per_model_metrics: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for mf in tqdm(model_files, desc="Evaluating models", unit="model"):
        model = init_model(config, model_path=mf, device=device)
        metrics = evaluate_model(model, val_dataloader, config.model.vocab_size, device)
        per_model_metrics[os.path.abspath(mf)] = metrics
        del model
        torch.cuda.empty_cache()

    # 4. 汇总每个模型
    results = []
    for mf in model_files:
        step = extract_step(mf)
        summary = summarize(*per_model_metrics[os.path.abspath(mf)])
        results.append({"model_path": os.path.normpath(mf), "step": step, **summary})

    results_sorted = sorted(results, key=lambda r: r["avg_loss"])
    best = results_sorted[0]

    print("\n" + "=" * 78)
    print(f"{'step':>8}  {'corpus_loss':>11}  {'doc mean±SEM':>16}  {'doc 95% CI':>22}  {'ppl':>7}")
    print("=" * 78)
    for r in sorted(results, key=lambda r: r["step"]):
        lo, hi = r["loss_ci95"]
        print(f"{r['step']:>8}  {r['avg_loss']:>11.4f}  "
              f"{r['document_avg_loss']:>9.4f}±{r['sem']:.4f}  "
              f"[{lo:.4f}, {hi:.4f}]  {r['perplexity']:>7.2f}")
    print("=" * 78)

    # 5. 与最佳模型做配对比较 (同一批样本, 配对差异的显著性)
    best_key = os.path.abspath(best["model_path"])
    best_nlls, best_token_counts = per_model_metrics[best_key]
    best_losses = best_nlls / best_token_counts
    print(f"\n[best] step={best['step']}  perplexity={best['perplexity']}")
    print(f"配对比较 (其余模型 vs {best['step']}, 差值=该模型loss-best loss, 正值表示更差):")
    paired_comparisons = []
    for r in sorted(results, key=lambda r: r["step"]):
        if os.path.normpath(r["model_path"]) == os.path.normpath(best["model_path"]):
            continue
        model_nlls, model_token_counts = per_model_metrics[
            os.path.abspath(r["model_path"])
        ]
        diffs = model_nlls / model_token_counts - best_losses
        n = len(diffs)
        mean_diff = float(diffs.mean())
        std_diff = float(diffs.std(ddof=1)) if n > 1 else 0.0
        sem_diff = std_diff / math.sqrt(n)
        z = mean_diff / sem_diff if sem_diff > 0 else 0.0
        significant = abs(z) > 1.96  # 近似双尾 95% 显著性 (大样本下 t≈z)
        paired_comparisons.append({
            "step": r["step"],
            "mean_loss_diff": round(mean_diff, 4),
            "sem_diff": round(sem_diff, 4),
            "z": round(z, 2),
            "significant_difference": bool(significant),
            "direction": "worse" if mean_diff > 0 else "better",
        })
        flag = (
            f" [显著{'更差' if mean_diff > 0 else '更好'}]"
            if significant
            else " [差异不显著]"
        )
        print(f"  step={r['step']:>7}:  avg_loss_diff = {mean_diff:+.4f} ± {sem_diff:.4f} "
              f"(z={z:.2f}){flag}")

    # 6. 保存结果
    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
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
