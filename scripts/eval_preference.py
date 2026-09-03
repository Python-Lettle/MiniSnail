"""Evaluate a DPO policy against its frozen SFT reference on held-out pairs."""

import argparse
import datetime
import json
import os
from functools import partial

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from minisnail.config import SnailConfig
from minisnail.dataset import DPODataset
from minisnail.model import init_model
from minisnail.preference import dpo_collate, get_sequence_logprob
from minisnail.tokenizer import get_tokenizer


def summarize_preference(
    policy_chosen: np.ndarray,
    policy_rejected: np.ndarray,
    reference_chosen: np.ndarray,
    reference_rejected: np.ndarray,
    chosen_lengths: np.ndarray,
    rejected_lengths: np.ndarray,
    beta: float,
) -> dict:
    policy_gap = policy_chosen - policy_rejected
    reference_gap = reference_chosen - reference_rejected
    implicit_margin = beta * (policy_gap - reference_gap)
    normalized_gap = (
        policy_chosen / chosen_lengths - policy_rejected / rejected_lengths
    )
    losses = np.logaddexp(0.0, -implicit_margin)
    return {
        "num_samples": int(len(policy_gap)),
        "dpo_loss": float(losses.mean()),
        "dpo_accuracy": float((implicit_margin > 0).mean()),
        "mean_implicit_reward_margin": float(implicit_margin.mean()),
        "policy_chosen_win_rate": float((policy_gap > 0).mean()),
        "reference_chosen_win_rate": float((reference_gap > 0).mean()),
        "mean_policy_logprob_gap": float(policy_gap.mean()),
        "mean_reference_logprob_gap": float(reference_gap.mean()),
        "mean_length_normalized_policy_gap": float(normalized_gap.mean()),
    }


def main():
    parser = argparse.ArgumentParser(
        description="在独立 DPO 验证集上评测 policy 相对 SFT reference 的偏好提升"
    )
    parser.add_argument("--config", default="./config.json")
    parser.add_argument("--data_path", required=True, help="未参与训练的 DPO JSONL")
    parser.add_argument("--model_path", required=True, help="待评测 DPO policy 权重")
    parser.add_argument("--reference_model_path", required=True, help="DPO 初始化时的 SFT 权重")
    parser.add_argument("--num_samples", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--output_path",
        default="./eval/history/eval_preference/eval_preference.json",
    )
    args = parser.parse_args()

    for path in (args.data_path, args.model_path, args.reference_model_path):
        if not os.path.isfile(path):
            raise SystemExit(f"[error] 文件不存在: {path}")
    if args.num_samples <= 0 or args.batch_size <= 0:
        parser.error("num_samples 与 batch_size 必须大于 0")

    config = SnailConfig.from_json(args.config)
    tokenizer = get_tokenizer(config)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    seed = config.system.seed if args.seed is None else args.seed
    device = torch.device(args.device or config.system.device)

    dataset = DPODataset(args.data_path, tokenizer, config.model.context_length)
    if len(dataset) == 0:
        raise SystemExit("[error] DPO 验证集为空")
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(dataset))[: min(args.num_samples, len(dataset))]
    dataloader = DataLoader(
        Subset(dataset, indices.tolist()),
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        collate_fn=partial(dpo_collate, pad_id=tokenizer.pad_token_id),
    )

    policy = init_model(config, model_path=args.model_path, device=device).eval()
    reference = init_model(
        config, model_path=args.reference_model_path, device=device
    ).eval()
    _, amp_dtype = config.get_torch_dtype()
    use_amp = config.training.use_amp and amp_dtype is not None

    collected = {key: [] for key in (
        "policy_chosen", "policy_rejected", "reference_chosen",
        "reference_rejected", "chosen_lengths", "rejected_lengths",
    )}
    with torch.inference_mode():
        for batch in tqdm(dataloader, desc="Evaluating preference pairs"):
            batch = {key: value.to(device) for key, value in batch.items()}
            with torch.autocast(
                device_type=device.type, dtype=amp_dtype, enabled=use_amp
            ):
                policy_chosen = get_sequence_logprob(
                    policy(batch["chosen_ids"]), batch["chosen_ids"], batch["chosen_mask"]
                )
                policy_rejected = get_sequence_logprob(
                    policy(batch["rejected_ids"]), batch["rejected_ids"], batch["rejected_mask"]
                )
                reference_chosen = get_sequence_logprob(
                    reference(batch["chosen_ids"]), batch["chosen_ids"], batch["chosen_mask"]
                )
                reference_rejected = get_sequence_logprob(
                    reference(batch["rejected_ids"]), batch["rejected_ids"], batch["rejected_mask"]
                )
            values = {
                "policy_chosen": policy_chosen,
                "policy_rejected": policy_rejected,
                "reference_chosen": reference_chosen,
                "reference_rejected": reference_rejected,
                "chosen_lengths": batch["chosen_mask"][:, 1:].sum(-1).clamp(min=1),
                "rejected_lengths": batch["rejected_mask"][:, 1:].sum(-1).clamp(min=1),
            }
            for key, value in values.items():
                collected[key].extend(value.float().cpu().tolist())

    arrays = {key: np.asarray(value, dtype=np.float64) for key, value in collected.items()}
    metrics = summarize_preference(beta=config.training.dpo_beta, **arrays)
    output = {
        "eval_type": "held_out_dpo_preference",
        "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "params": {
            "data_path": os.path.normpath(args.data_path),
            "model_path": os.path.normpath(args.model_path),
            "reference_model_path": os.path.normpath(args.reference_model_path),
            "seed": seed,
            "device": str(device),
            "batch_size": args.batch_size,
            "beta": config.training.dpo_beta,
        },
        "metrics": metrics,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as file:
        json.dump(output, file, indent=2, ensure_ascii=False)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"Results written to {args.output_path}")


if __name__ == "__main__":
    main()
