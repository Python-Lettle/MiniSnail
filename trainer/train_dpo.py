import os
import time
import copy
import math
import random
import argparse
import traceback
import wandb
import numpy as np
import torch
import torch.nn.functional as F
from functools import partial
from typing import IO, Any, BinaryIO

from minisnail.model import SnailModel, init_model
from minisnail.config import SnailConfig
from minisnail.debug import console
from minisnail.util import load_config, setup_seed, restore_rng_state
from minisnail.dataset import DPODataset, get_epoch_dataloader
from minisnail.tokenizer import get_tokenizer
from minisnail.functions import apply_optimizer_step

def save_checkpoint(
    policy_model: torch.nn.Module,
    reference_model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    global_step: int,
    epoch: int,
    step: int,
    optimizer_step: int,
    run: wandb.Run,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    """
    checkpoint 语义 (与 train_sft 一致):
    epoch = 断点所在 (正在训练的) epoch, step = 该 epoch 内已完成的 batch 数;
    只在梯度累积边界保存, 恢复时按 (seed, epoch) 确定性重建 batch 顺序并跳过前 step 个 batch。
    """
    # 1. Prepare the file to save the checkpoint
    if isinstance(out, str) or isinstance(out, os.PathLike):
        out = open(out, 'wb')

    # 2. Save the model state to the file
    torch.save(
        {
            'policy_model_state_dict': policy_model.state_dict(),
            'reference_model_state_dict': reference_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'global_step': global_step,
            'epoch': epoch,
            'last_step': step,
            'optimizer_step': optimizer_step,
            'wandb_id': run.id if run is not None else None,
            'rng_state':{
                "torch":
                    torch.get_rng_state(),

                "cuda":
                    torch.cuda.get_rng_state_all(),

                "numpy":
                    np.random.get_state(),

                "python":
                    random.getstate(),
            },
        },
        out
    )
    # 3. Close the file
    out.close()

def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
) -> dict[str, Any]:
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
    # 显式映射到 CPU, 避免续训时 checkpoint 权重+优化器状态占用双份显存直至 OOM
    checkpoint = torch.load(src, map_location="cpu", weights_only=False)

    return checkpoint

def get_sequence_logprob(logits, labels, mask):
    logits = logits[:,:-1,:]
    labels = labels[:,1:]
    mask = mask[:,1:].float()

    log_probs = F.log_softmax(
        logits.float(),
        dim=-1
    )
    token_log_probs = torch.gather(
        log_probs,
        dim=-1,
        index=labels.unsqueeze(-1)
    ).squeeze(-1)

    token_log_probs *= mask

    # 序列 logprob 用求和 (标准 DPO / MiniMind 同款): 偏好信号是整段回复的
    # 对数概率之差, 按长度取平均会把 gap 压小一个数量级, beta=0.1 下信号过弱
    seq_log_probs = token_log_probs.sum(-1)

    return seq_log_probs

def dpo_loss(
    policy_chosen,
    policy_rejected,
    ref_chosen,
    ref_rejected,
    beta
):
    policy_gap = policy_chosen - policy_rejected
    ref_gap = ref_chosen - ref_rejected

    logits = beta * (policy_gap - ref_gap)

    loss = -F.logsigmoid(logits)
    accuracy = (logits > 0).float().mean()

    return loss.mean(), accuracy


def dpo_collate(batch,pad_id):
    chosen_ids=[x["chosen_ids"] for x in batch]
    rejected_ids=[x["rejected_ids"] for x in batch]
    chosen_mask=[x["chosen_mask"] for x in batch]
    rejected_mask=[x["rejected_mask"] for x in batch]

    chosen_ids=torch.nn.utils.rnn.pad_sequence(
        chosen_ids,
        batch_first=True,
        padding_value=pad_id
    )

    rejected_ids=torch.nn.utils.rnn.pad_sequence(
        rejected_ids,
        batch_first=True,
        padding_value=pad_id
    )

    chosen_mask=torch.nn.utils.rnn.pad_sequence(
        chosen_mask,
        batch_first=True,
        padding_value=0
    )
    rejected_mask=torch.nn.utils.rnn.pad_sequence(
        rejected_mask,
        batch_first=True,
        padding_value=0
    )

    return {
        "chosen_ids":chosen_ids,
        "chosen_mask":chosen_mask,
        "rejected_ids":rejected_ids,
        "rejected_mask":rejected_mask
    }

def train_dpo(config: SnailConfig, save_model_dir: str, run: wandb.Run, checkpoint: dict,
              policy_model: SnailModel, reference_model: SnailModel, optimizer: torch.optim.Optimizer,
              dataset, collate_fn):
    os.makedirs(save_model_dir, exist_ok=True)

    # Params
    epochs = config.training.epochs
    accumulation_steps = config.training.accumulation_steps
    device = torch.device(config.system.device)
    # AMP 混合精度（与 train_sft.py 保持一致，6GB 显存下必需）
    _, amp_dtype = config.get_torch_dtype()
    use_amp = config.training.use_amp and amp_dtype is not None

    # fp16 混合精度需要 GradScaler 防止梯度下溢; fp32 / bf16 不需要
    scaler = torch.amp.GradScaler(enabled=(use_amp and amp_dtype == torch.float16))
    if checkpoint is not None and 'scaler_state_dict' in checkpoint:
        scaler.load_state_dict(checkpoint['scaler_state_dict'])

    console.print(f"beta: {config.training.dpo_beta}")
    console.print(f"Mixed precision (AMP): {amp_dtype if use_amp else 'disabled (fp32)'}")

    # Training loop
    steps_per_epoch = math.ceil(len(dataset) / config.training.batch_size)
    total_steps = steps_per_epoch * epochs
    total_optimizer_steps = math.ceil(total_steps / accumulation_steps)
    console.print(f"Total steps: {total_steps}")

    # checkpoint 语义 (与 train_sft 一致): epoch = 断点所在 epoch, last_step = 该 epoch 内已完成的 batch 数
    global_step = checkpoint['global_step'] if checkpoint else 0
    optimizer_step = checkpoint['optimizer_step'] if checkpoint else 0
    epoch = start_epoch = checkpoint['epoch'] if checkpoint else 0
    start_step = checkpoint.get('last_step', 0) if checkpoint else 0
    epoch_step = start_step
    # 学习率: 初始化为 optimizer 当前 lr (含断点恢复值), 保证首个日志点前已定义
    current_lr = optimizer.param_groups[0]["lr"]

    # 梯度累积状态: checkpoint 只在累积边界保存, 恢复时 accumulated 恒为 0
    accumulated_steps = 0
    has_pending_grads = False
    last_save_step = global_step

    def apply_pending_gradients():
        """提交当前累积窗口；残余窗口会按实际 micro-batch 数重新归一化。"""
        nonlocal has_pending_grads, accumulated_steps, optimizer_step, current_lr
        if not has_pending_grads:
            return
        optimizer_step, current_lr = apply_optimizer_step(
            policy_model.parameters(),
            optimizer,
            scaler,
            accumulated_steps=accumulated_steps,
            accumulation_steps=accumulation_steps,
            optimizer_step=optimizer_step,
            max_l2_norm=config.training.gradient_clip,
            max_learning_rate=config.scheduler.max_learning_rate,
            min_learning_rate=config.scheduler.min_learning_rate,
            warmup_iters=config.scheduler.warmup_iters,
            cosine_cycle_iters=config.scheduler.cosine_cycle_iters,
        )
        accumulated_steps = 0
        has_pending_grads = False

    def save_and_exit():
        """训练退出前正确提交残余梯度并保存一致的 checkpoint。"""
        if has_pending_grads:
            pending_count = accumulated_steps
            apply_pending_gradients()
            console.print(
                f"[yellow]Committed {pending_count} pending micro-batch(es) "
                f"as optimizer step {optimizer_step}"
            )
        save_checkpoint(
            policy_model, reference_model, optimizer, scaler,
            global_step, epoch, epoch_step, optimizer_step, run,
            os.path.join(save_model_dir, "dpo_checkpoint.pt"),
        )
        console.print(f"[green]Checkpoint saved at global_step {global_step} / {total_steps}")
        console.print("=" * 50)

    console.print(f"[blue]Training start at epoch {start_epoch}, step {start_step}, global_step {global_step}")
    policy_model.train()
    try:
        for epoch in range(start_epoch, epochs):
            # 每个 epoch 用独立 seed 确定性洗牌, 续训时重建相同 batch 顺序并跳过已完成的 batch
            skip = start_step if epoch == start_epoch else 0
            # 先更新 epoch_step 再构建 loader: 若中断恰好发生在 loader 构造期间,
            # 保存的 last_step 仍是本 epoch 的正确跳过值, 而不是上一轮的残留值
            epoch_step = skip
            loader = get_epoch_dataloader(
                dataset,
                batch_size=config.training.batch_size,
                seed=config.system.seed,
                epoch=epoch,
                skip_batches=skip,
                collate_fn=collate_fn,
            )
            for batch in loader:
                chosen_ids = batch["chosen_ids"].to(device)
                chosen_mask = batch["chosen_mask"].to(device)
                rejected_ids = batch["rejected_ids"].to(device)
                rejected_mask = batch["rejected_mask"].to(device)

                # policy 与 reference 使用一致的 autocast，保证 logits 精度一致
                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                    chosen_logits = policy_model(chosen_ids)
                    rejected_logits = policy_model(rejected_ids)
                    policy_chosen = get_sequence_logprob(chosen_logits, chosen_ids, chosen_mask)
                    policy_rejected = get_sequence_logprob(rejected_logits, rejected_ids, rejected_mask)
                    # reference（冻结，无梯度）
                    with torch.no_grad():
                        ref_chosen_logits = reference_model(chosen_ids)
                        ref_rejected_logits = reference_model(rejected_ids)
                        ref_chosen = get_sequence_logprob(
                            ref_chosen_logits,
                            chosen_ids,
                            chosen_mask
                        )
                        ref_rejected = get_sequence_logprob(
                            ref_rejected_logits,
                            rejected_ids,
                            rejected_mask
                        )

                loss, acc = dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta=config.training.dpo_beta)
                # 梯度累积：缩放 loss 使得梯度平均正确
                scaled_loss = loss / accumulation_steps
                scaler.scale(scaled_loss).backward()
                accumulated_steps += 1
                has_pending_grads = True
                # 只有 forward/backward 均成功后，这个 batch 才能记作已完成。
                global_step += 1
                epoch_step += 1

                if accumulated_steps == accumulation_steps:
                    apply_pending_gradients()

                chosen_reward = config.training.dpo_beta * (policy_chosen - ref_chosen)
                rejected_reward = config.training.dpo_beta * (policy_rejected - ref_rejected)
                reward_margin = chosen_reward - rejected_reward

                # Wandb logging
                if run is not None:
                    run.log({
                        "train/loss": loss.item(),
                        "train/accuracy": acc.item(),
                        "reward/chosen": chosen_reward.mean().item(),
                        "reward/rejected": rejected_reward.mean().item(),
                        "train/margin": reward_margin.mean().item(),
                        "train/lr": current_lr,
                    }, step=global_step)

                # Print information
                if global_step % config.training.print_interval == 0:
                    console.print(f"Epoch [{epoch + 1}/{epochs}]")
                    console.print(f"Step {global_step}/{total_steps}")
                    console.print(f"Loss: {loss.item():.4f}")
                    console.print(f"LR: {current_lr:.2e}")
                    console.print("="*50)

                # 只在梯度累积边界保存 checkpoint: 半截梯度状态无法跨进程恢复,
                # 强行 flush 会把 k/accum 份梯度当完整 step 消费并挤占 cosine 调度槽位
                if not has_pending_grads and global_step - last_save_step >= config.training.print_interval:
                    save_checkpoint(
                        policy_model, reference_model, optimizer, scaler,
                        global_step, epoch, epoch_step, optimizer_step, run,
                        os.path.join(save_model_dir, "dpo_checkpoint.pt"),
                    )
                    last_save_step = global_step
                    console.print(f"[green]Checkpoint saved at epoch {epoch + 1}, global_step {global_step}")

            # Epoch end
            console.print(f"[green]Epoch [{epoch + 1}/{epochs}] completed")

        # 训练正常结束: flush 残余梯度, 保存最终模型和 checkpoint
        console.print(f"[green]Training finished at global_step {global_step} / {total_steps}")
        epoch = epochs  # 标记全部 epoch 已完成, 续训时 range(epochs, epochs) 为空直接结束
        epoch_step = 0
        save_and_exit()

        # Save final model
        final_path = os.path.join(save_model_dir, "dpo_new.pt")
        torch.save(policy_model.state_dict(), final_path)
        console.print(f"Policy Model saved to {final_path}")

    except KeyboardInterrupt:
        # 手动中断: 保存断点以便续训
        console.print(f"[yellow]Interrupted by user at global_step {global_step} / {total_steps}")
        save_and_exit()

    except Exception:
        # 训练异常: 打印完整堆栈并保存断点
        console.print(f"[red]Error at global_step {global_step} / {total_steps}")
        console.print(traceback.format_exc())
        save_and_exit()
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DPO model")
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--data_path", type=str, default="./dataset/dpo.jsonl", help="Path to DPO dataset")
    parser.add_argument("--save_model_dir", type=str, default="./output/new_dpo", help="Path to save model dir")
    args = parser.parse_args()

    # 1. Load configuration
    config = load_config(args)

    # 设置随机种子 (在恢复 checkpoint rng_state 之前)
    setup_seed(config.system.seed)

    # 2. Load checkpoint if specified
    checkpoint = None
    if config.training.use_checkpoint:
        if not config.training.from_checkpoint:
            raise SystemExit("[error] use_checkpoint=True 但 training.from_checkpoint 未设置")
        checkpoint = load_checkpoint(config.training.from_checkpoint)
        console.print(f"Loaded checkpoint from {config.training.from_checkpoint}, "
                      f"global_step: {checkpoint['global_step']}, wandb_id: {checkpoint['wandb_id']}")

    # 3. Load model
    # 参数保持 fp32（AdamW 小 lr 更新需要精度），仅 forward 用 autocast(bf16) 降低激活显存
    # 注意：纯 bf16 权重 + lr=1e-6 时更新会被 bf16 舍入吞掉，loss 将恒为 ln(2)
    device = torch.device(config.system.device)
    policy_model = init_model(config, device=device)
    reference_model = copy.deepcopy(policy_model)

    # Load model parameters and tokenizer
    if checkpoint is not None:
        policy_model.load_state_dict(checkpoint['policy_model_state_dict'])
        reference_model.load_state_dict(checkpoint['reference_model_state_dict'])
    elif config.training.from_weight is not None:
        initial_state = torch.load(
            config.training.from_weight,
            map_location=device,
            weights_only=True,
        )
        policy_model.load_state_dict(initial_state)
        reference_model.load_state_dict(initial_state)

    reference_model.eval()
    for p in reference_model.parameters():
        p.requires_grad=False

    tokenizer = get_tokenizer(config)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 4. Load the datasets
    # batch 顺序由 train_dpo 内的 get_epoch_dataloader 按 (seed, epoch) 确定性生成
    dataset = DPODataset(args.data_path, tokenizer, config.model.context_length)
    # partial 保证 collate 可 pickle (非 Windows 平台 DataLoader 多 worker 需要)
    collate_fn = partial(dpo_collate, pad_id=tokenizer.pad_token_id)

    # 5. Load optimizer
    optimizer = torch.optim.AdamW(
        policy_model.parameters(),
        lr=config.training.lr,
        betas=config.training.betas,
        weight_decay=config.training.weight_decay,
        eps=1e-8
    )
    if checkpoint is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # 6. Start a new wandb run to track this script.
    run = None
    if config.training.use_wandb:
        if checkpoint is not None and checkpoint.get('wandb_id'):
            run = wandb.init(
                # Set the wandb entity where your project will be logged (generally your team name).
                entity=config.wandb.entity,
                # Set the wandb project where this run will be logged.
                project=config.wandb.project,
                # Track hyperparameters and run metadata.
                config=config,
                id=checkpoint['wandb_id'],
                resume="allow",
            )
            console.print(f"Resumed training from checkpoint, id: {run.id}")
        else:
            run = wandb.init(
                # Set the wandb entity where your project will be logged (generally your team name).
                entity=config.wandb.entity,
                # Set the wandb project where this run will be logged.
                project=config.wandb.project,
                # Track hyperparameters and run metadata.
                config=config,
            )
            console.print(f"Started new training, id: {run.id}")

    # 模型、数据、优化器和 W&B 初始化都可能消耗随机数，必须最后恢复。
    if checkpoint is not None and 'rng_state' in checkpoint:
        restore_rng_state(checkpoint['rng_state'])
        console.print("[green]Restored RNG state from checkpoint")

    start_time = time.time()
    train_dpo(config, args.save_model_dir, run, checkpoint, policy_model, reference_model,
              optimizer, dataset, collate_fn)
    end_time = time.time()

    console.print(f"Training time: {end_time - start_time:.2f} seconds")

    if run is not None:
        run.finish()
