import os
import time
import copy
import math
import argparse
import faulthandler
import wandb
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import IO, BinaryIO

# C 层崩溃（如显存耗尽被驱动杀进程）时打印 Python 调用栈，避免静默退出
faulthandler.enable()

from minisnail.model import SnailModel, init_model
from minisnail.config import SnailConfig
from minisnail.debug import console
from minisnail.util import load_config, setup_seed
from minisnail.dataset import DPODataset
from minisnail.tokenizer import get_tokenizer
from minisnail.functions import cosine_schedule, gradient_clipping

def save_checkpoint(
    policy_model: torch.nn.Module,
    reference_model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    global_step: int,
    epoch: int,
    run: wandb.Run,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    # 1. Prepare the file to save the checkpoint
    if isinstance(out, str) or isinstance(out, os.PathLike):
        out = open(out, 'wb')

    # 2. Save the model state to the file
    torch.save(
        {
            'policy_model_state_dict': policy_model.state_dict(),
            'reference_model_state_dict': reference_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step,
            'epoch': epoch,
            'wandb_id': run.id,
        },
        out
    )
    # 3. Close the file
    out.close()

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

def get_log_probs(logits, labels, mask):

    logits = logits[:, :-1, :]
    labels = labels[:,1:]
    mask = mask[:,1:]

    log_probs = F.log_softmax(logits,dim=-1)

    token_log_probs = torch.gather(
        log_probs,
        -1,
        labels.unsqueeze(-1)
    ).squeeze(-1)

    token_log_probs *= mask

    token_count = mask.sum(-1).clamp(min=1)

    seq_log_probs = (
        token_log_probs.sum(-1)
        /
        token_count
    )

    return seq_log_probs

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

def train_dpo(config: SnailConfig, run: wandb.Run, checkpoint: dict, policy_model: SnailModel, reference_model: SnailModel, loader: DataLoader):
    setup_seed(config.system.seed)

    # Params
    epochs = config.training.epochs
    save_model_dir = config.data.save_model_dir
    os.makedirs(save_model_dir, exist_ok=True)
    accumulation_steps = config.training.accumulation_steps
    device = torch.device(config.system.device)
    # AMP 混合精度（与 train_sft.py 保持一致，6GB 显存下必需）
    model_dtype, amp_dtype = config.get_torch_dtype()
    use_amp = amp_dtype is not None
    
    console.print(f"beta: {config.training.dpo_beta}")
    console.print(f"Mixed precision (AMP): {amp_dtype if use_amp else 'disabled (fp32)'}")

    # Load optimizer
    optimizer = torch.optim.AdamW(
        policy_model.parameters(),
        lr=config.training.lr,
        betas=config.training.betas,
        weight_decay=config.training.weight_decay,
        eps=1e-8
    )
    if checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    print("optimizer params:", len(optimizer.param_groups[0]["params"]))

    # Training loop
    steps_per_epoch = len(loader)
    total_steps = len(loader) * epochs
    console.print(f"Total steps: {total_steps}")
    global_step = checkpoint["global_step"] if checkpoint else 0
    optimizer_step = global_step // accumulation_steps
    total_optimizer_steps = math.ceil(total_steps / accumulation_steps)
    current_lr = config.training.lr
    accumulated_micro_steps = 0
    start_epoch = global_step // steps_per_epoch          # 从哪个 epoch 开始
    start_step_in_epoch = global_step % steps_per_epoch   # 该 epoch 内跳过前几步
    console.print("Training start at epoch", start_epoch)
    has_pending_grads = False
    optimizer.zero_grad()
    try:
        for epoch in range(start_epoch, config.training.epochs):
            policy_model.train()
            console.print(f"\n{'='*40}")
            console.print(f"Epoch [{epoch + 1}/{epochs}]")
            console.print(f"{'='*40}")
            for step, batch in enumerate(loader):
                if step < start_step_in_epoch:
                    continue
                global_step += 1

                chosen_ids = batch["chosen_ids"].to(device)
                chosen_mask = batch["chosen_mask"].to(device)
                rejected_ids = batch["rejected_ids"].to(device)
                rejected_mask = batch["rejected_mask"].to(device)

                # policy
                with torch.autocast(device_type="cuda", dtype=model_dtype):
                    chosen_logits = policy_model(chosen_ids)
                    rejected_logits = policy_model(rejected_ids)
                    policy_chosen = get_sequence_logprob(chosen_logits, chosen_ids, chosen_mask)
                    policy_rejected = get_sequence_logprob(rejected_logits, rejected_ids, rejected_mask)
                # reference
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

                loss,acc = dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta=config.training.dpo_beta)
                scaled_loss = loss / accumulation_steps
                scaled_loss.backward()
                has_pending_grads = True

                # Learning rate scheduler (computed every step for logging)
                accumulated_micro_steps += 1

                if accumulated_micro_steps == accumulation_steps:
                    optimizer_step += 1
                    current_lr = cosine_schedule(
                        optimizer_step,
                        max_learning_rate=config.scheduler.max_learning_rate,
                        min_learning_rate=config.scheduler.min_learning_rate,
                        warmup_iters=config.scheduler.warmup_iters,
                        cosine_cycle_iters=total_optimizer_steps,
                    )

                    gradient_clipping(policy_model.parameters(), config.training.gradient_clip)
                    for param_group in optimizer.param_groups:
                        param_group["lr"] = current_lr

                    optimizer.step()
                    optimizer.zero_grad()
                    accumulated_micro_steps = 0
                    has_pending_grads = False

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
                    # Save checkpoint
                    save_checkpoint(policy_model, reference_model, optimizer, global_step, epoch, run, os.path.join(save_model_dir, "dpo_checkpoint.pt"))
                    console.print(f"Checkpoint saved at epoch {epoch+1}, global_step {global_step}")
            start_step_in_epoch = 0
               
    except KeyboardInterrupt:
        console.print("Training interrupted by user.")
        save_checkpoint(policy_model, reference_model, optimizer, global_step, epoch, run, os.path.join(save_model_dir, "dpo_checkpoint.pt"))
        console.print(f"Checkpoint saved at epoch {epoch+1}, global_step {global_step}")
        console.print("="*50)
        return
    except Exception as e:
        console.print(f"Training error: {e}")
        save_checkpoint(policy_model, reference_model, optimizer, global_step, epoch, run, os.path.join(save_model_dir, "dpo_checkpoint.pt"))
        console.print(f"Checkpoint saved at epoch {epoch+1}, global_step {global_step}")
        console.print("="*50)
        raise e
    if has_pending_grads:
        gradient_clipping(policy_model.parameters(), config.training.gradient_clip)
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr
        optimizer.step()
        optimizer.zero_grad()
        console.print("Applied final optimizer step for remaining accumulated gradients")

    # Save final model
    final_path = os.path.join(save_model_dir, "dpo_new.pt")
    os.makedirs(save_model_dir, exist_ok=True)
    torch.save(policy_model.state_dict(), final_path)
    console.print(f"Policy Model saved to {final_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DPO model")
    parser.add_argument("--config", type=str, help="Path to config file")
    args = parser.parse_args()

    # 1. Load configuration
    config = load_config(args)

    # 2. Load checkpoint if specified
    if config.training.use_checkpoint:
        checkpoint = load_checkpoint(config.training.from_checkpoint)
        console.print(f"Loaded checkpoint from {config.training.from_checkpoint}, global_step: {checkpoint['global_step']}, wandb_id: {checkpoint['wandb_id']}")
    else:
        checkpoint = None
    
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
        policy_model.load_state_dict(torch.load(config.training.from_weight))
        reference_model.load_state_dict(torch.load(config.training.from_weight))

    reference_model.eval()
    for p in reference_model.parameters():
        p.requires_grad=False
    
    tokenizer = get_tokenizer(config)
    
    # 4. Load the datasets
    dataset = DPODataset(config.data.dpo_data_path, tokenizer, config.model.context_length)
    loader = DataLoader(
        dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
        collate_fn=lambda x:
            dpo_collate(
                x,
                tokenizer.pad_token_id
            )
    )
    
    # 5. Start a new wandb run to track this script.
    if config.training.use_checkpoint:
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
    
    start_time = time.time()
    train_dpo(config, run, checkpoint, policy_model, reference_model, loader)
    end_time = time.time()

    console.print(f"Training time: {end_time - start_time:.2f} seconds")
    run.finish()