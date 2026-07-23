import os
import time
import wandb
import random
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
import torch.nn.functional as F
from typing import IO, BinaryIO
from transformers import AutoTokenizer

import multiprocessing as mp

from minisnail.debug import console, LossMonitor
from minisnail.functions import cosine_schedule, gradient_clipping
from minisnail.util import setup_seed
from minisnail.config import SnailConfig, DEFAULT_CONFIG
from minisnail.dataset import SFTDataset
from minisnail.model import init_model

def save_checkpoint(
	model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    global_step: int,
    epoch: int,
    run: wandb.Run,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    """
    Given a model, optimizer, and an iteration number, serialize them to disk.

    Args:
        model (torch.nn.Module): Serialize the state of this model.
        optimizer (torch.optim.Optimizer): Serialize the state of this optimizer.
        iteration (int): Serialize this value, which represents the number of training iterations
            we've completed.
        out (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialize the model, optimizer, and iteration to.
    """
    # 1. Prepare the file to save the checkpoint
    if isinstance(out, str) or isinstance(out, os.PathLike):
        out = open(out, 'wb')

    # 2. Save the model state to the file
    torch.save(
        {
            'model_state_dict': model.state_dict(),
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
    checkpoint = torch.load(src)
    
    return checkpoint


def train_sft(config: SnailConfig, run: wandb.Run, checkpoint: dict | None = None):
    setup_seed(config.system.seed)
    # 1. Prepare training parameters
    input_ids_path = config.data.input_ids_path
    labels_path = config.data.labels_path
    save_model_dir = config.data.save_model_dir
    os.makedirs(save_model_dir, exist_ok=True)

    device = torch.device(config.system.device)
    epochs: int = config.training.epochs
    lr: float = config.training.lr
    betas: tuple[float, float] = config.training.betas
    weight_decay: float = config.training.weight_decay
    
    console.print("Input ids path:", input_ids_path)
    console.print("Labels path:", labels_path)
    console.print("Save model dir:", save_model_dir)
    console.print("Epochs:", epochs)
    console.print("Device:", device)
    console.print("lr:", lr)
    console.print("betas:", betas)
    console.print("weight_decay:", weight_decay)

    vocab_size: int = config.model.vocab_size
    context_length: int = config.model.context_length
    d_model: int = config.model.d_model
    num_layers: int = config.model.num_layers
    num_heads: int = config.model.num_heads
    d_ff: int = config.model.d_ff
    rope_theta: float = config.model.rope_theta
    batch_size: int = config.training.batch_size

    console.print("vocab_size:", vocab_size)
    console.print("context_length:", context_length)
    console.print("d_model:", d_model)
    console.print("num_layers:", num_layers)
    console.print("num_heads:", num_heads)
    console.print("d_ff:", d_ff)
    console.print("rope_theta:", rope_theta)
    console.print("batch_size:", batch_size)

    # 2. Load datasets
    train_data = SFTDataset(input_ids_path, labels_path)
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
    )
    total_steps = len(train_loader) * epochs
    console.print(f"数据集大小: {len(train_data)} 条, "
               f"每 epoch {len(train_loader)} 步, "
               f"共 {epochs} epoch = {total_steps} 步")

    # 3. Create the model and optimizer
    model = init_model(config, device=device)
    if config.training.from_weight:
        model.load_state_dict(torch.load(config.training.from_weight))
        console.print("[yellow]Loading model from weight:", config.training.from_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=betas, eps=1e-8, weight_decay=weight_decay)

    # 4. Load checkpoint
    start_epoch = 0
    if checkpoint is not None:
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        console.print("Load checkpoint from:", start_epoch)

    train_loss_monitor = LossMonitor(title="Train Loss Monitor", show_stats=False)

    # 5. Train the model
    console.print("Training start at epoch", start_epoch)
    global_step = checkpoint['global_step'] if checkpoint is not None else 0
    min_loss = float('inf')

    steps_per_epoch = len(train_loader)
    start_epoch = global_step // steps_per_epoch          # 从哪个 epoch 开始
    start_step_in_epoch = global_step % steps_per_epoch   # 该 epoch 内跳过前几步
    try:
        for epoch in range(start_epoch, epochs):
            console.print(f"\n{'='*40}")
            console.print(f"Epoch [{epoch + 1}/{epochs}]")
            console.print(f"{'='*40}")

            epoch_loss = 0.0
            epoch_steps = 0
            epoch_start = time.time()

            for step, (input_ids, labels) in enumerate(train_loader):
                if step < start_step_in_epoch:
                    continue
                global_step += 1
                epoch_steps += 1
                
                input_ids = input_ids.to(device)
                labels = labels.to(device)

                # console.print(f"Step {global_step}/{total_steps}")
                logits = model(input_ids)
                loss = F.cross_entropy(
                    logits[:, :-1, :].contiguous().view(-1, vocab_size),
                    labels[:, 1:].contiguous().view(-1),
                    ignore_index=-100
                )
                epoch_loss += loss.item()
                # console.print(f"Loss: {loss.item():.4f}")

                optimizer.zero_grad()
                loss.backward()

                gradient_clipping(model.parameters(), config.training.gradient_clip)

                # Learning rate scheduler
                current_lr = cosine_schedule(
                    global_step,
                    max_learning_rate=config.scheduler.max_learning_rate,
                    min_learning_rate=config.scheduler.min_learning_rate,
                    warmup_iters=config.scheduler.warmup_iters,
                    cosine_cycle_iters=config.scheduler.cosine_cycle_iters,
                )
                for param_group in optimizer.param_groups:
                    param_group["lr"] = current_lr
                
                optimizer.step()
                train_loss_monitor.add_loss(global_step, loss.item())

                # Wandb logging
                if run is not None:
                    run.log({
                        "train/loss": loss.item(),
                        "train/lr": current_lr,
                    }, step=global_step)

                # Print information
                if global_step % config.training.print_interval == 0:
                    console.print(f"Epoch [{epoch + 1}/{epochs}]")
                    console.print(f"Step {global_step}/{total_steps}")
                    console.print(f"Loss: {loss.item():.4f}")
                    console.print(f"{config.training.print_interval} Steps completed")
                    console.print("="*50)

                    # Save checkpoint
                    if loss.item() < min_loss:
                        min_loss = loss.item()
                        torch.save(model.state_dict(), os.path.join(save_model_dir, "sft_best.pt"))

            # Epoch end
            avg_epoch_loss = epoch_loss / max(epoch_steps, 1)
            epoch_time = (time.time() - epoch_start) / 60
            console.print(
                f"Epoch [{epoch + 1}/{epochs}] completed | "
                f"Average Loss: {avg_epoch_loss:.4f} | "
                f"Time: {epoch_time:.1f} min"
            )

    except KeyboardInterrupt:
        console.print("Training interrupted by user.")
        save_checkpoint(model, optimizer, global_step, epoch, run, os.path.join(save_model_dir, "checkpoint.pt"))
        console.print(f"Checkpoint saved at epoch {epoch+1}, global_step {global_step}")
        console.print("="*50)
        return

    # 6. Save final model
    final_path = os.path.join(save_model_dir, "sft_new.pt")
    os.makedirs(save_model_dir, exist_ok=True)
    torch.save(model.state_dict(), final_path)
    console.print(f"Model saved to {final_path}")

    # 7. Loss curve
    train_loss_monitor.finalize(save_path=os.path.join(save_model_dir, "train_loss_curve.png"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniSnail SFT")
    parser.add_argument("--config_path", type=str, help="Path to config JSON file")
    args = parser.parse_args()

    # 1. Load configuration
    if args.config_path:
        config = SnailConfig.from_json(args.config_path)
        console.print(f"Loaded config from {args.config_path}")
    elif os.path.exists("config.json"):
        config = SnailConfig.from_json("config.json")
        console.print("Loaded config from default config.json")
    else:
        config = DEFAULT_CONFIG
        console.print("Loaded default config")
    
    # 2. Load checkpoint
    if config.training.use_checkpoint:
        checkpoint = load_checkpoint(config.training.from_checkpoint)
        console.print(f"Loaded checkpoint from {config.training.from_checkpoint}, global_step: {checkpoint['global_step']}, wandb_id: {checkpoint['wandb_id']}")
    else:
        checkpoint = None
    
    # 3. Start a new wandb run to track this script.
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
    train_sft(config, run, checkpoint)
    end_time = time.time()

    console.print(f"Training time: {end_time - start_time:.2f} seconds")
    run.finish()
