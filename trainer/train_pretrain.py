import os
import time
import json
import wandb
import argparse
from typing import IO, BinaryIO
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer

from minisnail.dataset import PretrainDataset, get_dataloader
from minisnail.tokenizer import get_tokenizer
from minisnail.config import SnailConfig
from minisnail.util import setup_seed, print_train_config
from minisnail.debug import console
from minisnail.model import init_model

def save_checkpoint(
	model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    global_step: int,
    epoch: int,
    step: int,
    run: wandb.Run,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    """
    Given a model, optimizer, and an iteration number, serialize them to disk.

    Args:
        model (torch.nn.Module): Serialize the state of this model.
        optimizer (torch.optim.Optimizer): Serialize the state of this optimizer.
        global_step (int): Serialize this value, which represents the number of training iterations
            we've completed.
        epoch (int): Current epoch index.
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
            'last_step': step,
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

def train_loop(config: SnailConfig, dataloader: DataLoader, model: nn.Module, optimizer: Optimizer, checkpoint: dict[str, any] | None = None, run: wandb.Run | None = None):
    # 计算 epoch 和 step 的数量
    steps_per_epoch = len(train_dataloader)
    total_steps = len(train_dataloader) * config.training.epochs
    console.print(f"Train dataset samples: {len(train_dataloader.dataset)} \n"
               f"Val dataset samples: {len(val_dataloader.dataset)} \n"
               f"Total epochs: {config.training.epochs} \n"
               f"Total steps: {total_steps}")
    epoch_step = start_epoch = 0
    start_step = checkpoint['last_step'] if checkpoint is not None else 0
    global_step = 0
    if checkpoint is not None:
        global_step = checkpoint['global_step']
        start_epoch = global_step // steps_per_epoch if steps_per_epoch > 0 else 0

    # 调整混合精度
    

    console.print(f"[blue]Training start at epoch {start_epoch}, global_step {global_step}")
    model.train()
    try:
        for epoch in range(start_epoch, config.training.epochs):
            for step, (input_ids, labels) in enumerate(dataloader, start=start_step):
                input_ids, labels = input_ids.to(model.device), labels.to(model.device)
                global_step += 1
                epoch_step += 1

                with torch.autocast(device_type=model.device.type, dtype=amp_dtype, enabled=config.training.use_amp):
                    logits = model(input_ids)

                optimizer.zero_grad()
                optimizer.step()
                global_step += 1
                console.print(f"[blue]Epoch {epoch}, Step {step}, Loss: {loss.item():.4f}")
            
            epoch_step = 0
    
    except Exception as e:
        console.print(f"[red]Error at global_step {global_step} / {total_steps}: {e}")
        save_checkpoint(model, optimizer, global_step, epoch, epoch_step, run, os.path.join(config.data.save_model_dir, "checkpoint.pt"))
        console.print(f"[blue]Checkpoint saved at global_step {global_step} / {total_steps}")
        console.print("="*50)
        return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./config.json')
    parser.add_argument('--data_path', type=str, default='tests/data/pretrain.jsonl')
    parser.add_argument('--train_ratio', type=float, default=0.95)
    args = parser.parse_args()
    
    # 加载 config 和 tokenizer
    config = SnailConfig.from_json(args.config)
    print_train_config(config)
    tokenizer = get_tokenizer(config)
    # 设置随机种子
    setup_seed(config.system.seed)
    
    # 加载样本
    samples: list[dict] = []
    with open(args.data_path, 'r', encoding="utf-8") as f:
        for line in f:
            samples.append(json.loads(line))
    
    # 划分训练集和验证集
    split_index = int(len(samples) * args.train_ratio)
    train_samples = samples[:split_index]
    val_samples = samples[split_index:]

    train_dataset = PretrainDataset(
        samples=train_samples,
        tokenizer=tokenizer,
        max_length=config.model.context_length,
    )
    
    train_dataloader = get_dataloader(
        dataset=train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
    )
    
    val_dataset = PretrainDataset(
        samples=val_samples,
        tokenizer=tokenizer,
        max_length=config.model.context_length,
    )
    
    val_dataloader = get_dataloader(
        dataset=val_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
    )
    
    # 尝试加载断点
    checkpoint = None
    if config.training.use_checkpoint:
        checkpoint = load_checkpoint(config.training.from_checkpoint)
        console.print(f"Loaded checkpoint from {config.training.from_checkpoint}, global_step: {checkpoint['global_step']}, wandb_id: {checkpoint['wandb_id']}")

    # 使用 wandb
    run = None
    if config.training.use_wandb:
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

    # 加载模型 (默认从 config.training.from_weight 加载)
    model = init_model(config, model_path=config.training.from_weight)
    if checkpoint is not None:
        model.load_state_dict(checkpoint['model_state_dict'])
        console.print(f"[green]Loaded model state from checkpoint")

    # 编译模型
    if config.training.use_compile:
        model = torch.compile(model)
        console.print("[green]Model compiled successfully")
    
    # 加载 Tokenizer & 优化器
    tokenizer = get_tokenizer(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.training.lr, betas=config.training.betas, weight_decay=config.training.weight_decay)
    if checkpoint is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        console.print(f"[green]Loaded optimizer state from checkpoint")
    
    # 开始训练
    start_time = time.time()
    train_loop(config, train_dataloader, model, optimizer, run=run)
    end_time = time.time()

    console.print(f"[green]Training time: {end_time - start_time:.2f} s")

    