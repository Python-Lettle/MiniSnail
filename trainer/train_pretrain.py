import os
import time
import json
import wandb
import random
import argparse
import traceback
import numpy as np
from typing import IO, Any, BinaryIO
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
import torch.nn.functional as F

from minisnail.dataset import PretrainDataset, get_dataloader
from minisnail.functions import cosine_schedule, gradient_clipping
from minisnail.tokenizer import get_tokenizer
from minisnail.config import SnailConfig
from minisnail.util import setup_seed, print_train_config
from minisnail.debug import console
from minisnail.model import init_model

def save_checkpoint(
	model: torch.nn.Module,
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
    Given a model, optimizer, and an iteration number, serialize them to disk.

    Args:
        model (torch.nn.Module): Serialize the state of this model.
        optimizer (torch.optim.Optimizer): Serialize the state of this optimizer.
        scaler (torch.amp.GradScaler): Serialize the state of this gradient scaler.
        global_step (int): Serialize this value, which represents the number of training iterations
            we've completed.
        epoch (int): Next epoch index to resume from (已完成 epoch 数).
        step (int): Step index within the current epoch (保留字段, 恒为 0).
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
            'scaler_state_dict': scaler.state_dict(),
            'global_step': global_step,
            'epoch': epoch,
            'last_step': step,
            'optimizer_step': optimizer_step,
            'wandb_id': run.id if run is not None else None,
            "rng_state":{
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
    # Load the model state from the checkpoint
    checkpoint = torch.load(src, weights_only=False)

    return checkpoint

def train_loop(config: SnailConfig, train_dataloader: DataLoader, val_dataloader: DataLoader, model: nn.Module, optimizer: Optimizer, checkpoint: dict[str, Any] | None = None, run: wandb.Run | None = None):
    # 计算 epoch 和 step 的数量
    total_steps = len(train_dataloader) * config.training.epochs
    console.print(f"Train dataset samples: {len(train_dataloader.dataset)} \n"
               f"Val dataset samples: {len(val_dataloader.dataset)} \n"
               f"Total epochs: {config.training.epochs} \n"
               f"Total steps: {total_steps}")
    epoch = start_epoch = 0
    step = 0
    global_step = 0
    optimizer_step = 0
    if checkpoint is not None:
        # checkpoint 中 epoch 语义为"下一个待训练的 epoch"
        # (shuffle 的 dataloader 无法对齐轮内 batch, 因此不做轮内恢复, 从下一轮开头继续)
        start_epoch = checkpoint['epoch']
        global_step = checkpoint['global_step']
        optimizer_step = checkpoint['optimizer_step']

    # torch.compile 包装后的模型, 保存 state_dict 时使用原始模块, 避免键名带上 _orig_mod 前缀
    base_model = getattr(model, "_orig_mod", model)

    # 调整混合精度
    # 注意: dtype="float32" 时 amp_dtype 为 None, autocast 不启用 (真正纯 fp32 训练)
    model_dtype, amp_dtype = config.get_torch_dtype()
    autocast_enabled = config.training.use_amp and amp_dtype is not None

    # fp16 混合精度需要 GradScaler 防止梯度下溢; fp32 / bf16 不需要
    scaler = torch.amp.GradScaler(enabled=(config.training.use_amp and amp_dtype == torch.float16))
    if checkpoint is not None and 'scaler_state_dict' in checkpoint:
        scaler.load_state_dict(checkpoint['scaler_state_dict'])

    # 学习率: 初始化为 optimizer 当前 lr (含断点恢复值), 保证首个日志点前已定义
    current_lr = optimizer.param_groups[0]["lr"]

    # 梯度累积变量初始化
    has_pending_grads = False
    accumulated_steps = 0

    val_min_loss = np.inf

    def save_and_exit():
        """训练退出前: flush 残余梯度并保存 checkpoint"""
        nonlocal has_pending_grads
        # 若有累积的梯度, 则完成最后一次参数更新
        if has_pending_grads:
            # 异常可能发生在 unscale 之后, 此时不能重复 unscale
            try:
                scaler.unscale_(optimizer)
            except RuntimeError:
                pass
            gradient_clipping(model.parameters(), config.training.gradient_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            has_pending_grads = False
            console.print("[yellow]Pending gradients updated")
        save_checkpoint(
            base_model, optimizer, scaler, global_step, epoch + 1, 0, optimizer_step, run,
            os.path.join(config.data.save_model_dir, "checkpoint.pt"),
        )
        console.print(f"[green]Checkpoint saved at global_step {global_step} / {total_steps}")
        console.print("="*50)

    console.print(f"[blue]Training start at epoch {start_epoch}, global_step {global_step}")
    model.train()
    try:
        for epoch in range(start_epoch, config.training.epochs):
            for step, (input_ids, labels) in enumerate(train_dataloader):
                input_ids, labels = input_ids.to(model.device), labels.to(model.device)
                global_step += 1

                with torch.autocast(device_type=model.device.type, dtype=amp_dtype, enabled=autocast_enabled):
                    logits = model(input_ids)
                    loss = F.cross_entropy(
                        logits[:, :-1].contiguous().view(-1, config.model.vocab_size),
                        labels[:, 1:].contiguous().view(-1),
                        ignore_index=-100
                    )

                # 梯度累积：缩放 loss 使得梯度平均正确
                scaled_loss = loss / config.training.accumulation_steps
                scaler.scale(scaled_loss).backward()
                accumulated_steps += 1
                has_pending_grads = True

                # 如果到了梯度累积的步数，更新梯度
                if accumulated_steps % config.training.accumulation_steps == 0:
                    accumulated_steps = 0
                    optimizer_step += 1
                    # 梯度先反缩放再裁剪, 保证裁剪阈值作用于真实梯度值
                    scaler.unscale_(optimizer)
                    gradient_clipping(model.parameters(), config.training.gradient_clip)

                    # 学习率调度：每个迭代计算一次学习率
                    current_lr = cosine_schedule(
                        optimizer_step,
                        max_learning_rate=config.scheduler.max_learning_rate,
                        min_learning_rate=config.scheduler.min_learning_rate,
                        warmup_iters=config.scheduler.warmup_iters,
                        cosine_cycle_iters=config.scheduler.cosine_cycle_iters,
                    )
                    # 更新学习率
                    for param_group in optimizer.param_groups:
                        param_group["lr"] = current_lr

                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    has_pending_grads = False

                # Wandb 日志
                if run is not None:
                    run.log({
                        "train/loss": loss.item(),
                        "train/lr": current_lr,
                    }, step=global_step)

                # 终端输出训练信息
                if global_step % config.training.print_interval == 0:
                    console.print(f"Epoch [{epoch + 1}/{config.training.epochs}]")
                    console.print(f"Step {global_step}/{total_steps}")
                    console.print(f"Loss: {loss.item():.4f}")
                    console.print(f"{config.training.print_interval} Steps completed")
                    console.print("="*50)

                # ----------------------------------------
                #                Validate
                # ----------------------------------------
                if global_step % config.training.valid_interval == 0:
                    model.eval()
                    with torch.no_grad():
                        val_losses = []
                        # 随机采样 valid 样本，避免固定取数据集开头造成的偏置
                        # 注意: 索引范围是 dataset 样本数, 而不是 dataloader 的 batch 数
                        num_val_samples = len(val_dataloader.dataset)
                        valid_indices = random.sample(range(num_val_samples), min(20, num_val_samples))
                        for vi in valid_indices:
                            inputs_val, targets_val = val_dataloader.dataset[vi]
                            inputs_val, targets_val = inputs_val.unsqueeze(0).to(model.device), targets_val.unsqueeze(0).to(model.device)
                            with torch.autocast(device_type=model.device.type, dtype=amp_dtype, enabled=autocast_enabled):
                                val_logits = model(inputs_val)
                                val_loss = F.cross_entropy(
                                    val_logits[:, :-1].contiguous().view(-1, config.model.vocab_size),
                                    targets_val[:, 1:].contiguous().view(-1),
                                    ignore_index=-100
                                )
                            val_losses.append(val_loss.item())
                        val_loss_mean = np.mean(val_losses)
                        # 更新 val_min_loss
                        console.print(f"VALID mean loss: {val_loss_mean:.4f}")
                        is_min_loss = val_loss_mean < val_min_loss
                        if is_min_loss:
                            val_min_loss = val_loss_mean
                            torch.save(base_model.state_dict(), os.path.join(config.data.save_model_dir, "model_best.pt"))

                        # Wandb logging
                        if run is not None:
                            run.log({
                                "valid/loss": val_loss_mean,
                            }, step=global_step)    
                    model.train()

            # Epoch end
            console.print(f"[green]Epoch [{epoch + 1}/{config.training.epochs}] completed")

        # 训练正常结束: flush 残余梯度, 保存最终模型和 checkpoint
        console.print(f"[green]Training finished at global_step {global_step} / {total_steps}")
        torch.save(base_model.state_dict(), os.path.join(config.data.save_model_dir, "model_final.pt"))
        save_and_exit()

    except KeyboardInterrupt:
        # 手动中断: 保存断点以便续训
        console.print(f"[yellow]Interrupted by user at global_step {global_step} / {total_steps}")
        save_and_exit()

    except Exception:
        # 训练异常: 打印完整堆栈并保存断点
        console.print(f"[red]Error at global_step {global_step} / {total_steps}")
        console.print(traceback.format_exc())
        save_and_exit()

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

    # 恢复随机状态 (仅断点续训时; key 与 save_checkpoint 中的 rng_state 对应)
    if checkpoint is not None and 'rng_state' in checkpoint:
        rng_state = checkpoint['rng_state']
        torch.set_rng_state(rng_state['torch'])
        torch.cuda.set_rng_state_all(rng_state['cuda'])
        np.random.set_state(rng_state['numpy'])
        random.setstate(rng_state['python'])

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
    # 注意: 编译失败通常发生在首次 forward (而非 compile 调用时), 需要 warmup 验证
    if config.training.use_compile:
        try:
            compiled = torch.compile(model)
            with torch.no_grad():
                compiled(torch.zeros(1, 2, dtype=torch.long, device=model.device))
            model = compiled
            console.print("[green]Model compiled successfully")
        except Exception as e:
            console.print(f"[yellow]torch.compile 不可用, 回退 eager 模式: {type(e).__name__}: {e}")
    
    # 加载 Tokenizer & 优化器
    tokenizer = get_tokenizer(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.training.lr, betas=config.training.betas, weight_decay=config.training.weight_decay)
    if checkpoint is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        console.print(f"[green]Loaded optimizer state from checkpoint")
    
    # 开始训练
    start_time = time.time()
    train_loop(config, train_dataloader, val_dataloader, model, optimizer, checkpoint=checkpoint, run=run)
    end_time = time.time()

    console.print(f"[green]Training time: {end_time - start_time:.2f} s")

    