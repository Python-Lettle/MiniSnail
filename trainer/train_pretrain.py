import os
# torch.compile: 默认缓存目录 (~/.triton 与 %TEMP%\torchinductor_*) 在 TRAE 沙箱内
# 会被拦截 (PermissionError: WinError 5), 指到项目目录规避;
# 必须在 import torch 之前设置, setdefault 保证外部显式设置优先
os.environ.setdefault("TRITON_CACHE_DIR", os.path.abspath("./.cache/triton"))
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", os.path.abspath("./.cache/inductor"))
import time
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

from minisnail.dataset import LazyPretrainDataset, get_dataloader, get_epoch_dataloader, load_line_offsets
from minisnail.functions import apply_optimizer_step
from minisnail.tokenizer import get_tokenizer
from minisnail.config import SnailConfig
from minisnail.util import setup_seed, restore_rng_state, load_config, print_train_config
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
    val_min_loss: float,
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
        epoch (int): Index of the epoch the checkpoint was saved in (正在训练的 epoch).
        step (int): Number of completed batches within that epoch (轮内续训时跳过的 batch 数).
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
            'val_min_loss': val_min_loss,
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
    # 注意: checkpoint 是从 GPU 模型保存的, 默认 map_location 会把张量加载回原设备(GPU),
    # 导致续训时同时占用一份 checkpoint 权重+优化器状态和新建模型, 显存翻倍直至 OOM。
    # 显式映射到 CPU, load_state_dict 会按需拷贝到模型所在设备。
    checkpoint = torch.load(src, map_location="cpu", weights_only=False)

    return checkpoint


def initialize_pretrain_model(
    config: SnailConfig,
    checkpoint: dict[str, Any] | None = None,
) -> nn.Module:
    """初始化预训练模型；只允许随机初始化或恢复同阶段 checkpoint。"""
    if checkpoint is None and config.training.from_weight is not None:
        console.print("[yellow]Pretrain ignores training.from_weight and starts from scratch")
    model = init_model(config)
    if checkpoint is not None:
        model.load_state_dict(checkpoint['model_state_dict'])
        console.print("[green]Loaded model state from checkpoint")
    return model

def train_loop(config: SnailConfig, train_dataloader: DataLoader, val_dataloader: DataLoader, model: nn.Module, optimizer: Optimizer, save_model_dir: str = "./output", checkpoint: dict[str, Any] | None = None, run: wandb.Run | None = None):
    # 计算 epoch 和 step 的数量
    total_steps = len(train_dataloader) * config.training.epochs
    total_optimizer_steps = (
        total_steps + config.training.accumulation_steps - 1
    ) // config.training.accumulation_steps
    console.print(f"Train dataset samples: {len(train_dataloader.dataset)} \n"
               f"Val dataset samples: {len(val_dataloader.dataset)} \n"
               f"Total epochs: {config.training.epochs} \n"
               f"Total steps: {total_steps}")
    epoch = start_epoch = 0
    global_step = 0
    optimizer_step = 0
    start_step = 0
    epoch_step = 0
    if checkpoint is not None:
        # checkpoint 语义: epoch = 断点所在 (正在训练的) epoch, last_step = 该 epoch 内已完成的 batch 数
        # 续训时按 (seed, epoch) 确定性重建同一 batch 顺序, 再跳过已完成的 last_step 个 batch
        start_epoch = checkpoint['epoch']
        global_step = checkpoint['global_step']
        optimizer_step = checkpoint['optimizer_step']
        start_step = checkpoint.get('last_step', 0)

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

    val_min_loss = checkpoint.get('val_min_loss', np.inf) if checkpoint else np.inf

    def apply_pending_gradients():
        """提交当前累积窗口；残余窗口会按实际 micro-batch 数重新归一化。"""
        nonlocal has_pending_grads, accumulated_steps, optimizer_step, current_lr
        if not has_pending_grads:
            return
        optimizer_step, current_lr = apply_optimizer_step(
            model.parameters(),
            optimizer,
            scaler,
            accumulated_steps=accumulated_steps,
            accumulation_steps=config.training.accumulation_steps,
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
            base_model, optimizer, scaler, global_step, epoch, epoch_step,
            optimizer_step, val_min_loss, run,
            os.path.join(save_model_dir, "checkpoint.pt"),
        )
        console.print(f"[green]Checkpoint saved to {save_model_dir} at global_step {global_step} / {total_steps}")
        console.print("="*50)

    console.print(f"[blue]Training start at epoch {start_epoch}, step {start_step}, global_step {global_step}")
    model.train()
    try:
        for epoch in range(start_epoch, config.training.epochs):
            # 每个 epoch 用独立 seed 确定性洗牌, 续训时重建相同 batch 顺序并跳过已完成的 batch
            skip = start_step if epoch == start_epoch else 0
            # 先更新 epoch_step 再构建 loader: 若中断恰好发生在 loader 构造期间,
            # 保存的 last_step 仍是本 epoch 的正确跳过值, 而不是上一轮的残留值
            epoch_step = skip
            epoch_loader = get_epoch_dataloader(
                train_dataloader.dataset,
                batch_size=train_dataloader.batch_size,
                seed=config.system.seed,
                epoch=epoch,
                skip_batches=skip,
            )
            for step, (input_ids, labels) in enumerate(epoch_loader):
                input_ids, labels = input_ids.to(model.device), labels.to(model.device)
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
                # 只有 forward/backward 均成功后，这个 batch 才能记作已完成。
                global_step += 1
                epoch_step += 1

                # 如果到了梯度累积的步数，更新梯度
                if accumulated_steps == config.training.accumulation_steps:
                    apply_pending_gradients()

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
                    console.print(f"Optimizer Step: {optimizer_step}/{total_optimizer_steps}")
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
                        valid_indices = random.sample(range(num_val_samples), min(config.training.valid_batches, num_val_samples))
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
                            torch.save(base_model.state_dict(), os.path.join(save_model_dir, "model_best.pt"))

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
        epoch = config.training.epochs  # 标记全部 epoch 已完成, 续训时 range(epochs, epochs) 为空直接结束
        epoch_step = 0
        save_and_exit()  # 先 flush 残余梯度并保存 checkpoint, 保证 model_final 与 checkpoint 权重一致
        torch.save(base_model.state_dict(), os.path.join(save_model_dir, "model_final.pt"))

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./config.json')
    parser.add_argument('--data_path', type=str, default='./dataset/full/pretrain_t2t.jsonl')
    parser.add_argument('--save_model_dir', type=str, default='./output/new_pretrain')
    parser.add_argument('--train_ratio', type=float, default=0.95)
    args = parser.parse_args()
    
    # 加载 config 和 tokenizer
    config = load_config(args)
    print_train_config(config)
    tokenizer = get_tokenizer(config)
    # 设置随机种子
    setup_seed(config.system.seed)
    
    # 加载数据 (懒加载: 只建行偏移索引, 不把全量文本读入内存)
    # 首次运行扫描全文件建索引并缓存到 <data_path>.idx.npz, 之后秒级启动
    # train/val 划分: 用固定 seed 对全量行号做 permutation 后按比例切分
    # (固定 seed 保证重启/断点续训时划分完全一致, 验证集不会混入训练集)
    num_lines = len(load_line_offsets(args.data_path))
    split_index = int(num_lines * args.train_ratio)
    perm = np.random.default_rng(config.system.seed).permutation(num_lines)
    train_indices, val_indices = perm[:split_index], perm[split_index:]

    train_dataset = LazyPretrainDataset(
        data_path=args.data_path,
        tokenizer=tokenizer,
        max_length=config.model.context_length,
        indices=train_indices,
    )

    train_dataloader = get_dataloader(
        dataset=train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        drop_last=False,
    )

    val_dataset = LazyPretrainDataset(
        data_path=args.data_path,
        tokenizer=tokenizer,
        max_length=config.model.context_length,
        indices=val_indices,
    )
    
    val_dataloader = get_dataloader(
        dataset=val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        drop_last=True,
    )
    
    # 尝试加载断点
    checkpoint = None
    if config.training.use_checkpoint:
        if not config.training.from_checkpoint:
            raise SystemExit(
                "[error] use_checkpoint=True 但 training.from_checkpoint 未设置"
            )
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

    # 预训练必须从随机初始化或 checkpoint 开始。training.from_weight 属于后续
    # SFT/DPO 阶段的共享配置，不能在这里静默加载，避免阶段间权重污染。
    model = initialize_pretrain_model(config, checkpoint)

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

    # 必须放在 W&B、模型初始化、compile warmup 和优化器恢复之后；这些步骤都可能
    # 消耗随机数。这样训练循环看到的才是 checkpoint 保存时的精确 RNG 状态。
    if checkpoint is not None and 'rng_state' in checkpoint:
        restore_rng_state(checkpoint['rng_state'])
        console.print("[green]Restored RNG state from checkpoint")

    # 开始训练
    os.makedirs(args.save_model_dir, exist_ok=True)
    start_time = time.time()
    train_loop(config, train_dataloader, val_dataloader, model, optimizer,
               save_model_dir=args.save_model_dir, checkpoint=checkpoint, run=run)
    end_time = time.time()

    console.print(f"[green]Training time: {end_time - start_time:.2f} s")

    if run is not None:
        run.finish()
