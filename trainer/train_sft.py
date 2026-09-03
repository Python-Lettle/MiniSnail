import os
# torch.compile: 默认缓存目录 (~/.triton 与 %TEMP%\torchinductor_*) 
# 会被拦截 (PermissionError: WinError 5), 指到项目目录规避;
# 必须在 import torch 之前设置, setdefault 保证外部显式设置优先
os.environ.setdefault("TRITON_CACHE_DIR", os.path.abspath("./.cache/triton"))
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", os.path.abspath("./.cache/inductor"))
import json
import time
import wandb
import random
import argparse
import traceback
import numpy as np
from typing import IO, Any, BinaryIO
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.optim import Optimizer
import torch.nn.functional as F

from minisnail.dataset import SFTDataset, get_dataloader, get_epoch_dataloader
from minisnail.functions import apply_optimizer_step
from minisnail.chat_protocol import CHAT_PROTOCOL_VERSION
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
    # 显式映射到 CPU, 避免续训时 checkpoint 权重+优化器状态占用双份显存直至 OOM
    checkpoint = torch.load(src, map_location="cpu", weights_only=False)

    return checkpoint

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
        # 保存 checkpoint
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
                    console.print(f"LR: {current_lr:.2e}")
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
                            torch.save(base_model.state_dict(), os.path.join(save_model_dir, "sft_best.pt"))

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
        save_and_exit()  # 先 flush 残余梯度并保存 checkpoint,保证 sft_final 与 checkpoint 权重一致
        torch.save(base_model.state_dict(), os.path.join(save_model_dir, "sft_final.pt"))

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

def load_sft_meta(data_dir: str) -> dict | None:
    """读取预处理产出的 sft_meta.json, 没有则返回 None。"""
    path = os.path.join(data_dir, "sft_meta.json")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def validate_sft_context_length(dataset_length: int, model_length: int) -> None:
    """拒绝使用与模型上下文长度不一致的预处理数据。"""
    if dataset_length != model_length:
        raise ValueError(
            f"npy 序列长度 {dataset_length} 与 "
            f"config.model.context_length={model_length} 不一致；"
            f"请用相同的 --max_length 重新预处理，拒绝以错误配置继续训练"
        )


def validate_sft_protocol(meta: dict | None) -> None:
    """Reject legacy shards whose label mask/template semantics are unknown."""
    actual = meta.get("chat_protocol") if meta else None
    if actual != CHAT_PROTOCOL_VERSION:
        raise ValueError(
            f"SFT 数据协议为 {actual!r}，当前要求 {CHAT_PROTOCOL_VERSION!r}；"
            "请重新运行 scripts/preprocess_sft_data.py，不能混用旧分片"
        )

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./config.json')
    parser.add_argument('--data_dir', type=str, default='./dataset/full',
                        help='经过预处理后数据的目录 (含 sft_input_ids*.npy 与 sft_labels*.npy)')
    parser.add_argument('--save_model_dir', type=str, default='./output/new_sft')
    parser.add_argument('--valid_ratio', type=float, default=0.005,
                        help='验证集占比 (从数据集中按固定 seed 划分)')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='只用前 N 条样本 (调试用, 不限制则跑全量)')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    # 加载 config 和 tokenizer
    config = load_config(args)
    print_train_config(config)
    tokenizer = get_tokenizer(config)
    # 设置随机种子
    setup_seed(config.system.seed)

    if config.training.from_weight is None:
        console.print("[red]training.from_weight 为空, 将从随机初始化开始 SFT (通常应指向预训练权重)")

    # 加载数据 (mmap 读 npy, 不把全量矩阵读进物理内存)
    # 全量约 500 万条 x 512, int16 单份就 5GB+, 直接 np.load 常驻内存会挤爆
    full_dataset = SFTDataset(data_dir=args.data_dir)

    meta = load_sft_meta(args.data_dir)
    try:
        validate_sft_protocol(meta)
    except ValueError as e:
        raise SystemExit(f"[error] {e}") from e
    if meta is not None:
        console.print(f"[Data] 预处理: 输入 {meta.get('total_input_samples')} 条 -> 保留 "
                      f"{meta.get('num_samples')} 条 | 分片 {meta.get('num_shards')} 个 | "
                      f"dtype={meta.get('dtype')}")
        console.print(f"[Data] 丢弃统计: {meta.get('stats')}")
    try:
        validate_sft_context_length(
            full_dataset.max_length,
            config.model.context_length,
        )
    except ValueError as e:
        raise SystemExit(f"[error] {e}") from e

    # train/val 划分: 用固定 seed 对全量样本做 permutation 后按比例切分
    # (固定 seed 保证重启/断点续训时划分完全一致, 验证集不会混入训练集)
    n_total = len(full_dataset)
    if args.max_samples:
        n_total = min(n_total, int(args.max_samples))
    valid_size = max(1, int(n_total * args.valid_ratio))
    if valid_size >= n_total:
        raise SystemExit(
            f"[error] 验证集 {valid_size} 条 >= 可用样本 {n_total} 条, 请调小 --valid_ratio")
    perm = np.random.default_rng(config.system.seed).permutation(n_total)
    val_indices, train_indices = perm[:valid_size], perm[valid_size:]

    train_dataset = Subset(full_dataset, train_indices.tolist())
    val_dataset = Subset(full_dataset, val_indices.tolist())
    console.print(f"[Data] 总样本 {n_total} -> 训练 {len(train_dataset)}, 验证 {len(val_dataset)}")

    train_dataloader = get_dataloader(
        dataset=train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        drop_last=False,
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

    # 加载模型 (默认从 config.training.from_weight 加载预训练权重)
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

    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.training.lr, betas=config.training.betas, weight_decay=config.training.weight_decay)
    if checkpoint is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        console.print(f"[green]Loaded optimizer state from checkpoint")

    # 放到所有初始化之后，避免模型初始化、compile warmup 或 W&B 再次推进 RNG。
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
