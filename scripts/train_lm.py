from minisnail.debug import console, DEBUG, LossMonitor
from minisnail.functions import cross_entropy_loss, cosine_schedule, gradient_clipping
from minisnail.config import SnailConfig, DEFAULT_CONFIG
from minisnail.model import init_model
from minisnail.dataset import get_dataloader
from minisnail.util import setup_seed, load_config
import torch
from typing import IO, BinaryIO
import numpy as np
import os
import time
import random
import argparse
import wandb

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

def train_lm(config: SnailConfig = DEFAULT_CONFIG, wandb_run = None, checkpoint = None):
    setup_seed(config.system.seed)
    # 1. Prepare training parameters
    train_data_path = config.data.train_data_path
    valid_data_path = config.data.valid_data_path

    save_model_dir = config.data.save_model_dir
    os.makedirs(save_model_dir, exist_ok=True)
    
    device = torch.device(config.system.device)
    model_dtype, amp_dtype = config.get_torch_dtype()
    use_amp = amp_dtype is not None
    epochs: int = config.training.epochs
    valid_interval: int = config.training.valid_interval
    lr: float = config.training.lr
    betas: tuple[float, float] = config.training.betas
    weight_decay: float = config.training.weight_decay
    
    console.print("Training data path:", train_data_path)
    console.print("Validation data path:", valid_data_path)
    console.print("Save model dir:", save_model_dir)
    console.print("Epochs:", epochs)
    console.print("Valid interval:", valid_interval)
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
    accumulation_steps: int = config.training.accumulation_steps

    console.print("vocab_size:", vocab_size)
    console.print("context_length:", context_length)
    console.print("d_model:", d_model)
    console.print("num_layers:", num_layers)
    console.print("num_heads:", num_heads)
    console.print("d_ff:", d_ff)
    console.print("rope_theta:", rope_theta)
    console.print("batch_size:", batch_size)
    console.print("accumulation_steps:", accumulation_steps)
    console.print(f"Mixed precision (AMP): {amp_dtype if use_amp else 'disabled (fp32)'}")

    # 2. Create dataloaders (torch official DataLoader)
    train_loader = get_dataloader(
        train_data_path, block_size=context_length, batch_size=batch_size,
    )
    valid_loader = get_dataloader(
        valid_data_path, block_size=context_length, batch_size=batch_size,
        shuffle=False, drop_last=False,
    )

    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * epochs
    console.print(f"数据集大小: {len(train_loader.dataset)} 条, "
               f"每 epoch {steps_per_epoch} 步, "
               f"共 {epochs} epoch = {total_steps} 步")

    # 3. Create the model and optimizer
    model = init_model(config, device=device, dtype=model_dtype)
    if config.training.from_weight:
        model.load_state_dict(torch.load(config.training.from_weight, weights_only=False))
        console.print("[yellow]Loading model from weight:", config.training.from_weight)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=betas, eps=1e-8, weight_decay=weight_decay)

    # 4. Load checkpoint
    start_epoch = 0
    global_step = 0
    if checkpoint is not None:
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        global_step = checkpoint['global_step']
        start_epoch = global_step // steps_per_epoch if steps_per_epoch > 0 else 0
        console.print("Load checkpoint from global_step:", global_step)

    train_loss_monitor = LossMonitor(title="Train Loss Monitor", show_stats=False)
    valid_loss_monitor = LossMonitor(title="Valid Loss Monitor", show_stats=False)

    # 5. Train the model
    console.print("Training start at epoch", start_epoch)
    has_pending_grads = False
    min_loss = float('inf')
    model.train()
    try:
        for epoch in range(start_epoch, epochs):
            console.print(f"\n{'='*40}")
            console.print(f"Epoch [{epoch + 1}/{epochs}]")
            console.print(f"{'='*40}")

            epoch_loss = 0.0
            epoch_steps = 0
            epoch_start = time.time()

            for step, (inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                global_step += 1
                epoch_steps += 1

                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                    logits = model(inputs)
                    loss = cross_entropy_loss(logits, targets)

                # Gradient accumulation: scale loss so gradients average correctly
                scaled_loss = loss / accumulation_steps
                scaled_loss.backward()
                has_pending_grads = True

                # Learning rate scheduling (computed every iteration for logging)
                current_lr = cosine_schedule(
                    global_step,
                    max_learning_rate=config.scheduler.max_learning_rate,
                    min_learning_rate=config.scheduler.min_learning_rate,
                    warmup_iters=config.scheduler.warmup_iters,
                    cosine_cycle_iters=config.scheduler.cosine_cycle_iters,
                )

                # Perform optimizer step every accumulation_steps iterations
                if global_step % accumulation_steps == 0:
                    # Gradient clipping
                    gradient_clipping(model.parameters(), config.training.gradient_clip)

                    # Update learning rate
                    for param_group in optimizer.param_groups:
                        param_group["lr"] = current_lr

                    optimizer.step()
                    optimizer.zero_grad()
                    has_pending_grads = False

                epoch_loss += loss.item()
                train_loss_monitor.add_loss(global_step, loss.item())

                # Wandb logging
                if wandb_run is not None:
                    wandb_run.log({
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

                # ----------------------------------------
                #                Validate
                # ----------------------------------------
                if global_step % valid_interval == 0:
                    model.eval()
                    with torch.no_grad():
                        val_losses = []
                        # 随机采样 valid batches，避免固定取数据集开头造成的偏置
                        valid_indices = random.sample(range(len(valid_loader)), min(20, len(valid_loader)))
                        for vi in valid_indices:
                            inputs_val, targets_val = valid_loader.dataset[vi]
                            inputs_val, targets_val = inputs_val.unsqueeze(0).to(device), targets_val.unsqueeze(0).to(device)
                            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                                val_logits = model(inputs_val)
                            val_loss = cross_entropy_loss(val_logits, targets_val)
                            val_losses.append(val_loss.item())
                        val_loss_mean = np.mean(val_losses)
                        is_min_loss = valid_loss_monitor.add_loss(global_step, val_loss_mean)
                        console.print(f"VALID mean loss: {val_loss_mean:.4f}")

                        # Wandb logging
                        if wandb_run is not None:
                            wandb_run.log({
                                "valid/loss": val_loss_mean,
                            }, step=global_step)
                        if is_min_loss:
                            torch.save(model.state_dict(), os.path.join(save_model_dir, "model_best.pt"))
                    model.train()

            # Epoch end
            avg_epoch_loss = epoch_loss / max(epoch_steps, 1)
            epoch_time = (time.time() - epoch_start) / 60
            console.print(
                f"Epoch [{epoch + 1}/{epochs}] completed | "
                f"Average Loss: {avg_epoch_loss:.4f} | "
                f"Time: {epoch_time:.1f} min"
            )

    except AssertionError as e:
        console.print(f"AssertionError at global_step {global_step} / {total_steps}: {e}")
        console.print(f"Max token ID: {inputs.max()}")
        console.print(f"Max token ID: {targets.max()}")
        save_checkpoint(model, optimizer, global_step, epoch, wandb_run, os.path.join(save_model_dir, "checkpoint.pt"))
        console.print(f"Checkpoint saved at global_step {global_step} / {total_steps}")
        console.print("="*50)
        return
    except KeyboardInterrupt:
        console.print(f"KeyboardInterrupt at global_step {global_step} / {total_steps}")
        save_checkpoint(model, optimizer, global_step, epoch, wandb_run, os.path.join(save_model_dir, "checkpoint.pt"))
        console.print(f"Checkpoint saved at global_step {global_step} / {total_steps}")
        console.print("="*50)
        return
    except Exception as e:
        console.print(f"Error at global_step {global_step} / {total_steps}: {e}")
        save_checkpoint(model, optimizer, global_step, epoch, wandb_run, os.path.join(save_model_dir, "checkpoint.pt"))
        console.print(f"Checkpoint saved at global_step {global_step} / {total_steps}")
        console.print("="*50)
        return

    # Apply remaining accumulated gradients from the last partial accumulation cycle
    if has_pending_grads:
        gradient_clipping(model.parameters(), config.training.gradient_clip)
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr
        optimizer.step()
        optimizer.zero_grad()
        console.print("Applied final optimizer step for remaining accumulated gradients")

    # 6. Save model
    final_path = os.path.join(save_model_dir, "model_new.pt")
    torch.save(model.state_dict(), final_path)
    console.print(f"Model saved to {final_path}")

    # 7. Loss curve
    train_loss_monitor.finalize(save_path=os.path.join(save_model_dir, "train_loss_curve.png"))
    valid_loss_monitor.finalize(save_path=os.path.join(save_model_dir, "valid_loss_curve.png"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Language Model")
    parser.add_argument("--config", help="Path to config JSON file")
    args = parser.parse_args()

    # 1. Load configuration
    config = load_config(args)

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
    train_lm(config, run, checkpoint)
    end_time = time.time()

    console.print(f"Training time: {end_time - start_time:.2f} seconds")
    run.finish()