from minisnail.debug import console, DEBUG, LossMonitor
from minisnail.functions import cross_entropy_loss, cosine_schedule, gradient_clipping
from minisnail.config import SnailConfig, DEFAULT_CONFIG
from minisnail.model import init_model
from minisnail.util import read_memmap_data, data_loader, setup_seed
import torch
from typing import IO, BinaryIO
from tqdm import tqdm
import numpy as np
import numpy.typing as npt
import os
import time
import argparse
import wandb

def save_checkpoint(
	model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
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
            'iteration': iteration,
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

def val_iterator(memmap_arr, batch_size, context_length):
    N = len(memmap_arr)
    nb = (N-context_length-1)//batch_size
    for bi in range(nb):
        base = bi*batch_size
        x = np.stack([memmap_arr[i:i+context_length] for i in range(base, base+batch_size)])
        y = np.stack([memmap_arr[i+1:i+context_length+1] for i in range(base, base+batch_size)])
        yield torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)

def train_lm(config: SnailConfig = DEFAULT_CONFIG, wandb_run = None, checkpoint = None):
    setup_seed(config.system.seed)
    # 1. Prepare training parameters
    train_data_path = config.data.train_data_path
    valid_data_path = config.data.valid_data_path

    save_model_dir = config.data.save_model_dir
    os.makedirs(save_model_dir, exist_ok=True)
    
    device = torch.device(config.system.device)
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

    console.print("vocab_size:", vocab_size)
    console.print("context_length:", context_length)
    console.print("d_model:", d_model)
    console.print("num_layers:", num_layers)
    console.print("num_heads:", num_heads)
    console.print("d_ff:", d_ff)
    console.print("rope_theta:", rope_theta)
    console.print("batch_size:", batch_size)

    # 2. Load the datasets
    train_data_tokens = read_memmap_data(train_data_path)    # 1D array
    console.print("train_data_tokens shape:", train_data_tokens.shape)

    valid_data_tokens = read_memmap_data(valid_data_path)    # 1D array
    console.print("valid_data_tokens shape:", valid_data_tokens.shape)

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
        start_epoch = checkpoint['iteration']
        console.print("Load checkpoint from:", start_epoch)

    train_loss_monitor = LossMonitor(title="Train Loss Monitor", show_stats=False)
    valid_loss_monitor = LossMonitor(title="Valid Loss Monitor", show_stats=False)

    # 5. Train the model
    console.print("Training start at epoch", start_epoch)
    for epoch in range(start_epoch, epochs):
        try:
            # ----------------------------------------
            #                  Train
            # ----------------------------------------
            model.train()
            inputs, targets = data_loader(train_data_tokens, batch_size=batch_size, context_length=context_length, device=device)

            logits = model(inputs)
            loss = cross_entropy_loss(logits, targets)

            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping and learning rate scheduling
            gradient_clipping(model.parameters(), config.training.gradient_clip)

            # Learning rate scheduling
            current_lr = cosine_schedule(
                epoch,
                max_learning_rate=config.scheduler.max_learning_rate,
                min_learning_rate=config.scheduler.min_learning_rate,
                warmup_iters=config.scheduler.warmup_iters,
                cosine_cycle_iters=config.scheduler.cosine_cycle_iters,
            )
            for param_group in optimizer.param_groups:
                param_group["lr"] = current_lr

            optimizer.step()
            train_loss_monitor.add_loss(epoch, loss.item())

            # Wandb logging
            if wandb_run is not None:
                wandb_run.log({
                    "train/loss": loss.item(),
                    "train/lr": current_lr,
                }, step=epoch)

            # Print information
            if epoch % config.training.print_interval == 0:
                console.print(f"Epoch {epoch+1}/{epochs}")
                console.print(f"Loss: {loss.item():.4f}")
                console.print("Epoch completed")
                console.print("="*50)
            
            # ----------------------------------------
            #                Validate
            # ----------------------------------------
            if epoch % valid_interval == 0:
                model.eval()
                with torch.no_grad():
                    val_losses = []
                    count = 0
                    for inputs_val, targets_val in val_iterator(valid_data_tokens, batch_size, context_length):
                        inputs_val, targets_val = inputs_val.to(device), targets_val.to(device)
                        val_logits = model(inputs_val)
                        val_loss = cross_entropy_loss(val_logits, targets_val)
                        # Collect val_loss
                        val_losses.append(val_loss.item())
                        count += 1
                        if count >= 10:
                            break
                    val_loss_mean = np.mean(val_losses)
                    is_min_loss = valid_loss_monitor.add_loss(epoch, val_loss_mean)
                    console.print(f"VALID mean loss: {val_loss_mean:.4f}")
                    
                    # Wandb logging
                    if wandb_run is not None:
                        wandb_run.log({
                            "valid/loss": val_loss_mean,
                        }, step=epoch)
                    if is_min_loss:
                        torch.save(model.state_dict(), os.path.join(save_model_dir, "model_best.pt"))


        except AssertionError as e:
            console.print(f"AssertionError in epoch {epoch+1}: {e}")
            console.print(f"Max token ID: {inputs.max()}")
            console.print(f"Max token ID: {targets.max()}")
            # Save checkpoint
            save_checkpoint(model, optimizer, epoch, run, os.path.join(save_model_dir, "checkpoint.pt"))
            console.print(f"Checkpoint saved at epoch {epoch+1}")
            console.print("="*50)
            return
        except KeyboardInterrupt:
            console.print(f"KeyboardInterrupt in epoch {epoch+1}")
            # Save checkpoint
            save_checkpoint(model, optimizer, epoch, run, os.path.join(save_model_dir, "checkpoint.pt"))
            console.print(f"Checkpoint saved at epoch {epoch+1}")
            console.print("="*50)
            return
        except Exception as e:
            console.print(f"Error in epoch {epoch+1}: {e}")
            # Save checkpoint
            save_checkpoint(model, optimizer, epoch, run, os.path.join(save_model_dir, "checkpoint.pt"))
            console.print(f"Checkpoint saved at epoch {epoch+1}")
            console.print("="*50)
            return
    
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
    if args.config:
        config = SnailConfig.from_json(args.config)
        console.print(f"Loaded config from {args.config}")
    elif os.path.exists("config.json"):
        config = SnailConfig.from_json("config.json")
        console.print("Loaded config from default config.json")
    else:
        config = DEFAULT_CONFIG
        console.print("Loaded default config")

    # 2. Load checkpoint
    if config.training.use_checkpoint:
        checkpoint = load_checkpoint(config.training.from_checkpoint)
        console.print(f"Loaded checkpoint from {config.training.from_checkpoint}, iteration: {checkpoint['iteration']}, wandb_id: {checkpoint['wandb_id']}")
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