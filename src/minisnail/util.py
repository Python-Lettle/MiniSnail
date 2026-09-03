import random
import os
import argparse
import numpy as np
import torch

from minisnail.debug import console
from minisnail.config import SnailConfig, DEFAULT_CONFIG

def setup_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def restore_rng_state(rng_state: dict) -> None:
    """在所有会消耗随机数的初始化完成后恢复 checkpoint 的 RNG 状态。"""
    torch.set_rng_state(rng_state["torch"])
    if torch.cuda.is_available() and rng_state.get("cuda") is not None:
        torch.cuda.set_rng_state_all(rng_state["cuda"])
    np.random.set_state(rng_state["numpy"])
    random.setstate(rng_state["python"])

def print_train_config(config: SnailConfig):
    console.print(f"---------- Training Config ----------")
    console.print("Device:", config.system.device)
    console.print("Dtype:", config.system.dtype)
    console.print("Seed:", config.system.seed)
    console.print("Use checkpoint:", config.training.use_checkpoint)
    console.print("Use wandb:", config.training.use_wandb)
    console.print("Use compile:", config.training.use_compile)
    
    console.print("Epochs:", config.training.epochs)
    console.print("Batch size:", config.training.batch_size)
    console.print("Accumulation steps:", config.training.accumulation_steps)
    
    console.print("lr:", config.training.lr)
    console.print("betas:", config.training.betas)
    console.print("weight_decay:", config.training.weight_decay)

    console.print("max_learning_rate:", config.scheduler.max_learning_rate)
    console.print("min_learning_rate:", config.scheduler.min_learning_rate)
    console.print("warmup_iters:", config.scheduler.warmup_iters)
    console.print("cosine_cycle_iters:", config.scheduler.cosine_cycle_iters)

    console.print("Valid interval:", config.training.valid_interval)
    console.print("Print interval:", config.training.print_interval)

    console.print(f"---------- Model Config ----------")
    console.print("Vocab size:", config.model.vocab_size)
    console.print("Context length:", config.model.context_length)
    console.print("d_model:", config.model.d_model)
    console.print("Num layers:", config.model.num_layers)
    console.print("Num heads:", config.model.num_heads)
    console.print("FFN hidden size:", config.model.d_ff)
    console.print("RoPE theta:", config.model.rope_theta)
    console.print("rms_norm_eps:", config.model.rms_norm_eps)

def load_config(args: argparse.Namespace) -> SnailConfig:
    config = DEFAULT_CONFIG
    if args.config:
        config = SnailConfig.from_json(args.config)
        console.print(f"Loaded config from {args.config}")
    elif os.path.exists("config.json"):
        config = SnailConfig.from_json("config.json")
        console.print("Loaded config from default config.json")
    else:
        console.print("Loaded default config")
    return config
