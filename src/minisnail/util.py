import os
import random
import numpy as np
import argparse
import torch
from rich.console import Console
from minisnail.config import SnailConfig, DEFAULT_CONFIG
console = Console()

def setup_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def read_memmap_data(train_data_path: str | os.PathLike):
    '''
    Read the training dataset from disk.
    Args:
        train_data_path (str | os.PathLike): Path to the training dataset.
    Returns:
        np.memmap: The training dataset as a numpy memory-mapped array.
    '''
    return np.memmap(
        train_data_path,
        dtype=np.int32,
        mode="r",
    )

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