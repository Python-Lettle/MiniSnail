import os
import random
import numpy as np
import torch
from rich.console import Console
console = Console()

def setup_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def read_memmap_data(train_data_path: str | os.PathLike):
    '''Read the memmap file.
    '''
    return np.memmap(
        train_data_path,
        dtype=np.int32,
        mode="r",
    )