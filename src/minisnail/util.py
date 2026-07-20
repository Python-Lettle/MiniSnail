import os
import random
import numpy as np
import numpy.typing as npt
import torch
from rich.console import Console
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

def data_loader(
    dataset: npt.NDArray,
    batch_size: int,
    context_length: int,
    device,
)-> tuple[torch.Tensor, torch.Tensor]:
    '''
    Given a dataset (a 1D numpy array of integers) and a desired batch size and
    context length, sample language modeling input sequences and their corresponding
    labels from the dataset.

    Args:
        dataset (np.array): 1D numpy array of integer token IDs in the dataset.
        batch_size (int): Desired batch size to sample.
        context_length (int): Desired context length of each sampled example.
        device (str): PyTorch device string (e.g., 'cpu' or 'cuda:0') indicating the device
            to place the sampled input sequences and labels on.

    Returns:
        Tuple of torch.LongTensors of shape (batch_size, context_length). The first tuple item
        is the sampled input sequences, and the second tuple item is the corresponding
        language modeling labels.
    '''
    # 1. Randomly sample start positions in the dataset for each batch
    indices = np.random.randint(
        low=0,
        high=len(dataset) - context_length,
        size=(batch_size,)
    )
    # 2. Get input sequences and labels from the dataset based on the start positions sampled
    inputs = np.stack([dataset[index:index+context_length] for index in indices])
    labels = np.stack([dataset[index+1:index+context_length+1] for index in indices])
    
    # 3. Convert numpy arrays to torch tensors and move to specified device
    return (
        torch.from_numpy(inputs).long().to(device),
        torch.from_numpy(labels).long().to(device)
    )