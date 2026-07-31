import os
import torch
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader
from minisnail.debug import console

def get_dataloader(
    data_path: str,
    block_size: int = 128,
    batch_size: int = 32,
    num_workers: int | None = None,
    shuffle: bool = True,
    drop_last: bool = True,
):
    if num_workers is None:
        num_workers = max(1, (os.cpu_count() or 4) // 2)

    dataset = PretrainDataset(data_path, block_size)
    console.print(f"[DataLoader] num_samples: {len(dataset)}, num_workers={num_workers}, shuffle={shuffle}, drop_last={drop_last}")

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=drop_last,
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
    )
    return dataloader

# ---------- PretrainDataset ----------

class PretrainDataset(Dataset):
    """
    Pre-training dataset based on one-dimensional token array.
    Randomly sample consecutive segments in __getitem__ to achieve true 'lazy loading'——
    It's not about pre dividing fixed samples, but randomly selecting a segment each time.
    """
    def __init__(self, data_path: str, block_size: int = 128):
        """
        Args:
            data_path: np.int32 1d array
            block_size: context_length
        """
        super().__init__()
        self.block_size = block_size
        
        # Loading in mmap mode will not read the entire file into memory
        self.data = np.memmap(data_path, dtype=np.int32, mode='r')
        
        # Calculate the number of available samples (non-overlapping chunks)
        # Each sample consists of block_size + 1 tokens, and the samples do not overlap with each other.
        self.num_samples = len(self.data) // (block_size + 1)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Return a sample pair (x, y) based on the index.
        Take the index-th non-overlapping block as the index-th block, and select consecutive block_size + 1 tokens.
        During each epoch, each token is accessed exactly once. The shuffling function of DataLoader is responsible for randomizing the order.
        """
        # The starting position of the index-th non-overlapping block
        start = index * (self.block_size + 1)
        chunk = self.data[start:start + self.block_size + 1]

        # Convert to numpy array (because memmap slices return subviews)
        chunk = np.asarray(chunk, dtype=np.int64)

        # Construct x (input) and y (label, offset one bit to the right)
        x = torch.from_numpy(chunk[:-1].copy()).long()  # [block_size]
        y = torch.from_numpy(chunk[1:].copy()).long()   # [block_size]

        return x, y

# ---------- SFTDataset ----------

class SFTDataset(Dataset):
    def __init__(self, input_path, labels_path):
        self.input_ids = np.load(input_path)
        self.labels    = np.load(labels_path)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, index):
        return (torch.tensor(self.input_ids[index], dtype=torch.long),
                torch.tensor(self.labels[index], dtype=torch.long))