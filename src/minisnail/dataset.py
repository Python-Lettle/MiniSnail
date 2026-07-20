import os
import torch
import random
import numpy as np
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader

def get_dataloader(
    data_path: str,
    block_size: int = 128,
    batch_size: int = 32,
    num_workers: int | None = None,
):
    if num_workers is None:
        num_workers = max(1, (os.cpu_count() or 4) // 2)
    
    dataset = PretrainDataset(data_path, block_size)
    console.print(f"[DataLoader] num_samples: {len(dataset)}, num_workers={num_workers}")
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
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
        
        # Calculate the number of available samples (each sample requires block_size+1 token)
        self.num_samples = len(self.data) - block_size
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Return a sample pair (x, y) based on the index.
        Note: Although the index is passed in here, we actually use a random sampling strategy,
            Consistent with the behavior of np.random.randint in your data-loader function.
            This "random sampling" method is very common in pre-training.
        """
        # Randomly select a starting position (the same as np. random. randint in data-loader)
        start = np.random.randint(0, self.num_samples)
        
        # Take consecutive block_size+1 token
        chunk = self.data[start:start + self.block_size + 1]
        
        # Convert to numpy array (because memmap slices return subviews)
        chunk = np.asarray(chunk, dtype=np.int64)
        
        # Construct x (input) and y (label, offset one bit to the right)
        x = torch.from_numpy(chunk[:-1].copy()).long()  # [block_size]
        y = torch.from_numpy(chunk[1:].copy()).long()   # [block_size]
        
        return x, y

class JSONLDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = load_dataset('json', data_files=data_path, split='train')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        tokens = self.tokenizer(str(sample['text']), add_special_tokens=False, max_length=self.max_length - 2, truncation=True).input_ids
        tokens = [self.tokenizer.bos_token_id] + tokens + [self.tokenizer.eos_token_id]
        input_ids = tokens + [self.tokenizer.pad_token_id] * (self.max_length - len(tokens))
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        labels = input_ids.clone()
        labels[input_ids == self.tokenizer.pad_token_id] = -100
        return input_ids, labels

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