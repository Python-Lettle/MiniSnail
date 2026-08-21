import os
import sys
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizer

from minisnail.debug import console

def get_dataloader(
    dataset: Dataset,
    batch_size: int = 32,
    num_workers: int | None = None,
    shuffle: bool = True,
    pin_memory: bool = True,
    drop_last: bool = True,
):
    # 设置 num_workers:
    # Windows 平台不使用多线程, 设置为 0
    # 其他平台根据 CPU 核心数设置多线程
    if sys.platform == 'win32':
        num_workers = 0
    elif num_workers is None:
        num_workers = max(1, (os.cpu_count() or 4) // 2)

    # 加载 DataLoader
    console.print(f"[DataLoader] num_samples: {len(dataset)}, num_workers={num_workers}, shuffle={shuffle}, drop_last={drop_last}")
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
    )
    return dataloader

# ---------- PretrainDataset ----------
class PretrainDataset(Dataset):
    '''
    预训练数据集
    按文档切分, 每个文档为一个样本, 样本长度为 max_length
    '''
    def __init__(self, data_path: str, tokenizer: PreTrainedTokenizer, max_length: int = 512):
        """
        Args:
            data_path: np.int32 1d array
            block_size: context_length
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # 将 jsonl 数据加载到内存中
        self.samples = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.samples.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        根据 index 获取样本, 每个样本为一个文档
        取文档的前 max_length 个 token, 少则用 pad token 填充, 多则截断
        首部添加 bos token, 尾部添加 eos token
        """
        # 获取样本
        sample: dict = self.samples[index]
        text: str = sample['text']
        
        tokens: list = self.tokenizer(
            text, add_special_tokens=False, max_length=self.max_length - 2, truncation=True)['input_ids']
        tokens = [self.tokenizer.bos_token_id] + tokens + [self.tokenizer.eos_token_id]
        input_ids: list = tokens + [self.tokenizer.pad_token_id] * (self.max_length - len(tokens))
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        labels = input_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        return input_ids, labels

