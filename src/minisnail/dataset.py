import os
import sys
import json
import numpy as np
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

# ---------- 编码工具 ----------
def encode_text(tokenizer: PreTrainedTokenizer, text: str, max_length: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    将一段文本编码为 (input_ids, labels)
    取文档的前 max_length 个 token, 少则用 pad token 填充, 多则截断
    首部添加 bos token, 尾部添加 eos token, pad 位置的 label 置为 -100
    """
    tokens: list = tokenizer(
        text, add_special_tokens=False, max_length=max_length - 2, truncation=True)['input_ids']
    tokens = [tokenizer.bos_token_id] + tokens + [tokenizer.eos_token_id]
    input_ids: list = tokens + [tokenizer.pad_token_id] * (max_length - len(tokens))
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    labels = input_ids.clone()
    labels[labels == tokenizer.pad_token_id] = -100
    return input_ids, labels

# ---------- 行偏移索引 (懒加载用) ----------
def build_line_offsets(data_path: str) -> np.ndarray:
    """扫描 jsonl 文件, 返回每行的起始字节偏移 (int64 数组, len = 行数)"""
    offsets: list[int] = []
    pos = 0
    with open(data_path, 'rb') as f:
        for line in f:
            offsets.append(pos)
            pos += len(line)
    return np.array(offsets, dtype=np.int64)

def load_line_offsets(data_path: str) -> np.ndarray:
    """
    加载行偏移索引, 带缓存 (data_path + '.idx.npz')
    缓存按文件大小 + mtime 校验, 数据文件变化后自动重建
    """
    cache_path = data_path + '.idx.npz'
    st = os.stat(data_path)
    if os.path.exists(cache_path):
        cached = np.load(cache_path)
        if int(cached['size']) == st.st_size and float(cached['mtime']) == st.st_mtime:
            return cached['offsets']
    offsets = build_line_offsets(data_path)
    np.savez(cache_path, size=st.st_size, mtime=st.st_mtime, offsets=offsets)
    return offsets

# ---------- PretrainDataset ----------
class PretrainDataset(Dataset):
    '''
    预训练数据集 (全量驻留内存版)
    按文档切分, 每个文档为一个样本, 样本长度为 max_length
    适合小数据集; 大数据集请使用 LazyPretrainDataset
    '''
    def __init__(self, samples: list[dict], tokenizer: PreTrainedTokenizer, max_length: int = 512):
        """
        Args:
            samples: 已经划分好的 jsonl 数据样本 (list[{"text": ...}])
            tokenizer: 分词器
            max_length: 样本最大长度 (context_length)
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length

        # 已经提前划分好是训练集还是验证集的 jsonl 数据样本
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        # 获取样本
        sample: dict = self.samples[index]
        text: str = sample['text']
        return encode_text(self.tokenizer, text, self.max_length)

# ---------- LazyPretrainDataset ----------
class LazyPretrainDataset(Dataset):
    '''
    预训练数据集 (懒加载版)
    内存中只保存每行的字节偏移 (每行 8 字节), __getitem__ 时才按需读取该行并分词
    适合全量数据放不进内存的大数据集; 数据格式与 PretrainDataset 相同 (每行一个 {"text": ...})
    '''
    def __init__(self, data_path: str, tokenizer: PreTrainedTokenizer, max_length: int = 512,
                 start: int = 0, end: int | None = None):
        """
        Args:
            data_path: jsonl 文件路径 (整份文件, 通过 start/end 切片划分子集)
            tokenizer: 分词器
            max_length: 样本最大长度 (context_length)
            start/end: 行号区间 [start, end), 用于在同一份文件上划分训练集/验证集
        """
        super().__init__()
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.offsets = load_line_offsets(data_path)
        num_lines = len(self.offsets)
        self.start = start
        self.end = num_lines if end is None else min(end, num_lines)
        # 文件句柄按进程持有 (DataLoader 多 worker 时每个进程独立打开)
        self._fh = None
        self._fh_pid = None

    def __len__(self) -> int:
        return self.end - self.start

    def _get_fh(self):
        if self._fh is None or self._fh_pid != os.getpid():
            self._fh = open(self.data_path, 'rb')
            self._fh_pid = os.getpid()
        return self._fh

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        fh = self._get_fh()
        fh.seek(int(self.offsets[self.start + index]))
        line = fh.readline()
        try:
            text: str = json.loads(line)['text']
        except (json.JSONDecodeError, KeyError) as e:
            raise ValueError(
                f"{self.data_path} 第 {self.start + index} 行解析失败: {e!r}") from e
        return encode_text(self.tokenizer, text, self.max_length)

