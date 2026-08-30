import os
import sys
import glob
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

def get_epoch_dataloader(
    dataset: Dataset,
    batch_size: int = 32,
    seed: int = 42,
    epoch: int = 0,
    skip_batches: int = 0,
    pin_memory: bool = True,
):
    """
    构建指定 epoch 的确定性洗牌 dataloader (断点续训用)。
    每个 epoch 用独立 seed (seed * 1000003 + epoch) 生成固定排列, 同一 (seed, epoch)
    在重启/续训时得到完全一致的 batch 顺序, 因此可按 skip_batches 精确跳过已完成的 batch,
    既不重复训练也不漏数据。不丢弃尾部不满 batch (drop_last=False, 与预训练 train loader 一致)。
    """
    if sys.platform == 'win32':
        num_workers = 0
    else:
        num_workers = max(1, (os.cpu_count() or 4) // 2)

    g = torch.Generator()
    g.manual_seed(seed * 1000003 + epoch)
    perm = torch.randperm(len(dataset), generator=g).tolist()
    batches = [perm[i:i + batch_size] for i in range(0, len(perm), batch_size)]
    if skip_batches > 0:
        batches = batches[skip_batches:]
    console.print(f"[DataLoader] epoch={epoch}, num_samples={len(dataset)}, "
                  f"skip_batches={skip_batches}, remaining_batches={len(batches)}, num_workers={num_workers}")
    return DataLoader(
        dataset,
        batch_sampler=batches,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

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
                 indices: np.ndarray | None = None):
        """
        Args:
            data_path: jsonl 文件路径 (整份文件, 通过 indices 行号数组划分子集)
            tokenizer: 分词器
            max_length: 样本最大长度 (context_length)
            indices: 本子集包含的行号数组 (int64); None 表示全部行。
                     用于在同一份文件上划分训练集/验证集 (如固定 seed 的 permutation 切分)
        """
        super().__init__()
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.offsets = load_line_offsets(data_path)
        num_lines = len(self.offsets)
        self.indices = np.arange(num_lines, dtype=np.int64) if indices is None else np.asarray(indices, dtype=np.int64)
        # 文件句柄按进程持有 (DataLoader 多 worker 时每个进程独立打开)
        self._fh = None
        self._fh_pid = None

    def __len__(self) -> int:
        return len(self.indices)

    def _get_fh(self):
        if self._fh is None or self._fh_pid != os.getpid():
            self._fh = open(self.data_path, 'rb')
            self._fh_pid = os.getpid()
        return self._fh

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        fh = self._get_fh()
        line_no = int(self.indices[index])
        fh.seek(int(self.offsets[line_no]))
        line = fh.readline()
        try:
            text: str = json.loads(line)['text']
        except (json.JSONDecodeError, KeyError) as e:
            raise ValueError(
                f"{self.data_path} 第 {line_no} 行解析失败: {e!r}") from e
        return encode_text(self.tokenizer, text, self.max_length)


# ---------- SFT ----------
class SFTDataset(Dataset):
    """
    读取预处理好的定长 npy: sft_input_ids*.npy / sft_labels*.npy

    两种构造方式:
        SFTDataset(data_dir="dataset/full")                  # 目录下所有分片, 按文件名排序
        SFTDataset(input_path="a.npy", labels_path="b.npy")  # 显式指定单对文件

    关于 dtype:
        npy 以 int16 落盘 (词表 6400, labels 的 -100 也落在 int16 范围内, 比 int32 省一半磁盘);
        但 nn.Embedding 的索引与 F.cross_entropy 的 target 都只接受 Long/Int, int16 (short) 会直接
        抛 RuntimeError, 所以 __getitem__ 出栈时必须转成 int64。dtype 参数只描述"落盘类型",
        不影响返回张量的类型。
    """
    def __init__(self, input_path=None, labels_path=None, data_dir=None,
                 dtype: torch.dtype = torch.int16, mmap: bool = True):
        super().__init__()

        if data_dir is not None:
            id_files = sorted(glob.glob(os.path.join(data_dir, "sft_input_ids*.npy")))
            lb_files = sorted(glob.glob(os.path.join(data_dir, "sft_labels*.npy")))
        elif input_path is not None and labels_path is not None:
            id_files, lb_files = [input_path], [labels_path]
        else:
            raise ValueError("SFTDataset 需要 data_dir, 或同时给出 input_path 与 labels_path")

        if not id_files or not lb_files:
            raise FileNotFoundError(
                f"没有找到 sft_input_ids*.npy / sft_labels*.npy "
                f"(data_dir={data_dir}, input_path={input_path})")
        if len(id_files) != len(lb_files):
            raise ValueError(
                f"input_ids 分片 {len(id_files)} 个与 labels 分片 {len(lb_files)} 个不匹配")

        # 只读 mmap: 训练侧不写回, 'r+' 反而要求文件可写且有误写风险
        mode = "r" if mmap else None
        self.input_ids = [np.load(p, mmap_mode=mode) for p in id_files]
        self.labels = [np.load(p, mmap_mode=mode) for p in lb_files]

        for i, (a, b) in enumerate(zip(self.input_ids, self.labels)):
            if a.shape != b.shape:
                raise ValueError(f"分片 {i} 形状不一致: {a.shape} vs {b.shape}")
            if a.ndim != 2:
                raise ValueError(f"分片 {i} 应为二维 (样本数, 序列长度), 实际 {a.shape}")
            # labels 需要容纳 -100, 无符号类型会在落盘时静默溢出成大正数
            if b.dtype.kind != "i":
                raise ValueError(
                    f"分片 {i} 的 labels dtype 为 {b.dtype}, 必须是有符号整型 (需容纳 -100)")

        self.shard_files = list(zip(id_files, lb_files))
        self.shard_sizes = [len(a) for a in self.input_ids]
        # cum[i] = 前 i 个分片的样本总数, 用于把全局下标定位到分片
        self.cum = np.cumsum([0] + self.shard_sizes)
        self.total = int(self.cum[-1])
        self.max_length = int(self.input_ids[0].shape[1])
        self.dtype = dtype

    def __len__(self):
        return self.total

    def __getitem__(self, index):
        if index < 0:
            index += self.total
        shard = int(np.searchsorted(self.cum, index, side="right")) - 1
        offset = index - int(self.cum[shard])
        # astype 会把 memmap 行拷成常规 ndarray, 再交给 from_numpy 零拷贝包装
        ids = np.asarray(self.input_ids[shard][offset]).astype(np.int64, copy=False)
        labels = np.asarray(self.labels[shard][offset]).astype(np.int64, copy=False)
        return torch.from_numpy(ids), torch.from_numpy(labels)



