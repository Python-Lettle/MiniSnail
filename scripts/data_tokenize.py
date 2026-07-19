import json
import os
import gc
import argparse
import time
from tqdm import tqdm
import numpy as np
import multiprocessing as mp

_tokenizer = None


def _init_worker(tokenizer_path: str):
    """After fork, each worker loads the tokenizer on init (lazy import to avoid loading transformers in the main process)."""
    global _tokenizer
    abs_path = os.path.abspath(tokenizer_path)
    from transformers import AutoTokenizer
    _tokenizer = AutoTokenizer.from_pretrained(
        abs_path,
        local_files_only=True,
        trust_remote_code=True,
    )
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    print(f"Tokenizer loaded from {abs_path}, with bos_token_id={_tokenizer.bos_token_id}, eos_token_id={_tokenizer.eos_token_id}")


def _encode_chunk(task):
    """Encode a batch of json lines, return (split_tag, chunk_id, np.int32 array)."""
    split_tag, chunk_id, lines = task
    texts = []
    for line in lines:
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            t = data.get("text") or data.get("content") or data.get("s") or ""
            if not t and "prompt" in data and "answer" in data:
                t = data["prompt"] + data["answer"]
            if t:
                texts.append(t)
        elif isinstance(data, str):
            texts.append(data)

    if not texts:
        return (split_tag, chunk_id, np.array([], dtype=np.int32))

    bos_token_id: int = _tokenizer.bos_token_id
    eos_token_id: int = _tokenizer.eos_token_id
    enc = _tokenizer(texts, padding=False, truncation=False, add_special_tokens=False)
    ids = []
    for x in enc["input_ids"]:
        if bos_token_id is not None:
            ids.append(bos_token_id)
        ids.extend(x)
        if eos_token_id is not None:
            ids.append(eos_token_id)
    return (split_tag, chunk_id, np.array(ids, dtype=np.int32))


def main():
    parser = argparse.ArgumentParser(
        description="MiniSnail Parallel Data Preprocessing (fully streaming)"
    )
    parser.add_argument("--tokenizer_path", default="./model/minimind")
    parser.add_argument("--data_path", default="./data/pretrain_t2t_mini.jsonl")
    parser.add_argument("--train_output_path", default="./data/train_dataset.npy")
    parser.add_argument("--valid_output_path", default="./data/valid_dataset.npy")
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--chunk_size", type=int, default=2000)
    parser.add_argument("--num_workers", type=int, default=None)
    args = parser.parse_args()

    if args.num_workers is None:
        args.num_workers = mp.cpu_count()

    print(f"Workers: {args.num_workers} | Chunk size: {args.chunk_size}")
    t0 = time.time()

    # ============================================================================
    # 第一遍扫描：统计总字节数，用于确定训练/验证集的临界分界点
    # ============================================================================
    print("Counting bytes (single pass)...")
    total_bytes = 0
    total_lines = 0
    with open(args.data_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue  # 与第二遍保持一致：空行不计入统计
            total_lines += 1
            total_bytes += len(line.encode("utf-8"))

    train_bytes_target = total_bytes * args.train_ratio
    print(f"Total lines: {total_lines:,}  Total bytes: {total_bytes:,}")
    print(f"Train target (bytes): {train_bytes_target:,.0f}  ({args.train_ratio*100:.0f}%)")

    train_f = open(args.train_output_path, "wb")
    valid_f = open(args.valid_output_path, "wb")
    train_tok = 0
    valid_tok = 0
    buf_train = []
    buf_valid = []
    cid = 0

    def gen_chunks():
        """
        以"行（文档）"为最小不可分割单位进行分割。

        核心策略：
          将整行完整地划入训练集，累积字节数。
          当累积字节达到或超过 train_bytes_target 时，说明已到达临界点。
          由于这整行已经完整归入训练集，从下一行开始全部划入验证集。

        这样做有两个保证：
          ① 任何一行文档不会被截断（不会一半在训练集一半在验证集）
          ② 训练集拿到的是完整的文档，不破坏 text → next token prediction 的完整性
        """
        nonlocal buf_train, buf_valid, cid
        accumulated_bytes = 0
        train_lines_count = 0
        valid_lines_count = 0
        switched = False  # 标记：训练集是否已装满，开始进入验证集

        with open(args.data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                line_bytes = len(line.encode("utf-8"))

                if not switched:
                    # === 训练集：整行完整加入，不拆分 ===
                    buf_train.append(line)
                    accumulated_bytes += line_bytes
                    train_lines_count += 1
                    if len(buf_train) >= args.chunk_size:
                        yield ("train", cid, buf_train)
                        cid += 1
                        buf_train = []

                    # 检查是否达到临界点
                    # 注意：当前行已完整加入 buf_train，after this point 切换
                    if accumulated_bytes >= train_bytes_target:
                        switched = True
                        actual_ratio = accumulated_bytes / total_bytes
                        print(f"\n  [Split Point] 训练集已满，切换至验证集")
                        print(f"    Target bytes: {train_bytes_target:,.0f}  ({args.train_ratio*100:.0f}%)")
                        print(f"    Actual bytes: {accumulated_bytes:,}  ({actual_ratio*100:.2f}%)")
                        print(f"    Train lines : {train_lines_count:,}")
                        print(f"    Overshoot   : +{accumulated_bytes - train_bytes_target:,} bytes "
                              f"(+{(actual_ratio - args.train_ratio)*100:.2f}%)")
                        print(f"    (原因是最后一行文档完整加入后超过了目标值，保证了训练集文档完整性)\n")
                else:
                    # === 验证集 ===
                    buf_valid.append(line)
                    valid_lines_count += 1
                    if len(buf_valid) >= args.chunk_size:
                        yield ("valid", cid, buf_valid)
                        cid += 1
                        buf_valid = []

        # 冲洗余量
        if buf_train:
            yield ("train", cid, buf_train)
            cid += 1
        if buf_valid:
            yield ("valid", cid, buf_valid)
            cid += 1

        actual_train_bytes = accumulated_bytes
        actual_valid_bytes = total_bytes - actual_train_bytes
        print(f"  [Final] Train: {actual_train_bytes:,} bytes ({actual_train_bytes/total_bytes*100:.2f}%) "
              f"| Valid: {actual_valid_bytes:,} bytes ({actual_valid_bytes/total_bytes*100:.2f}%) "
              f"| Total: {total_bytes:,} bytes")

    n_chunks_estimate = total_lines // args.chunk_size + 2
    with mp.Pool(args.num_workers, initializer=_init_worker, initargs=(args.tokenizer_path,)) as pool:
        for split_tag, chunk_id, arr in tqdm(
            pool.imap(_encode_chunk, gen_chunks(), chunksize=1),
            total=n_chunks_estimate,
            desc="Encoding",
        ):
            if split_tag == "train":
                arr.tofile(train_f)
                train_tok += arr.shape[0]
            else:
                arr.tofile(valid_f)
                valid_tok += arr.shape[0]

    train_f.close()
    valid_f.close()
    gc.collect()

    elapsed = time.time() - t0
    total_tok = train_tok + valid_tok
    print(f"\n{'=' * 60}")
    print(f"Train tokens : {train_tok:,}  ({train_tok*4/1024/1024:.2f} MB) -> {args.train_output_path}")
    print(f"Valid tokens : {valid_tok:,}  ({valid_tok*4/1024/1024:.2f} MB) -> {args.valid_output_path}")
    if total_tok > 0:
        print(f"Train ratio  : {train_tok/total_tok:.4f} (target {args.train_ratio})")
    print(f"Total tokens : {total_tok:,}")
    print(f"Total time   : {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"Throughput   : {total_tok/elapsed:,.0f} tokens/sec")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
