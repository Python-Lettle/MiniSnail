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

    # First pass to count total lines (pure IO, used to determine train/valid split point)
    print("Counting lines (single pass)...")
    total = 0
    with open(args.data_path, "r", encoding="utf-8") as f:
        for _ in f:
            total += 1
    train_end = int(total * args.train_ratio)
    print(f"Total lines: {total:,}  Train end (line#): {train_end:,}")

    train_f = open(args.train_output_path, "wb")
    valid_f = open(args.valid_output_path, "wb")
    train_tok = 0
    valid_tok = 0

    buf_train = []
    buf_valid = []
    cid = 0

    def gen_chunks():
        """Route to train/valid by line number, yield when a chunk is full (without pre-reading all lines)."""
        nonlocal buf_train, buf_valid, cid
        line_no = 0
        with open(args.data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                line_no += 1
                if line_no <= train_end:
                    buf_train.append(line)
                    if len(buf_train) >= args.chunk_size:
                        yield ("train", cid, buf_train)
                        cid += 1
                        buf_train = []   # Rebind; the old list is held by the worker, safe
                else:
                    buf_valid.append(line)
                    if len(buf_valid) >= args.chunk_size:
                        yield ("valid", cid, buf_valid)
                        cid += 1
                        buf_valid = []
        if buf_train:
            yield ("train", cid, buf_train)
            cid += 1
        if buf_valid:
            yield ("valid", cid, buf_valid)
            cid += 1

    n_chunks = (train_end // args.chunk_size + 1) + ((total - train_end) // args.chunk_size + 1)

    with mp.Pool(args.num_workers, initializer=_init_worker, initargs=(args.tokenizer_path,)) as pool:
        for split_tag, chunk_id, arr in tqdm(
            pool.imap(_encode_chunk, gen_chunks(), chunksize=1),
            total=n_chunks,
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
    print(f"Total tokens : {total_tok:,}")
    print(f"Total time   : {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"Throughput   : {total_tok/elapsed:,.0f} tokens/sec")
    print(f"{'=' * 60}")

if __name__ == "__main__":
    main()