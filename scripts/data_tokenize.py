import json
import os
import random
import argparse
import time
from tqdm import tqdm
import numpy as np
from datasets import Dataset
from transformers import AutoTokenizer
from transformers.tokenization_utils_tokenizers import TokenizersBackend

from minisnail.util import console, read_memmap_data

def encode_txt_nparray(tokenizer: TokenizersBackend, input_path, train_output_path, valid_output_path, train_ratio=0.8, is_json=False):
    start_time = time.time()
    
    # Calculate total lines in the file
    with open(input_path, "r", encoding="utf-8") as f:
        total_lines = sum(1 for line in f if line.strip())
    console.print(f"total_lines: {total_lines}")

    # Split point
    train_end = int(total_lines * train_ratio)
    console.print(f"train_lines: {train_end}, valid_lines: {total_lines - train_end}")
    
    # Generate line indices and shuffle them
    line_indices = list(range(total_lines))
    random.shuffle(line_indices)

    train_indices = set(line_indices[:train_end])
    valid_indices = set(line_indices[train_end:])

    # Calculate tokens in train and valid sets
    train_tokens = 0
    valid_tokens = 0

    with open(input_path, "r", encoding="utf-8") as fin:
        for line_num, line in enumerate(tqdm(fin, total=total_lines, desc="Counting tokens")):
            line = line.strip()
            if not line:
                continue
            if is_json:
                try:
                    data = json.loads(line)
                    text = data.get("text", "")
                except json.JSONDecodeError as e:
                    console.print(f"Skipping invalid JSON at line {line_num}: {e}")
                    continue
            else:
                text = line
            
            tokens = tokenizer.encode(text)
            if line_num in train_indices:
                train_tokens += len(tokens)
            elif line_num in valid_indices:
                valid_tokens += len(tokens)
    
    console.print(f"train_tokens: {train_tokens}, valid_tokens: {valid_tokens}")

    # Create mmap files
    train_arr = np.memmap(train_output_path, mode="w+", dtype=np.int32, shape=(train_tokens,))
    valid_arr = np.memmap(valid_output_path, mode="w+", dtype=np.int32, shape=(valid_tokens,))

    # Write
    train_idx = 0
    valid_idx = 0

    with open(input_path, "r", encoding="utf-8") as fin:
        for line_num, line in enumerate(tqdm(fin, total=total_lines, desc="Encoding")):
            line = line.strip()
            if not line:
                continue
            if is_json:
                try:
                    data = json.loads(line)
                    text = data.get("text", "")
                except json.JSONDecodeError:
                    continue
            else:
                text = line
            
            tokens = tokenizer.encode(text)
            n = len(tokens)
            
            if line_num in train_indices:
                train_arr[train_idx:train_idx + n] = tokens
                train_idx += n
            elif line_num in valid_indices:
                valid_arr[valid_idx:valid_idx + n] = tokens
                valid_idx += n
    
    # Flush the mmap files
    train_arr.flush()
    valid_arr.flush()

    console.print(f"Train set saved to: {train_output_path}, size: {train_tokens * 4 / 1024 / 1024:.2f} MB")
    console.print(f"Valid set saved to: {valid_output_path}, size: {valid_tokens * 4 / 1024 / 1024:.2f} MB")
    console.print(f"Encoding time: {time.time() - start_time:.2f} seconds")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="MiniSnail Data Preprocessing")
    parser.add_argument("--tokenizer_path", type=str, default="./model/minimind", help="Tokenizer path")
    parser.add_argument("--data_path", type=str, default="./data/testdata.txt", help="Raw data path")
    parser.add_argument("--train_output_path", type=str, default="./data/train_dataset.npy", help="Path to train file")
    parser.add_argument("--valid_output_path", type=str, default="./data/valid_dataset.npy", help="Path to valid file")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Ratio of train data")
    parser.add_argument("--is_json", type=bool, default=True, help="Whether the input file is in JSON format")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling")
    args = parser.parse_args()

    tokenizer: TokenizersBackend = AutoTokenizer.from_pretrained(args.tokenizer_path)
    
    random.seed(args.seed)
    np.random.seed(args.seed)

    encode_txt_nparray(tokenizer, args.data_path, args.train_output_path, args.valid_output_path, train_ratio=args.train_ratio, is_json=args.is_json)
    