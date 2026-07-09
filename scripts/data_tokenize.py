import json
import os
import argparse
import time
from tqdm import tqdm
import numpy as np
from datasets import Dataset
from transformers import AutoTokenizer
from transformers.tokenization_utils_tokenizers import TokenizersBackend

from minisnail.util import console

def encode_txt_nparray(tokenizer: TokenizersBackend, input_path: str | os.PathLike, output_path: str | os.PathLike, is_json: bool = False):
    '''Use the given Tokenizer to encode a text file, and save the encoded token array to a file
    '''
    start_time = time.time()
    
    # 1. Read the file and count the number of lines
    with open(input_path, "r", encoding="utf-8") as f:
        num_lines = sum(1 for _ in f)
    console.print("num_lines:", num_lines)
    
    # 2. The binary append mode opens the output file and writes in a streaming manner.
    total_tokens = 0
    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "wb") as fout:
        
        for line in tqdm(fin, total=num_lines, desc="Encoding"):
            line = line.strip()
            if not line:
                continue
            if is_json:
                line = json.loads(line)["text"]
            
            # 只 encode 一次
            token = tokenizer.encode(line)
            arr = np.array(token, dtype=np.int32)
            # 立即写入磁盘，不持有任何历史数据
            fout.write(arr.tobytes())
            total_tokens += len(token)
    
    console.print("total_tokens:", total_tokens)
    console.print(f"Binary saved to: {output_path}")
    console.print(f"File size: {total_tokens * 4 / 1024 / 1024:.2f} MB")
    console.print(f"Encoding time: {time.time() - start_time:.2f} seconds")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="MiniSnail Data Preprocessing")
    parser.add_argument("--tokenizer_path", type=str, default="./model/minimind", help="Tokenizer path")
    parser.add_argument("--data_path", type=str, default="./data/testdata.txt", help="Raw data path")
    parser.add_argument("--output_path", type=str, default="./data/train_dataset.npy", help="Path to output encoded file")
    parser.add_argument("--is_json", type=bool, default=True, help="Whether the input file is in JSON format")
    args = parser.parse_args()

    tokenizer: TokenizersBackend = AutoTokenizer.from_pretrained(args.tokenizer_path)
    
    encode_txt_nparray(tokenizer, args.data_path, args.output_path, is_json=args.is_json)
    