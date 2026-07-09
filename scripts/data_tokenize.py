import json
import os
import argparse
from tqdm import tqdm
import numpy as np
from datasets import Dataset
from transformers import AutoTokenizer
from transformers.tokenization_utils_tokenizers import TokenizersBackend

from minisnail.util import console

def encode_txt_nparray(tokenizer: TokenizersBackend, input_path: str | os.PathLike, output_path: str | os.PathLike):
    '''Use the given Tokenizer to encode a text file, and save the encoded token array to a file
    '''
    # 1. Read the file and count the number of lines
    with open(input_path, "r", encoding="utf-8") as f:
        num_lines = sum(1 for _ in f)
    console.print("num_lines:", num_lines)
    
    # 2. Encode the lines
    total_tokens = 0
    tokens = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, total=num_lines, desc="Encoding lines"):
            token = tokenizer.encode(line)
            tokens.extend(token)
            total_tokens += len(token)
    console.print("total_tokens:", total_tokens)
    
    # 3. Create the memmap file to be saved
    tokens_mm = np.memmap(output_path, dtype=np.int32, mode='w+', shape=(total_tokens,))

    # 4. Write the tokens to the memmap file
    tokens_mm[:total_tokens] = np.array(tokens, dtype=np.int32)
    tokens_mm.flush()
    console.print("Tokens array saved to:", output_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="MiniSnail Data Preprocessing")
    parser.add_argument("--tokenizer_path", type=str, default="./model/minimind", help="Tokenizer path")
    parser.add_argument("--data_path", type=str, default="./data/testdata.txt", help="Raw data path")
    parser.add_argument("--output_path", type=str, default="./data/train_dataset.npy", help="Path to output encoded file")
    args = parser.parse_args()

    tokenizer: TokenizersBackend = AutoTokenizer.from_pretrained(args.tokenizer_path)
    
    encode_txt_nparray(tokenizer, args.data_path, args.output_path)
    