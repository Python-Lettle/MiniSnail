import argparse
from transformers import AutoTokenizer

from minisnail.util import read_memmap_data
from minisnail.util import console

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer_path", type=str, default="./model/minimind")
    args = parser.parse_args()

    train_data = read_memmap_data("./data/train_dataset.npy")
    console.print(train_data)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

    raw_data = tokenizer.decode(train_data[:100])
    console.print(raw_data)