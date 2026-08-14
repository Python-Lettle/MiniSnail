import numpy as np

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('./model/minimind')

# 读取 input_ids.npy
input_ids = np.load('./data/sft_input_ids.npy', mmap_mode='r')

# 读取 sft_labels.npy
labels = np.load('./data/sft_labels.npy', mmap_mode='r')

print("input_ids shape:", input_ids.shape)
print("labels shape:", labels.shape)

# labels 中 -100 是 loss 的 ignore_index（mask 掉用户输入部分），
# tokenizer 无法解码负数，需替换为 pad_token_id
pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

def decode_labels(row):
    row = np.where(row == -100, pad_id, row)
    return tokenizer.decode(row, skip_special_tokens=False)

print('='*50)
print(tokenizer.decode(input_ids[0]))
print('-'*50)
print(decode_labels(labels[0]))