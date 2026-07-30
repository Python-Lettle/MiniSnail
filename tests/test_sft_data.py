import numpy as np

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('./model/minimind')

# 读取 input_ids.npy
input_ids = np.load('./data/sft_input_ids.npy', mmap_mode='r')
print(input_ids.shape)
# 输出 3 条对话
print('='*50)
print(tokenizer.decode(input_ids[0]))
print('='*50)
print(tokenizer.decode(input_ids[1]))
print('='*50)
print(tokenizer.decode(input_ids[2]))
