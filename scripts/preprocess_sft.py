import os
import json
import argparse
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer
import re
pattern = re.compile(r'<think>.*?</think>', re.S)

REPLACE_RULES = [
    (
        [
            "jingyaogong",
            "Jingyao Gong",
            "京遥",
            "作者"
        ],
        "Lettle"
    ),
    (
        [
            "MiniMind",
            "minimind",
            "MiniMind模型"
        ],
        "MiniSnail"
    ),
    (
        [
            "MiniMind项目",
            "MiniMind 项目"
        ],
        "MiniSnail项目"
    ),
]

FILTER_WORDS = [
    "github.com/jingyaogong",
    "https://github.com/jingyaogong",
    "QQ群",
    "微信群"
]

def remove_think(text):
    text = re.sub(
        pattern,
        "",
        text
    )
    return text.strip()

def clean_content(text):
    if not isinstance(text,str):
        return text
    # ----------过滤----------
    for word in FILTER_WORDS:
        if word.lower() in text.lower():
            return None
    # ----------替换----------
    for patterns, replacement in REPLACE_RULES:
        for pattern in patterns:
            text = re.sub(
                pattern,
                replacement,
                text,
                flags=re.I
            )

    # ----------去think----------
    text = remove_think(text)
    return text.strip()

def build_sample(messages, tokenizer, max_length):
    input_ids = []
    labels = []

    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        # 单轮chat template
        text = tokenizer.apply_chat_template(
            [
                {
                    "role": role,
                    "content": content
                }
            ],
            tokenize=False,
            add_generation_prompt=False,
        )
        text = clean_content(text)
        # 命中过滤词则跳过整个样本，避免破坏多轮对话结构
        if text is None:
            return None, None

        ids = tokenizer(text,add_special_tokens=False).input_ids
        input_ids.extend(ids)

        if role == "assistant":
            labels.extend(ids)
        else:
            labels.extend([-100] * len(ids))
    # 截断
    input_ids = input_ids[:max_length]
    labels = labels[:max_length]
    # padding
    pad_len = max_length - len(input_ids)
    input_ids += [tokenizer.pad_token_id] * pad_len
    labels += [-100] * pad_len

    return input_ids, labels



def process(data_path,tokenizer, max_length, output_dir, num_samples=None):

    os.makedirs(output_dir, exist_ok=True)
    dataset = load_dataset("json",data_files=data_path,split="train")

    if num_samples:
        dataset = dataset.select(range(num_samples))

    inputs=[]
    labels=[]

    skipped = 0
    for item in tqdm(dataset):
        messages = item["conversations"]
        ids, labs = build_sample(messages, tokenizer, max_length)
        # 命中过滤词的样本跳过
        if ids is None:
            skipped += 1
            continue
        inputs.append(ids)
        labels.append(labs)

    print(f"Skipped {skipped} samples by FILTER_WORDS")

    np.save(os.path.join(output_dir,"sft_input_ids.npy"), np.array(inputs, dtype=np.int32))
    np.save(os.path.join(output_dir,"sft_labels.npy"), np.array(labels,dtype=np.int32))

    print("Pre-process SFT dataset done!")

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--data_path", default="./data/sft_t2t_mini.jsonl")
    parser.add_argument("--tokenizer_root", default="./model/minimind")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--output_dir", default="./data")
    parser.add_argument("--num_samples", type=int, default=None)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_root, local_files_only=True, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    process(args.data_path, tokenizer, args.max_length, args.output_dir, num_samples=args.num_samples)
