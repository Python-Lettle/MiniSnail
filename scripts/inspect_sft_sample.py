"""Inspect one preprocessed SFT row against minisnail-chat-v1 invariants."""

import argparse
import json
import os
import sys

import numpy as np
from transformers import AutoTokenizer

from minisnail.chat_protocol import CHAT_PROTOCOL_VERSION, encode_message_parts


try:
    sys.stdout.reconfigure(errors="replace")
except AttributeError:
    pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="./dataset/full/sft_input_ids_0000.npy")
    parser.add_argument("--labels", default="./dataset/full/sft_labels_0000.npy")
    parser.add_argument("--meta", default="./dataset/full/sft_meta.json")
    parser.add_argument("--tokenizer_root", default="./model/minimind")
    parser.add_argument("--shard", type=int, default=None)
    parser.add_argument("--data_dir", default="./dataset/full")
    parser.add_argument("--index", type=int, default=0)
    args = parser.parse_args()

    if args.shard is not None:
        args.input = os.path.join(args.data_dir, f"sft_input_ids_{args.shard:04d}.npy")
        args.labels = os.path.join(args.data_dir, f"sft_labels_{args.shard:04d}.npy")

    input_ids = np.load(args.input, mmap_mode="r")
    labels = np.load(args.labels, mmap_mode="r")
    print(f"input : {args.input}  shape={input_ids.shape} dtype={input_ids.dtype}")
    print(f"labels: {args.labels}  shape={labels.shape} dtype={labels.dtype}")

    if not os.path.exists(args.meta):
        raise AssertionError("缺少 sft_meta.json，无法确认对话协议")
    with open(args.meta, "r", encoding="utf-8") as file:
        meta = json.load(file)
    assert meta.get("chat_protocol") == CHAT_PROTOCOL_VERSION, (
        f"旧协议分片 {meta.get('chat_protocol')!r}; 当前要求 {CHAT_PROTOCOL_VERSION!r}"
    )
    assert input_ids.ndim == 2 and labels.shape == input_ids.shape
    assert input_ids.shape[1] == meta.get("max_length")
    assert labels.dtype.kind == "i", "labels 必须是可容纳 -100 的有符号整数"
    if not 0 <= args.index < len(input_ids):
        raise SystemExit(f"index {args.index} 越界, 分片共 {len(input_ids)} 条")

    row_ids = input_ids[args.index].astype(np.int64)
    row_labels = labels[args.index].astype(np.int64)
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_root, local_files_only=True
    )
    pad_id = tokenizer.pad_token_id
    im_start = tokenizer.convert_ids_to_tokens(tokenizer.bos_token_id)
    im_end = tokenizer.convert_ids_to_tokens(tokenizer.eos_token_id)
    supervised = row_labels != -100

    pad_region = row_ids == pad_id
    first_pad = int(pad_region.argmax()) if pad_region.any() else len(row_ids)
    assert pad_region[first_pad:].all(), "padding 必须连续位于序列尾部"
    assert (row_labels[pad_region] == -100).all(), "padding 位置不能参与监督"
    assert supervised.any(), "样本没有 assistant 监督目标"

    valid = row_ids[:first_pad]
    tokens = [tokenizer.convert_ids_to_tokens(int(token)) for token in valid]
    assert tokens[0] == im_start and im_end in tokens

    assistant_header, _, _, _ = encode_message_parts(
        tokenizer, {"role": "assistant", "content": ""}
    )
    starts = np.flatnonzero(supervised & ~np.r_[False, supervised[:-1]])
    ends = np.flatnonzero(supervised & ~np.r_[supervised[1:], False])
    assert len(starts) == len(ends) > 0
    for start, end in zip(starts, ends):
        assert start >= len(assistant_header)
        assert row_ids[start - len(assistant_header):start].tolist() == assistant_header, (
            "监督正文之前不是标准 assistant 头部"
        )
        assert tokenizer.convert_ids_to_tokens(int(row_ids[end])) == im_end, (
            "每个 assistant 监督段必须以 <|im_end|> 结束"
        )
    supervised_tokens = [
        tokenizer.convert_ids_to_tokens(int(token)) for token in row_ids[supervised]
    ]
    assert im_start not in supervised_tokens, "assistant 角色头不应参与监督"

    print(f"\n===== 样本 index={args.index} =====")
    print(
        f"有效长度: {len(valid)} | pad: {len(row_ids) - first_pad} | "
        f"监督 tokens: {int(supervised.sum())}"
    )
    print("\n----- input_ids 解码 -----")
    print(tokenizer.decode(valid))
    print("\n----- labels != -100 解码 -----")
    print(tokenizer.decode(row_ids[supervised]))
    print("\n[ok] minisnail-chat-v1 协议、padding 与监督边界全部通过")


if __name__ == "__main__":
    main()
