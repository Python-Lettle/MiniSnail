"""
取出一条 SFT 预处理产物 (sft_input_ids_XXXX.npy / sft_labels_XXXX.npy) 的样本,
解码并检查是否与 preprocess_sft_data.build_sample 的约定一致:

  1. 形状为 (max_length,), dtype 与 sft_meta.json 记录一致
  2. 序列结构: <|im_start|>user\\n{问}<|im_end|>\\n<|im_start|>assistant\\n{答}<|im_end|>
  3. labels 只在 assistant 段 (含其 <|im_end|>) 非 -100, 其余 (prompt/分隔\\n/padding) 全 -100
  4. padding 区域: input_ids == pad_token_id 且 labels == -100
  5. 有监督 token 数 > 0

用法:
    python scripts/inspect_sft_sample.py                # 第 0 条
    python scripts/inspect_sft_sample.py --index 123    # 指定行
    python scripts/inspect_sft_sample.py --shard 1      # 指定分片
"""
import argparse
import json
import os
import sys

import numpy as np
from transformers import AutoTokenizer

# Windows 控制台默认 GBK, 遇到 'Ċ' 等字符会 UnicodeEncodeError, 改为替换输出
sys.stdout.reconfigure(errors="replace")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="./dataset/full/sft_input_ids_0000.npy")
    parser.add_argument("--labels", default="./dataset/full/sft_labels_0000.npy")
    parser.add_argument("--meta", default="./dataset/full/sft_meta.json")
    parser.add_argument("--tokenizer_root", default="./model/minimind")
    parser.add_argument("--shard", type=int, default=None,
                        help="覆盖 --input/--labels: 用分片序号拼路径 (dataset/full)")
    parser.add_argument("--data_dir", default="./dataset/full")
    parser.add_argument("--index", type=int, default=0, help="取分片内第几条样本")
    args = parser.parse_args()

    if args.shard is not None:
        args.input = os.path.join(args.data_dir, f"sft_input_ids_{args.shard:04d}.npy")
        args.labels = os.path.join(args.data_dir, f"sft_labels_{args.shard:04d}.npy")

    input_ids = np.load(args.input, mmap_mode="r")
    labels = np.load(args.labels, mmap_mode="r")
    print(f"input : {args.input}  shape={input_ids.shape} dtype={input_ids.dtype}")
    print(f"labels: {args.labels}  shape={labels.shape} dtype={labels.dtype}")

    # ---- 检查 1: 形状 / dtype ----
    meta = {}
    if os.path.exists(args.meta):
        with open(args.meta, "r", encoding="utf-8") as f:
            meta = json.load(f)
    max_length = meta.get("max_length")
    assert input_ids.ndim == 2 and labels.shape == input_ids.shape, "形状不一致或不是二维"
    if max_length is not None:
        assert input_ids.shape[1] == max_length, f"序列长度 {input_ids.shape[1]} != meta.max_length {max_length}"
    assert labels.dtype.kind == "i", "labels 必须是有符号整型 (需容纳 -100)"

    if not (0 <= args.index < len(input_ids)):
        raise SystemExit(f"index {args.index} 越界, 分片共 {len(input_ids)} 条")

    row_ids = input_ids[args.index].astype(np.int64)
    row_labs = labels[args.index].astype(np.int64)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_root, local_files_only=True)
    pad_id = tokenizer.pad_token_id
    im_start = tokenizer.convert_ids_to_tokens(1)  # <|im_start|>
    im_end = tokenizer.convert_ids_to_tokens(2)    # <|im_end|>
    # 该 tokenizer 的换行 token 是 'Ċ' (U+010A), 不是字面 '\n', 动态取一次
    NEWLINE_TOK = tokenizer.convert_ids_to_tokens(
        tokenizer("\n", add_special_tokens=False).input_ids[0])

    n_pad = int((row_ids == pad_id).sum())
    supervised = row_labs != -100
    n_sup = int(supervised.sum())

    print(f"\n===== 样本 index={args.index} =====")
    print(f"有效长度 (含 pad 前): {len(row_ids) - n_pad}  |  pad 数: {n_pad}  |  有监督 token 数: {n_sup}")

    # ---- 检查 4: padding 区域对齐 ----
    pad_region = row_ids == pad_id
    assert (row_labs[pad_region] == -100).all(), "padding 区域存在 labels != -100"
    # padding 应连续位于尾部
    first_pad = int(pad_region.argmax()) if pad_region.any() else len(row_ids)
    assert pad_region[first_pad:].all(), "padding 不在序列尾部"
    print("[ok] padding 全在尾部且 labels 全 -100")

    # ---- 检查 2: 整段解码 ----
    valid = row_ids[:first_pad]
    text = tokenizer.decode(valid)
    print(f"\n----- input_ids 解码 (有效 {len(valid)} tokens) -----")
    print(text)

    print("----- 逐 token (前 80 个) -----")
    for i, tid in enumerate(valid[:80]):
        mark = "SUP" if supervised[i] else "   "
        print(f"{i:3d} {mark} {tid:6d} {tokenizer.convert_ids_to_tokens(int(tid))!r}")

    # ---- 检查 2: 模板结构 ----
    assert valid[0] == 1, "序列应以 <|im_start|> (id=1) 开头"
    toks = [tokenizer.convert_ids_to_tokens(int(t)) for t in valid]
    assert im_end in toks, "缺少 <|im_end|>"
    print("[ok] 以 <|im_start|> 开头且含 <|im_end|>")

    # ---- 检查 3: 有监督段应恰为 assistant 正文 + 其 <|im_end|> ----
    sup_ids = row_ids[supervised]
    sup_text = tokenizer.decode(sup_ids)
    print(f"\n----- labels != -100 的 token 解码 (assistant 监督段) -----")
    print(sup_text)
    # 监督段应以内置的 <|im_end|> 结束 (不含消息间的分隔 \n)
    last_sup_tok = tokenizer.convert_ids_to_tokens(int(sup_ids[-1]))
    assert last_sup_tok == im_end, f"监督段最后一个 token 应为 <|im_end|>, 实际 {last_sup_tok!r}"
    # 监督段应以 <|im_start|>assistant\n 开头 (头部保留监督, 沿用 build_sample 约定)
    # 注意: 该 tokenizer 的换行是 'Ċ' (U+010A), 不是字面 '\n'
    sup_toks = [tokenizer.convert_ids_to_tokens(int(t)) for t in sup_ids[:5]]
    assert "".join(sup_toks).startswith(f"{im_start}assistant{NEWLINE_TOK}"), \
        f"监督段应以 <|im_start|>assistant\\n 开头, 实际: {sup_toks!r}"
    # 监督段之前是 prompt + 分隔 \n, 应以 <|im_end|> 结尾
    first_sup = int(np.flatnonzero(supervised)[0])
    assert first_sup >= 2 and toks[first_sup - 1] == NEWLINE_TOK, \
        f"监督段前应有分隔换行 \\n, 实际前一个 token: {toks[first_sup - 1]!r}"
    assert toks[first_sup - 2] == im_end, \
        f"分隔换行前应是 <|im_end|>, 实际: {toks[first_sup - 2]!r}"
    print("[ok] 监督段 = <|im_start|>assistant\\n + assistant 正文 + <|im_end|>, "
          "prompt 侧以 <|im_end|>\\n 分隔")

    # ---- 检查 5 ----
    assert n_sup > 0, "该样本没有任何监督目标"
    print("[ok] 有监督 token 数 > 0")

    print("\n全部检查通过 ✅")


if __name__ == "__main__":
    main()
