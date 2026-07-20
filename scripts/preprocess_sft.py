"""
SFT 数据预处理：将 jsonl 对话数据批量 tokenize，保存为 .npy 文件。
输出：
    ./data/sft_input_ids.npy   → shape: [num_samples, max_length]，内容为 input_ids
    ./data/sft_labels.npy      → shape: [num_samples, max_length]，内容为 labels（-100 mask）
"""

import os
import json
import argparse
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer
from datasets import load_dataset
from torch.utils.data import Dataset


# 复用 SFTDataset 中的预处理逻辑
import random


def extract_tools(conversations):
    """
    从对话中提取 tools 参数（与 SFTDataset.create_chat_prompt 逻辑一致）。
    如果第一条非 system 对话有 function 字段，提取为 tools。
    """
    for i, turn in enumerate(conversations):
        if turn.get("role") != "system":
            if turn.get("function"):
                return turn["function"]
            break
    return None


def pre_processing_chat(conversations, add_system_ratio=0.2):
    """概率性添加 system prompt。带有 tools 的对话不处理。"""
    SYSTEM_PROMPTS = [
        "你是一个知识丰富的AI，尽力为用户提供准确的信息。",
        "你是minisnail，一个小巧但有用的语言模型。",
        "你是一个专业的AI助手，请提供有价值的回答。",
        "你是minisnail，请尽力帮助用户解决问题。",
        "你是一个可靠的AI，请给出准确的回答。",
        "You are a helpful AI assistant.",
        "You are minisnail, a lightweight intelligent assistant.",
        "You are a friendly chatbot. Please answer the user's questions carefully.",
        "You are a knowledgeable AI. Try your best to provide accurate information.",
        "You are minisnail, a small but useful language model.",
    ]
    # 工具调用对话不添加 system prompt
    if any(conv.get('tools') for conv in conversations):
        return conversations
    if conversations[0].get('role') != 'system':
        if random.random() < add_system_ratio:
            return [{'role': 'system', 'content': random.choice(SYSTEM_PROMPTS)}] + conversations
    return conversations


def post_processing_chat(prompt_content, empty_think_ratio=0.2):
    """以 80% 概率移除空 <think> 标签"""
    if '<think>\n\n</think>\n\n' in prompt_content and random.random() > empty_think_ratio:
        prompt_content = prompt_content.replace('<think>\n\n</think>\n\n', '')
    return prompt_content


def _clean_turn(turn):
    """
    清洗对话中的每一轮，保留全部训练信号。
    核心策略：
    - role / function / name / tool_calls：直接保留
    - reasoning_content（DeepSeek 思考链）：合并到 content 中，不丢弃
    - 其他未知字段：丢弃

    MiniMind 的 ChatML 模板不识别 reasoning_content，
    但 CoT 思考链有训练价值，拼入 content 后模板能正常渲染。
    """
    # 合并 reasoning_content → content
    content = turn.get("content") or ""
    reasoning = turn.get("reasoning_content")
    if reasoning:
        content = f"{reasoning}\n\n{content}"

    result = {"role": turn["role"], "content": content}
    # 工具调用相关字段直接保留
    for key in ("function", "name", "tool_calls"):
        if key in turn:
            result[key] = turn[key]
    return result


def _render_safely(tokenizer, messages, tools, fallback_conversation):
    """
    安全渲染 apply_chat_template。
    主尝试：保留原始结构（含 function/tool_calls）+ 传 tools 参数
    若失败则降级：清理为 pure dialogue（只保留 role/content，不传 tools）
    """
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False, tools=tools
        )
    except Exception:
        # 降级：只留 role/content，不传 tools
        safe_msgs = [
            {"role": turn["role"], "content": turn.get("content") or ""}
            for turn in fallback_conversation
        ]
        return tokenizer.apply_chat_template(
            safe_msgs, tokenize=False, add_generation_prompt=False
        )


def process(jsonl_path, tokenizer, max_length, output_dir, num_samples=None):
    os.makedirs(output_dir, exist_ok=True)

    # 加载数据
    dataset = load_dataset("json", data_files=jsonl_path, split="train")
    if num_samples:
        dataset = dataset.select(range(num_samples))

    total = len(dataset)
    print(f"总样本数: {total}")

    # bos_id / eos_id（和 SFTDataset 一致）
    bos_id = tokenizer(f"{tokenizer.bos_token}assistant\n", add_special_tokens=False).input_ids
    eos_id = tokenizer(f"{tokenizer.eos_token}\n", add_special_tokens=False).input_ids

    def generate_labels(input_ids):
        labels = [-100] * len(input_ids)
        i = 0
        while i < len(input_ids):
            if input_ids[i:i + len(bos_id)] == bos_id:
                start = i + len(bos_id)
                end = start
                while end < len(input_ids):
                    if input_ids[end:end + len(eos_id)] == eos_id:
                        break
                    end += 1
                for j in range(start, min(end + len(eos_id), len(input_ids))):
                    labels[j] = input_ids[j]
                i = end + len(eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return labels

    # 统计
    tool_call_count = 0

    # 预分配数组
    input_ids_arr = np.zeros((total, max_length), dtype=np.int32)
    labels_arr    = np.full((total, max_length), -100, dtype=np.int32)

    # 逐条处理
    for idx in tqdm(range(total), desc="Tokenizing SFT data"):
        sample = dataset[idx]
        conversation = pre_processing_chat(sample["conversations"])

        # 🛡️ 清洗每一轮：去掉 reasoning_content 等模板不认识的字段
        conversation = [_clean_turn(turn) for turn in conversation]

        # ---- 提取 tools 参数 ----
        tools = extract_tools(conversation)

        if tools:
            tool_call_count += 1
            # 工具调用数据：保留原始结构，传 tools 参数让 template 渲染
            messages = conversation
            prompt = _render_safely(
                tokenizer, messages, tools=tools,
                fallback_conversation=conversation
            )
        else:
            # 纯对话：只保留 role/content，安全保险
            messages = [
                {"role": turn["role"], "content": turn.get("content") or ""}
                for turn in conversation
            ]
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )

        prompt = post_processing_chat(prompt)

        # tokenize
        tokens = tokenizer(prompt).input_ids[:max_length]
        # pad
        tokens = tokens + [tokenizer.pad_token_id] * (max_length - len(tokens))
        # labels
        labels = generate_labels(tokens)

        input_ids_arr[idx] = np.array(tokens, dtype=np.int32)
        labels_arr[idx]    = np.array(labels, dtype=np.int32)

    # 保存
    input_path = os.path.join(output_dir, "sft_input_ids.npy")
    labels_path = os.path.join(output_dir, "sft_labels.npy")
    np.save(input_path, input_ids_arr)
    np.save(labels_path, labels_arr)

    print(f"✅ 预处理完成")
    print(f"   input_ids: {input_path} → shape {input_ids_arr.shape}")
    print(f"   labels:    {labels_path} → shape {labels_arr.shape}")
    if tool_call_count > 0:
        print(f"   📞 工具调用数据: {tool_call_count} 条（已保留 function 字段）")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="./data/sft_t2t_mini.jsonl")
    parser.add_argument("--tokenizer_root", default="./model/minimind")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--output_dir", default="./data")
    parser.add_argument("--num_samples", type=int, default=None,
                        help="调试时可指定小数量，比如 1000 条看效果")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_root, local_files_only=True, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    random.seed(42)
    process(args.data_path, tokenizer, args.max_length, args.output_dir, args.num_samples)
