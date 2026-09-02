import os
import glob
import json
import argparse
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer
import re
# chat_template 给每条 assistant 注入的是
#   <|im_start|>assistant\n<think>\n{推理}\n</think>\n\n{正文}<|im_end|>\n
# 删掉 think 块时要把模板紧跟其后的两个换行一起吃掉, 否则留下 <|im_start|>assistant\n\n\n{正文},
# 与推理端 <|im_start|>assistant\n 差两个空行。只吃换行不吃空格, 避免误伤正文的缩进。
pattern = re.compile(r'<think>.*?</think>\n*', re.S)

REPLACE_RULES = []

FILTER_WORDS = []

SUPPORTED_RULE_EXTS = (".json", ".jsonl", ".txt")

def _is_rule_dict(obj):
    """判断一个 dict 是否为单条规则 {patterns: [...], replacement: "..."}"""
    keys = {"patterns", "replacement", "from", "to", "src", "dst"}
    return bool(keys & set(obj.keys()))

def _normalize_rule(obj, source, index):
    """把任意形式的规则对象规范化为 (patterns, replacement) 并做校验"""
    # 列表形式约定为 [patterns, replacement]
    if isinstance(obj, (list, tuple)):
        if len(obj) != 2:
            raise ValueError(
                f"{source} 第 {index} 条规则：列表形式必须是 [patterns, replacement]，当前长度为 {len(obj)}"
            )
        patterns, replacement = obj
    elif isinstance(obj, dict):
        patterns = obj.get("patterns", obj.get("from", obj.get("src")))
        replacement = obj.get("replacement", obj.get("to", obj.get("dst")))
    else:
        raise ValueError(f"{source} 第 {index} 条规则：不支持的类型 {type(obj).__name__}")

    if isinstance(patterns, str):
        patterns = [patterns]
    if not isinstance(patterns, (list, tuple)) or not patterns:
        raise ValueError(f"{source} 第 {index} 条规则：patterns 必须是非空字符串或字符串列表")
    if not isinstance(replacement, str):
        raise ValueError(f"{source} 第 {index} 条规则：replacement 必须是字符串")

    patterns = [p.strip() for p in patterns if isinstance(p, str) and p.strip()]
    if not patterns:
        raise ValueError(f"{source} 第 {index} 条规则：patterns 中没有有效的字符串")

    # 提前校验正则，避免运行时才炸
    for p in patterns:
        try:
            re.compile(p)
        except re.error as e:
            raise ValueError(f"{source} 第 {index} 条规则：pattern `{p}` 不是合法正则 - {e}")

    return (patterns, replacement)

def load_replace_rules(path):
    """
    从文件加载替换规则，返回 [(patterns, replacement), ...]

    支持三种格式：
      1. .json  —— 规则数组
            [
              {"patterns": ["MiniMind", "minimind"], "replacement": "MiniSnail"},
              {"patterns": ["京遥"], "replacement": "Lettle"}
            ]
         或 replacement -> patterns 的映射
            {"MiniSnail": ["MiniMind", "minimind"], "Lettle": ["京遥"]}
      2. .jsonl —— 每行一条规则（同上对象形式），# 开头为注释
      3. .txt   —— 每行一条，左侧多个 pattern 用 | 分隔，右侧为 replacement
            MiniMind|minimind => MiniSnail
    """
    ext = os.path.splitext(path)[1].lower()
    if ext not in SUPPORTED_RULE_EXTS:
        raise ValueError(
            f"不支持的规则文件类型 `{ext}`，仅支持 {' / '.join(SUPPORTED_RULE_EXTS)}"
        )

    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()

    objs = []
    if ext == ".json":
        data = json.loads(raw)
        if isinstance(data, dict) and not _is_rule_dict(data):
            # {"MiniSnail": ["MiniMind", ...]} 映射形式
            for replacement, patterns in data.items():
                objs.append({"patterns": patterns, "replacement": replacement})
        elif isinstance(data, dict):
            objs.append(data)
        elif isinstance(data, list):
            objs.extend(data)
        else:
            raise ValueError(f"{path}：JSON 顶层必须是数组、规则对象或 replacement->patterns 映射")
    elif ext == ".jsonl":
        for line in raw.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            objs.append(json.loads(line))
    else:
        for line_no, line in enumerate(raw.splitlines(), 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=>" in line:
                left, right = line.split("=>", 1)
            elif "->" in line:
                left, right = line.split("->", 1)
            else:
                raise ValueError(f"{path}:{line_no} 缺少 `=>` 分隔符：{line}")
            objs.append(([p.strip() for p in left.split("|")], right.strip()))

    rules = [_normalize_rule(obj, path, i) for i, obj in enumerate(objs, 1)]
    if not rules:
        raise ValueError(f"{path}：没有解析到任何有效规则")
    return rules

def load_filter_words(path):
    """从文件加载过滤词，支持 .json（字符串数组）/ .jsonl / .txt（每行一个）"""
    ext = os.path.splitext(path)[1].lower()
    if ext not in SUPPORTED_RULE_EXTS:
        raise ValueError(
            f"不支持的过滤词文件类型 `{ext}`，仅支持 {' / '.join(SUPPORTED_RULE_EXTS)}"
        )

    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()

    if ext == ".json":
        words = json.loads(raw)
        if not isinstance(words, list):
            raise ValueError(f"{path}：JSON 顶层必须是字符串数组")
    else:
        words = [
            line.strip()
            for line in raw.splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]

    words = [w for w in words if isinstance(w, str) and w.strip()]
    if not words:
        raise ValueError(f"{path}：没有解析到任何有效过滤词")
    return words

def remove_think(text):
    text = re.sub(pattern, "", text)
    # 不再 strip(): 模板渲染结果自带结尾换行, 那是消息之间的分隔符,
    # 之前 strip 掉它导致训练侧是 <|im_end|><|im_start|>assistant, 而推理侧是 <|im_end|>\n<|im_start|>assistant
    return text

def clean_content(text, replace_rules=None, filter_words=None):
    if not isinstance(text,str):
        return text

    replace_rules = REPLACE_RULES if replace_rules is None else replace_rules
    filter_words = FILTER_WORDS if filter_words is None else filter_words

    # ----------过滤----------
    for word in filter_words:
        if word.lower() in text.lower():
            return None
    # ----------替换----------
    for patterns, replacement in replace_rules:
        for pattern in patterns:
            text = re.sub(
                pattern,
                replacement,
                text,
                flags=re.I
            )

    # ----------去think----------
    # 保留模板的换行结构 (结尾换行由 build_sample 统一处理), 不要用 strip 抹掉
    text = remove_think(text)
    return text

def build_sample(messages, tokenizer, max_length, replace_rules=None, filter_words=None,
                 stats=None):
    """
    将一条对话转换为 输入ID 和 标签ID 列表

    产出的文本格式与推理端 model.chat() / tests/test_sft_lm.py 拼的 prompt 严格对齐:

        训练  <|im_start|>user\\n{问}<|im_end|>\\n<|im_start|>assistant\\n{答}<|im_end|>
        推理  <|im_start|>user\\n{问}<|im_end|>\\n<|im_start|>assistant\\n   <- 生成起点

    即训练序列的前缀与推理 prompt 的 token 序列完全相同, 模型在推理时看到的上下文
    在训练中出现过。两个要点:
      1. 消息之间是 <|im_end|>\\n<|im_start|>role, 不能丢了那个 \\n;
      2. assistant 段是 <|im_start|>assistant\\n 后直接接正文, 没有 think 块也没有残留空行。

    返回 (input_ids, labels); 样本不可用时返回 (None, None)。
    丢弃原因累加进 stats (可选): filtered / too_long / no_supervision / kept
    """
    input_ids = []
    labels = []
    has_supervision = False

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
        text = clean_content(text, replace_rules=replace_rules, filter_words=filter_words)
        # 命中过滤词则跳过整个样本，避免破坏多轮对话结构
        if text is None:
            if stats is not None:
                stats["filtered"] += 1
            return None, None

        # 模板渲染结果形如 "<|im_start|>role\n{正文}<|im_end|>\n":
        # 结尾那个 \n 是下一条消息的分隔符, 这里先去掉, 改由下面统一在消息之间补一个 \n,
        # 使训练侧恰好是 "<|im_end|>\n<|im_start|>assistant", 与推理端 model.chat() 拼的 prompt 一致。
        text = text.rstrip("\n")

        body_ids = tokenizer(text, add_special_tokens=False).input_ids

        # 首条消息之外补分隔换行。
        # 注意: \n 夹在 <|im_end|>(特殊 token) 与 <|im_start|>(特殊 token) 之间,
        # BPE 不会跨越特殊 token 合并, 所以这里单独补的 \n 与整段一次性编码的切分结果一致。
        sep_ids = [] if not input_ids else tokenizer("\n", add_special_tokens=False).input_ids

        ids = sep_ids + body_ids
        # 单条消息已超出整个上下文窗口时，必须丢弃整条对话。
        # 如果只跳过当前消息，后续 assistant 回复会在缺少对应 prompt 的
        # 情况下进入训练集，导致样本语义错位。
        if len(ids) > max_length:
            if stats is not None:
                stats["too_long"] += 1
            return None, None

        # 逐条累加后超预算则整条样本丢弃。
        # 不能只丢当前消息后继续: 一是丢掉 assistant 会留下 labels 全 -100 的空样本,
        # 二是多行累加超长会让行长 > max_length, 最终 np.array 因行长不一致而抛 ValueError。
        # 也不能截断: 截断会切掉结尾的 <|im_end|>, 模型就学不会主动停止。
        if len(input_ids) + len(ids) > max_length:
            if stats is not None:
                stats["too_long"] += 1
            return None, None

        input_ids.extend(ids)

        if role == "assistant":
            # 分隔换 \n 是推理端 prompt 的一部分 (由 model.chat() 手写拼出), 不计入 loss;
            # <|im_start|>assistant\n 这个头部保留监督, 沿用原有约定
            labels.extend([-100] * len(sep_ids) + body_ids)
            has_supervision = True
        else:
            labels.extend([-100] * len(ids))

    # assistant 全部被丢弃时 labels 全是 -100: 这种样本不产生任何梯度,
    # 却照样占 batch 位置、稀释 loss 统计, 验证时还会算出 nan
    if not has_supervision:
        if stats is not None:
            stats["no_supervision"] += 1
        return None, None

    # 走到这里 len(input_ids) <= max_length, pad_len 必为非负
    pad_len = max_length - len(input_ids)
    input_ids += [tokenizer.pad_token_id] * pad_len
    labels += [-100] * pad_len

    if stats is not None:
        stats["kept"] += 1

    return input_ids, labels

def count_lines(path):
    """快速统计行数 (按块读, 不把文件读进内存)。只用于进度条和 num_samples 夹取。"""
    n = 0
    last = b"\n"
    with open(path, "rb") as f:
        while True:
            buf = f.read(1 << 20)
            if not buf:
                break
            n += buf.count(b"\n")
            last = buf[-1:]
    # 末行没有换行符时补一个
    if last not in (b"\n", b""):
        n += 1
    return n

def iter_jsonl(path, limit=None, stats=None):
    """
    逐行读 jsonl, 跳过空行与解析失败的行。

    这里不用 datasets.load_dataset: 它按第一个 chunk 推断 arrow schema, 而本语料后半段
    (约 59% 处起) 混入了带 tools / tool_calls 字段的工具调用样本, 字段数与首个 chunk 推断出的
    schema 对不上, 会直接抛 "Couldn't cast array of type struct<...> to {...}" 让整体失败。
    逐行读顺带还有两个好处: 单行损坏只丢那一行, 且不会在 ~/.cache/huggingface 再落一份全量副本。

    stats 不为 None 时累加计数: blank (空行)、bad_line (JSON 解析失败),
    保证 kept + 各类丢弃 + blank + bad_line == 读入行数。
    """
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                return
            if not line.strip():
                if stats is not None:
                    stats["blank"] += 1
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                if stats is not None:
                    stats["bad_line"] += 1
                    if stats["bad_line"] <= 5:
                        print(f"[warn] 第 {i} 行 JSON 解析失败, 已跳过: {e}")
                continue

def has_tool_fields(messages):
    """样本是否带工具调用字段 (system.tools / assistant.tool_calls)"""
    for m in messages:
        if isinstance(m, dict) and ("tools" in m or "tool_calls" in m):
            return True
    return False

def _existing_shards(output_dir):
    """找出目录里已存在的历史产物(含旧的单文件命名 sft_input_ids.npy)"""
    ids_files = sorted(glob.glob(os.path.join(output_dir, "sft_input_ids*.npy")))
    lab_files = sorted(glob.glob(os.path.join(output_dir, "sft_labels*.npy")))
    return ids_files + lab_files

def process(data_path,tokenizer, max_length, output_dir, num_samples=None,
            replace_rules=None, filter_words=None, dtype=np.int16,
            shard_size=100000, overwrite=False, tool_policy="drop"):

    replace_rules = REPLACE_RULES if replace_rules is None else replace_rules
    filter_words = FILTER_WORDS if filter_words is None else filter_words

    os.makedirs(output_dir, exist_ok=True)

    # 历史产物必须清理: 分片命名带序号, 上一轮多出来的分片会被 dataset 一并 glob 进去,
    # 静默混进训练数据。不自动删, 让调用方显式确认。
    stale = _existing_shards(output_dir)
    if stale and not overwrite:
        raise SystemExit(
            f"{output_dir} 下已存在 {len(stale)} 个历史产物, 例如:\n  "
            + "\n  ".join(stale[:5])
            + ("\n  ..." if len(stale) > 5 else "")
            + "\n请先手动删除, 或加 --overwrite 覆盖。"
        )
    failed = []
    for path in stale:
        try:
            os.remove(path)
        except OSError as e:
            failed.append((path, e))
    if failed:
        raise SystemExit(
            "无法删除以下历史产物, 请手动清理后重试:\n  "
            + "\n  ".join(f"{p}  ({e})" for p, e in failed)
        )

    # 逐行流式读取, 不做 arrow schema 推断 (语料后半段混有 tools/tool_calls 字段, 推断必炸)
    n_lines = count_lines(data_path)
    # 越界会在读取时静默截断, 这里显式夹一下并提示
    n_read = min(int(num_samples), n_lines) if num_samples else n_lines

    stats = {"filtered": 0, "too_long": 0, "no_supervision": 0, "kept": 0,
             "tool": 0, "bad_line": 0, "blank": 0}

    # 分片缓冲写盘: 全量约 863 万条, 若先攒成 Python list 再 np.array,
    # 中间态 (list[list[int]]) 需要数百 GB 内存, 在写盘前就 OOM。
    # 这里预分配固定大小的 ndarray 缓冲, 攒满一个分片就落盘。
    shard_size = max(1, int(shard_size))
    buf_ids = np.empty((shard_size, max_length), dtype=dtype)
    buf_lab = np.empty((shard_size, max_length), dtype=dtype)
    n_buffered = 0
    shard_idx = 0

    def flush(count):
        nonlocal shard_idx
        if count <= 0:
            return
        id_path = os.path.join(output_dir, f"sft_input_ids_{shard_idx:04d}.npy")
        lb_path = os.path.join(output_dir, f"sft_labels_{shard_idx:04d}.npy")
        np.save(id_path, buf_ids[:count])
        np.save(lb_path, buf_lab[:count])
        shard_idx += 1

    for item in tqdm(iter_jsonl(data_path, limit=n_read, stats=stats),
                     total=n_read):
        if not isinstance(item, dict) or not isinstance(item.get("conversations"), list):
            stats["bad_line"] += 1
            continue
        messages = item["conversations"]

        # 工具调用样本 (system.tools / assistant.tool_calls) 是另一类任务:
        # 里面有一半 assistant 轮 content 为空、真正内容在 tool_calls 里, 直接编码会教模型
        # "输出空回复然后 <|im_end|>", 而且会引入 <tool_response> 等模型无法复现的标记。
        if tool_policy == "drop" and has_tool_fields(messages):
            stats["tool"] += 1
            continue

        ids, labs = build_sample(
            messages,
            tokenizer,
            max_length,
            replace_rules=replace_rules,
            filter_words=filter_words,
            stats=stats,
        )
        # 命中过滤词 / 超长 / 无监督目标的样本跳过
        if ids is None:
            continue
        buf_ids[n_buffered] = ids
        buf_lab[n_buffered] = labs
        n_buffered += 1
        if n_buffered == shard_size:
            flush(n_buffered)
            n_buffered = 0

    flush(n_buffered)

    meta = {
        "num_samples": stats["kept"],
        "max_length": int(max_length),
        "dtype": np.dtype(dtype).name,
        "num_shards": shard_idx,
        "source": os.path.abspath(data_path),
        "total_lines_in_file": int(n_lines),
        "total_input_samples": int(n_read),
        "tool_policy": tool_policy,
        "stats": stats,
    }
    with open(os.path.join(output_dir, "sft_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    dropped = n_read - stats["kept"]
    print(f"File lines {n_lines} | read {n_read} -> kept {stats['kept']} "
          f"({stats['kept'] / max(n_read, 1) * 100:.1f}%), dropped {dropped}")
    print(f"  by FILTER_WORDS : {stats['filtered']}")
    print(f"  by max_length   : {stats['too_long']}")
    print(f"  no supervision  : {stats['no_supervision']}")
    print(f"  tool samples    : {stats['tool']}  (--tool_policy={tool_policy})")
    print(f"  bad lines       : {stats['bad_line']}")
    print(f"  blank lines     : {stats['blank']}")
    print(f"Shards: {shard_idx} x <= {shard_size} samples, "
          f"dtype={np.dtype(dtype).name}, max_length={max_length}")
    print("Pre-process SFT dataset done!")

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--data_path", default="./dataset/full/sft_t2t.jsonl")
    parser.add_argument("--tokenizer_root", default="./model/minimind")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--output_dir", default="./dataset/full")
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument(
        "--replace_rules",
        default=None,
        help="替换规则文件路径（.json / .jsonl / .txt），不传则使用脚本内的 REPLACE_RULES",
    )
    parser.add_argument(
        "--filter_words",
        default=None,
        help="过滤词文件路径（.json / .jsonl / .txt，每行/每项一个），不传则使用脚本内的 FILTER_WORDS",
    )
    parser.add_argument(
        "--shard_size",
        type=int,
        default=100000,
        help="每个分片最多容纳多少条样本, 攒满即落盘 (控制峰值内存)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="覆盖 output_dir 下已有的 sft_input_ids*.npy / sft_labels*.npy",
    )
    parser.add_argument(
        "--tool_policy",
        choices=["drop", "keep"],
        default="drop",
        help="遇到带 tools / tool_calls 字段的样本如何处理: "
             "drop=整条丢弃 (默认, 这类样本一半 assistant 轮正文为空, "
             "直接编码会教模型输出空回复), keep=按普通样本编码",
    )
    args = parser.parse_args()

    replace_rules = REPLACE_RULES
    if args.replace_rules:
        replace_rules = load_replace_rules(args.replace_rules)
        print(
            f"Loaded {len(replace_rules)} replace rules from {args.replace_rules} "
            f"(covering {sum(len(p) for p, _ in replace_rules)} patterns)"
        )

    filter_words = FILTER_WORDS
    if args.filter_words:
        filter_words = load_filter_words(args.filter_words)
        print(f"Loaded {len(filter_words)} filter words from {args.filter_words}")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_root, local_files_only=True, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("Warning: pad_token is None, set to eos_token.")

    process(
        args.data_path,
        tokenizer,
        args.max_length,
        args.output_dir,
        num_samples=args.num_samples,
        replace_rules=replace_rules,
        filter_words=filter_words,
        shard_size=args.shard_size,
        overwrite=args.overwrite,
        tool_policy=args.tool_policy,
    )
