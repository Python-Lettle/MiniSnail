> [返回首页](README.md) | [下一篇: 模型架构](02-model.md)

# 训练流程①：分词器与数据预处理

本篇覆盖 MiniSnail 训练流水线的第一阶段：分词器加载与数据预处理。预训练阶段把纯文本语料切成 token 流并落盘为 memmap 二进制；SFT 阶段把对话语料渲染成 ChatML 文本并构造 `-100` label mask，只让 assistant 回复参与 loss。

### 1. 分词器（Tokenizer）

MiniSnail 直接复用 minimind 项目的分词器，它是一个基于 BPE 的 `PreTrainedTokenizerFast`。仓库通过 [tokenizer.py](file:///workspace/src/minisnail/tokenizer.py#L4-L7) 中的 [get_tokenizer](file:///workspace/src/minisnail/tokenizer.py#L4-L7) 加载，该函数体只有一行，调用 `AutoTokenizer.from_pretrained(config.tokenizer.tokenizer_root)`，默认根目录 `./model/minimind` 来自 [config.py](file:///workspace/src/minisnail/config.py#L7-L14) 中 [TokenizerConfig](file:///workspace/src/minisnail/config.py#L7-L14) 的默认值。

[TokenizerConfig](file:///workspace/src/minisnail/config.py#L7-L14) 的默认字段为：

- `vocab_size = 6400`
- `tokenizer_name = "minimind"`
- `tokenizer_root = "./model/minimind"`
- `bos_token_id = 1`
- `eos_token_id = 2`

特殊 token 与关键属性由 [tokenizer_config.json](file:///workspace/model/minimind/tokenizer_config.json) 定义：

| 字段 | 取值 | id |
| --- | --- | --- |
| `bos_token` | `<|im_start|>` | 1 |
| `eos_token` | `<|im_end|>` | 2 |
| `pad_token` | `<|endoftext|>` | 0 |
| `unk_token` | `<|endoftext|>` | 0 |

此外配置文件还声明：

- `model_max_length = 131072`
- `tokenizer_class = "PreTrainedTokenizerFast"`
- `add_bos_token = false`、`add_eos_token = false`（特殊 token 由脚本手动控制）
- 内嵌一段 ChatML 的 `chat_template`（Jinja2 模板），支持 `system` / `user` / `assistant` / `tool` 四种角色，并兼容 `<think>` 思考链标签、`tool_calls` 函数调用渲染以及 `tools` 工具签名注入。

### 2. 预训练数据 tokenize（scripts/data_tokenize.py）

[data_tokenize.py](file:///workspace/scripts/data_tokenize.py) 实现了一条并行 tokenize 流水线，将预训练 jsonl 文件转换成可被 `np.memmap` 直接读取的二进制 int32 张量。

#### 目标

把 `./data/pretrain_t2t_mini.jsonl` 文本对语料 tokenize 为两份二进制文件：`./data/train_dataset.npy` 与 `./data/valid_dataset.npy`。需要特别强调的是——尽管后缀是 `.npy`，它们并不是 `np.save` 产生的标准 `.npy` 格式，而是用 `arr.tofile(...)` 直接写入的原始 int32 字节流，训练时通过 `np.memmap(..., dtype=np.int32)` 加载。

#### CLI 参数

`main` 函数的命令行参数（[data_tokenize.py](file:///workspace/scripts/data_tokenize.py#L61-L72)）：

- `--tokenizer_path`，默认 `./model/minimind`
- `--data_path`，默认 `./data/pretrain_t2t_mini.jsonl`
- `--train_output_path`，默认 `./data/train_dataset.npy`
- `--valid_output_path`，默认 `./data/valid_dataset.npy`
- `--train_ratio`，默认 `0.8`
- `--chunk_size`，默认 `2000`
- `--num_workers`，默认 `None`；为 `None` 时取 `mp.cpu_count()`

#### 两遍扫描策略

- **第一遍**（[data_tokenize.py](file:///workspace/scripts/data_tokenize.py#L82-L95)）：单遍扫描整个 jsonl，统计非空行的总字节数 `total_bytes` 与总行数 `total_lines`，再计算训练集目标字节数 `train_bytes_target = total_bytes * train_ratio`。
- **第二遍**（[data_tokenize.py](file:///workspace/scripts/data_tokenize.py#L105-L175)）：流式重读文件，把每一整行（一份完整文档）追加进 `buf_train` 并累加字节数；当累积字节达到 `train_bytes_target` 后，将标志 `switched` 置真，从下一行起全部归入 `buf_valid`。该策略保证：
  - 任意一行文档都不会被切断（不会一半在训练集、一半在验证集）
  - 训练集始终拿到完整文档，不破坏 text → next token prediction 的语义完整性
  - 每攒满 `chunk_size` 行就 yield 一个 chunk 交给 worker 处理，末尾再冲洗余量

#### 多进程编码

使用 `mp.Pool` 启动多个工作进程，初始化函数为 [_init_worker](file:///workspace/scripts/data_tokenize.py#L13-L24)：在每个 worker 内部懒加载 tokenizer（避免主进程导入 `transformers`），设置 `TOKENIZERS_PARALLELISM=false` 防止内部线程争用，并打印 `bos_token_id` / `eos_token_id` 以便核对。

每个 chunk 由 [_encode_chunk](file:///workspace/scripts/data_tokenize.py#L27-L58) 处理：

1. 逐行 `json.loads`，从字典的 `text` / `content` / `s` 字段中提取文本；若都为空但存在 `prompt` 与 `answer`，则取 `prompt + answer`（兼容多种 jsonl schema）。
2. 调用 `tokenizer(texts, padding=False, truncation=False, add_special_tokens=False)` 批量编码，不自动添加特殊 token。
3. 对每份文档**手动**前置 `bos_token_id`、后置 `eos_token_id`，把多份文档拼接成一个一维 int32 数组。
4. 返回 `(split_tag, chunk_id, np.int32 array)`。

主进程按 `pool.imap` 顺序接收结果，分别用 `arr.tofile(train_f)` / `arr.tofile(valid_f)` 追加写入对应二进制文件，并统计 train/valid 的 token 数、字节占用与吞吐（tokens/sec）。

### 3. SFT 数据预处理（scripts/preprocess_sft.py）

[preprocess_sft.py](file:///workspace/scripts/preprocess_sft.py) 将 SFT 对话语料 tokenize 为定长数组，并构造用于 SFT 训练的 `-100` label mask。

#### 目标

将 `./data/sft_t2t_mini.jsonl` 转为两份**标准** `np.save` `.npy` 文件：

- `./data/sft_input_ids.npy`，shape `[num_samples, max_length]`，int32，内容为 input_ids
- `./data/sft_labels.npy`，shape `[num_samples, max_length]`，int32，`-100` 表示该位置不参与 loss

与预训练产物不同，这两份文件是真正的 `.npy` 格式，用 `np.load` 直接加载即可。

#### CLI 参数

`__main__` 中的命令行参数（[preprocess_sft.py](file:///workspace/scripts/preprocess_sft.py#L232-L240)）：

- `--data_path`，默认 `./data/sft_t2t_mini.jsonl`
- `--tokenizer_root`，默认 `./model/minimind`
- `--max_length`，默认 `512`
- `--output_dir`，默认 `./data`
- `--num_samples`，默认 `None`（调试时可指定小数量，例如 1000 条先看效果）

#### 对话预处理函数

- [apply_name_filter](file:///workspace/scripts/preprocess_sft.py#L37-L44)：对 `content` / `reasoning_content` / `system` 文本字段做关键词替换，将 `MiniMind`→`MiniSnail`、`jingyaogong`/`gongjy`→`Lettle`（与上游项目解耦）。
- [pre_processing_chat](file:///workspace/scripts/preprocess_sft.py#L60-L80)：以 `add_system_ratio=0.2` 的概率在对话最前面插入一条随机 system prompt；若对话已带 `tools` 则跳过不处理。
- [post_processing_chat](file:///workspace/scripts/preprocess_sft.py#L83-L87)：以 80% 概率移除空的 `<think>\n\n</think>\n\n` 标签（参数 `empty_think_ratio=0.2` 表示保留空 think 的概率）。
- [_clean_turn](file:///workspace/scripts/preprocess_sft.py#L90-L112)：把 `reasoning_content` 合并进 `content`（`f"{reasoning}\n\n{content}"`，保留 CoT 思考链训练信号，因为 minimind 的 ChatML 模板不识别 `reasoning_content` 字段），同时保留 `role` / `function` / `name` / `tool_calls`，丢弃其他未知字段。
- [extract_tools](file:///workspace/scripts/preprocess_sft.py#L47-L57)：从第一条非 system 对话中，若带 `function` 字段则提取为 `tools`，否则返回 `None`。
- [_render_safely](file:///workspace/scripts/preprocess_sft.py#L115-L133)：先尝试 `tokenizer.apply_chat_template(messages, tools=tools)` 渲染；若抛异常则降级为只保留 `role`/`content` 的纯对话重新渲染。

#### `-100` label mask 机制（核心）

SFT 的关键在于让模型只对 assistant 的回复学习，system / user / tool 角色的 token 不应参与 loss。[preprocess_sft.py](file:///workspace/scripts/preprocess_sft.py#L151-L167) 中的 [generate_labels](file:///workspace/scripts/preprocess_sft.py#L151-L167) 实现如下：

1. 默认 `labels = [-100] * len(input_ids)`，全部置为 `-100`（PyTorch `CrossEntropyLoss` 默认忽略 `-100`）。
2. 通过 [preprocess_sft.py](file:///workspace/scripts/preprocess_sft.py#L148-L149) 中预先探测的两个 marker 序列来定位 assistant 回复区间：
   - `bos_id = tokenizer(f"{tokenizer.bos_token}assistant\n", add_special_tokens=False).input_ids`，即 `<|im_start|>assistant\n` 的 token 序列，标记 assistant 回复开始。
   - `eos_id = tokenizer(f"{tokenizer.eos_token}\n", add_special_tokens=False).input_ids`，即 `<|im_end|>\n` 的 token 序列，标记 assistant 回复结束。
3. 在 `input_ids` 中线性扫描 `bos_id`，定位到 `start = i + len(bos_id)`；从 `start` 起向后扫描直到匹配到 `eos_id`，得到 `end`。
4. 将区间 `[start, end + len(eos_id))` 内的 labels 解除 mask，置为对应的 `input_ids[j]`（包含结尾的 eos token），其余位置保持 `-100`。

也就是说：只有 assistant 的回复内容（含结尾 `<|im_end|>`）参与 loss 计算，其他角色一律被 `-100` 屏蔽——这是 SFT 指令微调的核心机制。

#### tokenize 与 padding

单条样本的处理流程（[preprocess_sft.py](file:///workspace/scripts/preprocess_sft.py#L209-L217)）：

1. `tokens = tokenizer(prompt).input_ids[:max_length]`，先 tokenize 再截断到 `max_length`。
2. `tokens = tokens + [tokenizer.pad_token_id] * (max_length - len(tokens))`，用 `pad_token_id`（=0）补齐到 `max_length`。
3. `labels = generate_labels(tokens)`，在 padding 后的序列上生成 mask（padding 部分自然为 `-100`）。

输出数组预先分配（[preprocess_sft.py](file:///workspace/scripts/preprocess_sft.py#L173-L174)）：

- `input_ids_arr = np.zeros((total, max_length), dtype=np.int32)`
- `labels_arr = np.full((total, max_length), -100, dtype=np.int32)`

逐条填充后用 `np.save` 写出两份 `.npy` 文件，并打印最终 shape 与工具调用样本数。

### 4. 数据产物一览

| 产物文件 | 格式 | 加载方式 | 用途 |
| --- | --- | --- | --- |
| `./data/train_dataset.npy` | 原始 int32 二进制流 | `np.memmap(dtype=np.int32)` | 预训练训练集 |
| `./data/valid_dataset.npy` | 原始 int32 二进制流 | `np.memmap(dtype=np.int32)` | 预训练验证集 |
| `./data/sft_input_ids.npy` | np.save `.npy`（int32） | `np.load` | SFT input_ids |
| `./data/sft_labels.npy` | np.save `.npy`（int32） | `np.load` | SFT labels（含 -100 mask） |
