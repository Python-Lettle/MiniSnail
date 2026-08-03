# MiniSnail Code Wiki

## A — 项目概览

MiniSnail 是一个小参数语言模型训练框架，旨在用参数量小得多的模型模拟大模型的完整流程（预训练 pretrain → 监督微调 SFT → 推理 inference）。它是 minimind 项目的重实现/重命名版本。包名为 `minisnail`，版本 0.1.0，要求 Python >=3.10，通过 setuptools 打包，源码包位于 `src/` 下（见 [pyproject.toml](file:///workspace/pyproject.toml)）。

## B — 项目目录结构

```text
/workspace/
├── model/
│   └── minimind/
│       ├── tokenizer.json
│       └── tokenizer_config.json
├── scripts/
│   ├── data_tokenize.py
│   ├── generate_config.py
│   ├── get_model_from_checkpoint.py
│   ├── preprocess_sft.py
│   ├── train_lm.py
│   └── train_sft.py
├── src/
│   └── minisnail/
│       ├── __init__.py
│       ├── config.py
│       ├── dataset.py
│       ├── debug.py
│       ├── functions.py
│       ├── generate.py
│       ├── model.py
│       ├── tokenizer.py
│       └── util.py
├── tests/
│   ├── test_data_tokenize.py
│   ├── test_dataset.py
│   ├── test_lm.py
│   ├── test_sft_data.py
│   ├── test_sft_lm.py
│   └── test_tokenizer.py
├── pyproject.toml
├── README.md
└── LICENSE
```

说明：源代码位于 `src/minisnail/`；入口脚本位于 `scripts/`。

## C — 整体架构图

```mermaid
flowchart LR
    A[原始 jsonl 数据] --> B[分词器 minimind]
    B --> C[数据预处理 预训练tokenize / SFT预处理]
    C --> D[SnailModel Transformer]
    D --> E[预训练 train_lm]
    E --> F[SFT微调 train_sft]
    F --> G[检查点/模型权重]
    G --> H[推理生成 generate/chat]
```

- 原始 jsonl 数据经分词器（minimind）切分为 token 序列。
- 数据预处理阶段分两条路径：预训练 tokenize（生成 `.npy` 一维 token 数组）与 SFT 预处理（生成 `input_ids` 与 `labels` 数组）。
- 处理后的数据送入 SnailModel（Transformer 架构）进行预训练 `train_lm`，再进行 SFT 微调 `train_sft`。
- 训练产出的检查点/模型权重用于推理生成（`generate` / `chat`）。

## D — 模块职责表

| 文件 | 职责 | 定位 |
| --- | --- | --- |
| [config.py](file:///workspace/src/minisnail/config.py) | 配置系统：8 个 dataclass 子配置 + SnailConfig | `src/minisnail/config.py` |
| [model.py](file:///workspace/src/minisnail/model.py) | 模型架构：SnailModel / SnailBlock / 注意力 / RoPE / SwiGLU FFN | `src/minisnail/model.py` |
| [dataset.py](file:///workspace/src/minisnail/dataset.py) | 数据集与 DataLoader：PretrainDataset / SFTDataset | `src/minisnail/dataset.py` |
| [tokenizer.py](file:///workspace/src/minisnail/tokenizer.py) | 分词器加载 | `src/minisnail/tokenizer.py` |
| [functions.py](file:///workspace/src/minisnail/functions.py) | 训练核心函数：损失/调度/裁剪/注意力 | `src/minisnail/functions.py` |
| [generate.py](file:///workspace/src/minisnail/generate.py) | 推理生成封装 | `src/minisnail/generate.py` |
| [util.py](file:///workspace/src/minisnail/util.py) | 工具：随机种子/memmap 读取 | `src/minisnail/util.py` |
| [debug.py](file:///workspace/src/minisnail/debug.py) | 调试：Console / LossMonitor | `src/minisnail/debug.py` |
| [data_tokenize.py](file:///workspace/scripts/data_tokenize.py) | 预训练数据并行 tokenize | `scripts/data_tokenize.py` |
| [preprocess_sft.py](file:///workspace/scripts/preprocess_sft.py) | SFT 数据预处理 | `scripts/preprocess_sft.py` |
| [train_lm.py](file:///workspace/scripts/train_lm.py) | 预训练脚本 | `scripts/train_lm.py` |
| [train_sft.py](file:///workspace/scripts/train_sft.py) | SFT 微调脚本 | `scripts/train_sft.py` |
| [generate_config.py](file:///workspace/scripts/generate_config.py) | 生成默认 config.json | `scripts/generate_config.py` |
| [get_model_from_checkpoint.py](file:///workspace/scripts/get_model_from_checkpoint.py) | 从检查点提取模型权重 | `scripts/get_model_from_checkpoint.py` |

## E — 第三方依赖

以下依赖为运行时所需，但未在 [pyproject.toml](file:///workspace/pyproject.toml) 中声明（该文件仅声明构建元数据），需手动安装。

| 依赖 | 用途 |
| --- | --- |
| torch | 模型/训练/张量 |
| numpy | 数据数组/memmap |
| transformers | 分词器 AutoTokenizer |
| einops | rearrange/einsum 张量重排 |
| jaxtyping | 张量形状类型注解 |
| rich | 终端美化输出 Console |
| matplotlib | Loss 曲线绘制 |
| wandb | 实验追踪 |
| datasets | SFT jsonl 加载 |
| tqdm | 进度条 |

## F — 项目内部模块依赖关系

```text
minisnail 内部模块
├── model.py
│   ├── ← config.py (SnailConfig)
│   └── ← debug.py (console)
├── dataset.py
│   └── ← debug.py (console)
├── tokenizer.py
│   ├── ← config.py (SnailConfig)
│   └── ← transformers (AutoTokenizer, PreTrainedTokenizer)
├── generate.py
│   ├── ← model.py (SnailModel)
│   ├── ← config.py (SnailConfig)
│   └── ← util.py (console)
├── functions.py  (独立，仅依赖 torch, math, einops, jaxtyping)
├── util.py       (独立，仅依赖 os, random, numpy, torch, rich)
└── debug.py      (独立，仅依赖 rich, matplotlib, numpy)

scripts 入口脚本
├── train_lm.py
│   └── ← model, dataset, functions, config, util, debug
├── train_sft.py
│   └── ← model, dataset, functions, config, util, debug
└── preprocess_sft.py
    └── ← transformers, datasets, numpy
```

关键事实：

- model.py 从 config.py 导入 SnailConfig，从 debug.py 导入 console。
- dataset.py 从 debug.py 导入 console。
- tokenizer.py 从 config.py 导入 SnailConfig，并使用 transformers。
- generate.py 从 model.py 导入 SnailModel，从 config.py 导入 SnailConfig，从 util.py 导入 console。
- functions.py 独立（仅依赖 torch, math, einops, jaxtyping）。
- util.py 独立（仅依赖 os, random, numpy, torch, rich）。
- debug.py 独立（仅依赖 rich, matplotlib, numpy）。
- scripts/train_lm.py 导入 model, dataset, functions, config, util, debug。
- scripts/train_sft.py 导入 model, dataset, functions, config, util, debug。
- scripts/preprocess_sft.py 导入 transformers, datasets, numpy。

## G — 配置系统总览

SnailConfig 由 8 个 dataclass 子配置组合而成（见 [config.py](file:///workspace/src/minisnail/config.py#L89-L147)）。

| 子配置 | 关键字段 | 默认值 |
| --- | --- | --- |
| TokenizerConfig | vocab_size / tokenizer_name / tokenizer_root / bos_token_id / eos_token_id | 6400 / "minimind" / "./model/minimind" / 1 / 2 |
| ModelConfig | vocab_size / context_length / d_model / num_layers / num_heads / d_ff / rope_theta / rms_norm_eps | 6400 / 512 / 512 / 4 / 16 / 1344 / 10000.0 / 1e-6 |
| TrainingConfig | epochs / batch_size / lr / betas / weight_decay / valid_interval / gradient_clip / accumulation_steps / print_interval / from_weight / use_checkpoint / from_checkpoint | 6000 / 32 / 0 / (0.9, 0.95) / 0.001 / 400 / 1.0 / 1 / 200 / None / False / None |
| SchedulerConfig | max_learning_rate / min_learning_rate / warmup_iters / cosine_cycle_iters | 0.0005 / 0.00005 / 600 / 6000 |
| [DataConfig](file:///workspace/src/minisnail/config.py#L52-L62) | train_data_path / valid_data_path / input_ids_path / labels_path / save_model_dir / dataset_name | "./data/train_dataset.npy" / "./data/valid_dataset.npy" / "./data/sft_input_ids.npy" / "./data/sft_labels.npy" / "./output/" / "t2t" |
| [SystemConfig](file:///workspace/src/minisnail/config.py#L64-L69) | device / seed / dtype | "cuda" / 42 / "float32" |
| GenerationConfig | model_path / max_tokens / temperature / top_k / device / repetition_penalty | "./output/model_best.pt" / 512 / 0.8 / 40 / "cuda" / 1.2 |
| WandbConfig | entity / project / id | "lettle-hong" / "MiniSnail" / None |

SnailConfig 方法：

- `get_torch_dtype()`（见 [config.py](file:///workspace/src/minisnail/config.py#L100-L107)）：将 `system.dtype` 字符串映射为 `(model_dtype, amp_dtype)` 元组。"float32" → `(fp32, None)`；"bfloat16" → `(bf16, bf16)`；"float16" → `(fp16, fp16)`。
- `from_json(json_path)`：从 JSON 文件加载配置。
- `to_json(json_path)`：将配置保存为 JSON 文件。
- `from_dict(d)`：从字典创建配置。
- `to_dict()`：将配置转为字典。

## H — 训练流程主线串联图

以下是 LLM 训练流程的“主线”，每个阶段映射到对应的文档：

1. 准备阶段 → 本页（README）
2. 数据阶段（分词器 + 数据预处理）→ [01-tokenizer-data.md](01-tokenizer-data.md)
3. 模型阶段（模型架构）→ [02-model.md](02-model.md)
4. 训练阶段（数据加载 + 训练函数 + 预训练 + SFT + 检查点 + 实验追踪）→ [03-training.md](03-training.md)
5. 推理阶段（推理与生成）→ [04-inference.md](04-inference.md)
6. 运行方式 → [05-how-to-run.md](05-how-to-run.md)

## I — 文档导航目录

| 文档 | 阶段 | 内容 |
| --- | --- | --- |
| [01-tokenizer-data.md](01-tokenizer-data.md) | 训练流程① | 分词器与数据预处理（预训练 tokenize / SFT 预处理） |
| [02-model.md](02-model.md) | 训练流程② | 模型架构与关键类/函数 |
| [03-training.md](03-training.md) | 训练流程③ | 数据加载、训练函数、预训练、SFT、检查点、实验追踪 |
| [04-inference.md](04-inference.md) | 训练流程④ | 推理与生成 |
| [05-how-to-run.md](05-how-to-run.md) | 训练流程⑤ | 项目运行方式 |
