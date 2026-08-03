# 生成 MiniSnail 项目 Code Wiki 文档 Spec

## Why
MiniSnail 是一个用于模拟大模型全流程能力的小型语言模型训练框架（源自 minimind 的复刻/改名）。项目代码已成型但缺少结构化文档，不利于后续"以更高代码质量复刻"的目标。需要先产出一套以「大模型训练流程」为主线、完整结构化的 Code Wiki，作为复刻工作的参考蓝本与认知基线。

## What Changes
- 新增 `docs/` 目录，存放一整套 Markdown Code Wiki 文档
- Wiki 以「大模型训练流程」为主线按顺序组织：项目概览与架构 → 分词器与数据预处理 → 模型架构 → 训练流程（预训练 + SFT）→ 推理生成 → 运行方式
- 文档需覆盖：项目整体架构、模块职责、关键类与函数说明、依赖关系、项目运行方式
- 文档中所有代码引用均使用可点击的 `file:///` 链接定位到真实文件与行号
- 文档末尾附「训练流程主线串联图」，把各阶段串成一条完整链路

## Impact
- Affected specs: 无（首个 spec）
- Affected code: 不改动任何源码，仅新增 `docs/` 目录下的只读文档
- 依赖源码文件：
  - `src/minisnail/`：`config.py`、`model.py`、`dataset.py`、`tokenizer.py`、`functions.py`、`generate.py`、`util.py`、`debug.py`
  - `scripts/`：`data_tokenize.py`、`preprocess_sft.py`、`train_lm.py`、`train_sft.py`、`generate_config.py`、`get_model_from_checkpoint.py`
  - `pyproject.toml`、`model/minimind/tokenizer_config.json`、`tests/`

## ADDED Requirements

### Requirement: Code Wiki 文档目录结构
系统 SHALL 在 `docs/` 目录下生成以下 Markdown 文档，并以 `README.md` 作为导航首页：
- `README.md` — 导航首页 + 项目概览 + 整体架构图 + 模块职责表 + 依赖关系 + 配置系统总览
- `01-tokenizer-data.md` — 训练流程①：分词器 + 数据预处理（预训练 tokenize / SFT 预处理）
- `02-model.md` — 训练流程②：模型架构 + 关键类与函数
- `03-training.md` — 训练流程③：数据加载 + 训练核心函数 + 预训练 + SFT + 检查点 + 实验追踪
- `04-inference.md` — 训练流程④：推理与生成
- `05-how-to-run.md` — 项目完整运行方式

#### Scenario: 文档可导航
- **WHEN** 用户打开 `docs/README.md`
- **THEN** 能通过目录链接跳转到每一篇子文档，且每篇子文档顶部有「返回首页」与「上/下一篇」导航

### Requirement: 以大模型训练流程为主线串联
文档 SHALL 按照大模型训练的自然顺序组织内容主线，顺序为：
1. 准备阶段：项目概览、整体架构、环境依赖、目录结构、配置系统
2. 数据阶段：分词器加载 → 预训练数据 tokenize → SFT 数据预处理（含标签 -100 mask 机制）
3. 模型阶段：Token Embedding → RoPE → N×SnailBlock（Pre-Norm + 多头自注意力 + SwiGLU FFN + 残差）→ RMSNorm → 输出层
4. 训练阶段：配置加载 → DataLoader → 损失函数 → 余弦学习率调度 → 梯度裁剪 → AdamW → 梯度累积 → AMP 混合精度 → 验证 → 检查点 → Wandb 追踪 → LossMonitor
5. 推理阶段：temperature / top-k / repetition_penalty 采样生成 + chat 模板对话

#### Scenario: 主线清晰可追溯
- **WHEN** 读者从 README 按编号顺序阅读至 `05-how-to-run.md`
- **THEN** 能完整还原"从原始 jsonl 数据到一个可对话的小模型"的端到端链路，且每一步均能定位到对应源码文件

### Requirement: 关键类与函数说明
文档 SHALL 对以下关键类与函数给出职责、输入输出、所在文件定位说明：
- 配置：`SnailConfig` 及其子配置（`TokenizerConfig`/`ModelConfig`/`TrainingConfig`/`SchedulerConfig`/`DataConfig`/`SystemConfig`/`GenerationConfig`/`WandbConfig`），`from_json`/`to_json`/`get_torch_dtype`
- 模型：`SnailModel`、`SnailBlock`、`MultiHeadSelfAttention`、`RotaryPositionalEmbedding`、`PWFFN`、`init_model`，`SnailModel.generate`/`chat`
- 数据：`PretrainDataset`、`SFTDataset`、`get_dataloader`
- 训练函数：`cross_entropy_loss`、`cosine_schedule`、`gradient_clipping`、`silu`、`softmax`、`scaled_dot_product_attention`
- 训练脚本：`train_lm`/`train_sft`、`save_checkpoint`/`load_checkpoint`
- 推理：`generate_text`、`get_tokenizer`
- 工具：`setup_seed`、`read_memmap_data`、`LossMonitor`、`console`

#### Scenario: 函数可定位
- **WHEN** 读者在文档中查看任一关键类/函数
- **THEN** 文档以可点击 `file:///` 链接指向其在源码中的定义位置（文件 + 行号区间）

### Requirement: 依赖关系说明
文档 SHALL 在 README 中给出：
- 第三方依赖清单（torch、numpy、transformers、einops、jaxtyping、rich、matplotlib、wandb、datasets、tqdm）及其用途
- 模块间内部依赖关系（如 `model.py` 依赖 `config.py` 与 `debug.py`；`train_lm.py` 依赖 `model`/`dataset`/`functions`/`util`/`config`/`debug`）

#### Scenario: 依赖可读
- **WHEN** 读者查看依赖关系章节
- **THEN** 能清楚区分"外部第三方依赖"与"项目内部模块依赖"，并理解每个依赖的用途

### Requirement: 项目运行方式
`05-how-to-run.md` SHALL 给出从零到可对话的完整可执行步骤：
1. 安装依赖（`pip install -e .` + 第三方依赖）
2. 生成默认配置（`python scripts/generate_config.py`）
3. 预训练数据预处理（`python scripts/data_tokenize.py --tokenizer_path ... --data_path ...`）
4. SFT 数据预处理（`python scripts/preprocess_sft.py --data_path ... --tokenizer_root ...`）
5. 预训练（`python scripts/train_lm.py --config config.json`）
6. SFT 微调（`python scripts/train_sft.py --config_path config.json`）
7. 从检查点提取模型（`python scripts/get_model_from_checkpoint.py --checkpoint ...`）
8. 推理生成（运行 `tests/test_lm.py` 或调用 `generate_text`/`model.chat`）

#### Scenario: 可照此运行
- **WHEN** 读者按 `05-how-to-run.md` 顺序执行命令
- **THEN** 每一步均能对应到 `scripts/` 下真实存在的脚本及其参数

## MODIFIED Requirements
无（本项目首个 spec，无既有需求需修改）

## REMOVED Requirements
无
