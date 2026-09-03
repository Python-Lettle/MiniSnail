# MiniSnail

MiniSnail 是一个轻量级语言模型项目，目标是**用远少于大型模型的参数量，模拟大模型的核心能力，并完整走通大语言模型的训练全流程**——从数据、预训练，到指令微调（SFT）与偏好对齐（DPO）。

## 项目特点

- **Small** — 更小的模型规模
- **Efficient** — 更高效的训练与推理
- **Capable** — 尽可能保留语言模型的核心能力
- **Experimental** — 用于探索小型语言模型的实现方式
- **Simple** — 保持代码结构简单、易于理解和修改

## 项目进展

```
Raw   Data -> MiniMind Tokenizer   (vocab_size = 6400)
JSONL Data -> Pre-training         (已完成)
           -> SFT                  (已完成)
           -> DPO                  (已完成)
```

## 快速开始

### 运行环境

- OS：Windows 11
- CPU：11th Gen Intel(R) Core(TM) i7-11800H @ 2.30GHz
- RAM：16 GB
- GPU：NVIDIA GeForce RTX 3060 Laptop（6GB）
- CUDA：13.2
- Python：3.12.9

### Step 1：克隆与安装

```bash
git clone https://github.com/Python-Lettle/MiniSnail.git
cd MiniSnail
python -m venv .venv
```

Windows PowerShell：

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
```

项目固定使用 Python 3.12 和已验证的直接依赖版本。`.[test]` 会额外安装自动化测试依赖。
如需复现项目训练时的 CUDA 12.6 环境，请在安装项目前先安装对应的 PyTorch wheel：

```powershell
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -e ".[compile,test]"
```

安装后先运行不需要 GPU、数据集或模型权重的快速回归测试：

```powershell
python -m pytest
```

`tests/unit` 是 pytest 自动化回归测试；`tests/test_pretrain_lm.py` 等原有文件是需要模型权重和人工输入的交互式评测脚本，不会被 pytest 收集。

### Step 2：数据集

项目使用 **JSONL** 格式的预训练数据，每行一条样本：

```
{"text": "如何才能摆脱拖延症？治愈拖延症并不容易，但以下建议可能有所帮助。"}
{"text": "清晨的阳光透过窗帘洒进房间，桌上的书页被风轻轻翻动。"}
{"text": "Transformer 通过自注意力机制建模上下文关系，是现代大语言模型的重要基础结构。"}
```

数据通过 `LazyPretrainDataset` 按行懒加载（仅建立行偏移索引，不整读进内存），因此**无需预先分词成 bin 文件**。

SFT 数据为对话格式：

```json
{
    "conversations": [
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "你好！"},
        {"role": "user", "content": "再见"},
        {"role": "assistant", "content": "再见！"}
    ]
}
```

DPO 数据每行一条偏好样本（`chosen` 与 `rejected` 共享相同的提问，仅回复不同）：

```json
{
    "chosen":   [{"role": "user", "content": "问题"}, {"role": "assistant", "content": "更好的回复"}],
    "rejected": [{"role": "user", "content": "问题"}, {"role": "assistant", "content": "更差的回复"}]
}
```

建议将数据集放在 `./dataset` 目录下。

### 对话协议

MiniSnail 不直接使用 tokenizer 自带的 Qwen/MiniMind `chat_template`，因为该模板会注入当前模型未训练的 thinking/tool 标记。训练与推理统一使用 `src/minisnail/chat_protocol.py` 中的 `minisnail-chat-v1`：

```text
<|im_start|>user
问题<|im_end|>
<|im_start|>assistant
回答<|im_end|>
```

- SFT 只监督 assistant 正文和 `<|im_end|>`；角色头、system/user/tool 消息及消息分隔符均为上下文；
- DPO 的 prompt 与推理 prompt 使用同一个编码函数，completion 同样只包含 assistant 正文和 `<|im_end|>`；
- `model.chat()`、自动生成评测和交互式测试使用完全相同的 assistant 生成起点。

`sft_meta.json` 会记录协议版本。旧版 SFT 分片缺少该字段，训练脚本会拒绝加载，必须重新预处理，避免新旧标签语义静默混用。

### Step 3：预训练

1. 生成配置文件（或直接编辑 `config.json`）：

```bash
python scripts/get_config.py
```

2. 运行训练脚本。数据路径和模型保存目录通过命令行参数传入；固定验证集数量由 `config.json` 中的 `training.valid_samples` 指定：

```bash
python trainer/train_pretrain.py \
    --config ./config.json \
    --data_path ./dataset/full/pretrain_t2t.jsonl \
    --save_model_dir ./output/new_pretrain
```

- 训练中断后，可通过 `config.json` 中的 `training.use_checkpoint` / `training.from_checkpoint` 断点续训（checkpoint 内含模型、优化器、GradScaler 及完整的 RNG 状态）；
- 若启用 `training.use_wandb`，请先执行 `wandb login`。

3. 测试生成效果（`--model` 指定权重文件，步数体现在文件名中）：

```bash
python tests/test_pretrain_lm.py --model ./model/new_pretrain/pretrain_lm_XXXX.pt
```

> 提示：预训练时每个序列都以 `<|im_start|>` 开头，测试脚本在生成前会自动补上该前缀，避免模型产生空输出或退化为重复字符。

4. 从断点导出纯模型权重。训练 checkpoint 除权重外还包含优化器、缩放器等状态，需单独导出：

```bash
python scripts/get_pretrain_model_from_cpt.py \
    --checkpoint ./output/new_pretrain/checkpoint.pt \
    --output ./model/new_pretrain
```

### Step 4：标准评测流程

评测时必须使用与权重对应的 `config.json`。不同模型横向比较时，应固定数据划分、seed、生成长度和解码方式，并使用 `--greedy`；采样结果只适合人工观察，不适合作为模型排名依据。

#### 4.1 快速回归测试

```bash
python -m pytest
```

该步骤验证模型前向、KV Cache、断点状态、数据集、训练—推理协议以及评测指标实现，不需要 GPU 或模型权重。

#### 4.2 预训练 PPL

```bash
python scripts/eval_perplexity.py \
    --config ./model/new_pretrain/config.json \
    --data_path ./dataset/full/pretrain_t2t.jsonl \
    --models_dir ./model/new_pretrain \
    --pattern "pretrain_lm_*.pt" \
    --output_path ./eval/history/eval_ppl/eval_ppl_score.json
```

验证集划分与预训练脚本一致。主指标是按 token 加权的语料级 `avg_loss` / `perplexity`；文档均值、95% CI 和配对差异用于判断 checkpoint 间的变化是否稳定。只能直接比较使用同一 tokenizer 和同一验证集的模型。

#### 4.3 预训练生成冒烟测试

```bash
python scripts/eval_generation.py \
    --config ./model/new_pretrain/config.json \
    --model_path ./model/new_pretrain/model_final.pt \
    --prompt_format pretrain \
    --greedy \
    --output_path ./eval/history/eval_generation/pretrain.json
```

`pretrain` 格式使用“BOS + 原始文本”，用于观察基础续写能力，不应拿它评测 SFT/DPO 对话权重。

#### 4.4 SFT / DPO 对话生成评测

```bash
python scripts/eval_generation.py \
    --config ./model/new_sft/config.json \
    --model_path ./model/new_sft/sft_final.pt \
    --prompt_format chat \
    --greedy \
    --output_path ./eval/history/eval_generation/sft.json

python scripts/eval_generation.py \
    --config ./model/new_dpo/config.json \
    --model_path ./model/new_dpo/dpo_new.pt \
    --prompt_format chat \
    --greedy \
    --output_path ./eval/history/eval_generation/dpo.json
```

输出包含完整回答、`reference_hit_rate` 和 `eos_stop_rate`。`reference_hit_rate` 只是固定题集的回归/冒烟指标，不等价于公开 benchmark 的准确率；开放题仍需人工检查。

#### 4.5 DPO 偏好评测

```bash
python scripts/eval_preference.py \
    --config ./model/new_dpo/config.json \
    --data_path ./dataset/dpo_eval.jsonl \
    --model_path ./model/new_dpo/dpo_new.pt \
    --reference_model_path ./model/new_sft/sft_final.pt \
    --num_samples 2000 \
    --output_path ./eval/history/eval_preference/dpo.json
```

可用 `python scripts/create_dpo_eval.py` 从现有数据确定性抽取并校验 2000 条流程测试样本。注意：如果源文件已经参与训练，这份抽样只能验证评测流程，不能代表严格的泛化结果。正式报告时，`dpo_eval.jsonl` 必须在 DPO 训练前划出并从训练集排除。重点观察 `dpo_accuracy`、`mean_implicit_reward_margin`，同时结合 `policy_chosen_win_rate` 与长度归一化 gap 排查回复长度偏置。

### Step 5：SFT 与 DPO

**SFT** 需先对对话数据做预处理，将变长对话编码为定长 int16 分片（训练时 mmap 读取，不整读进内存）：

```bash
python scripts/preprocess_sft_data.py \
    --data_path ./dataset/full/sft_t2t.jsonl \
    --output_dir ./dataset/full \
    --overwrite
```

> `minisnail-chat-v1` 修改了 assistant 标签边界。升级后必须重新生成全部 SFT 分片；不要把旧 `.npy` 与新分片混用。

训练时通过 `--data_dir` 指定分片目录，`training.from_weight` 应指向预训练导出的权重：

```bash
python trainer/train_sft.py \
    --config ./config.json \
    --data_dir ./dataset/full \
    --save_model_dir ./output/new_sft
```

预训练与 SFT 都会按 `system.seed` 固定划出 `training.valid_samples` 条验证样本，并在每次验证时完整复用同一验证集。

训练产物：`sft_best.pt`（验证集 loss 最低）与 `sft_final.pt`（训练结束权重）。

**DPO** 以 SFT 模型为基座（`training.from_weight` 指向 `sft_final.pt`），loss 初始值应接近 ln(2) ≈ 0.693（policy 与 reference 初始权重相同），训练中 margin 由负转正、accuracy 缓慢上升即正常：

```bash
python trainer/train_dpo.py \
    --config ./config.json \
    --data_path ./dataset/dpo.jsonl \
    --save_model_dir ./output/new_dpo
```

训练产物：`dpo_new.pt`。

> SFT 与 DPO 的断点续训方式与预训练一致：通过 `config.json` 中的 `training.use_checkpoint` / `training.from_checkpoint` 恢复（checkpoint 内含双模型/优化器/GradScaler 及完整 RNG 状态）。
