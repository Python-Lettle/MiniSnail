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
           -> SFT                  (见 legacy/main 分支)
           -> DPO                  (见 legacy/main 分支)
```

> 预训练相关代码与工具位于本仓库 `main` 分支；SFT 与 DPO 的训练脚本在 `legacy/main` 分支。

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
pip install -e .
```

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

建议将数据集放在 `./dataset` 目录下。

### Step 3：预训练

1. 生成配置文件（或直接编辑 `config.json`）：

```bash
python scripts/get_config.py
```

2. 运行训练脚本。数据路径、模型保存目录与训练/验证划分比例通过命令行参数传入：

```bash
python trainer/train_pretrain.py \
    --config ./config.json \
    --data_path ./dataset/full/pretrain_t2t.jsonl \
    --save_model_dir ./output/new_pretrain \
    --train_ratio 0.95
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

### Step 4：模型评测

- 困惑度（perplexity）评测：遍历模型目录下所有权重，给出平均 loss / perplexity 及显著性检验：

```bash
python scripts/eval_perplexity.py
```

- 自建提示词测试集上的生成式评测：逐题生成并保存完整输出，便于人工判读：

```bash
python scripts/eval_generation.py
```

### Step 5：SFT 与 DPO

SFT 与 DPO 的训练脚本位于 `legacy/main` 分支，本分支暂不包含对应实现。
