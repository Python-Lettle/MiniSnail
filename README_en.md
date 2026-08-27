# MiniSnail

MiniSnail is a lightweight language model project. Its goal is to **simulate the core capabilities of large models with far fewer parameters, and walk through the entire training pipeline of large language models** — from data and pre-training, to instruction fine-tuning (SFT) and preference alignment (DPO).

## Project Features

- **Small** — Smaller model size
- **Efficient** — More efficient training and inference
- **Capable** — Retain as much of the core capabilities of language models as possible
- **Experimental** — Explore implementation approaches for small language models
- **Simple** — Keep the code structure simple, easy to understand and modify

## Project Progress

```
Raw   Data -> MiniMind Tokenizer   (vocab_size = 6400)
JSONL Data -> Pre-training         (Completed)
           -> SFT                  (See the legacy/main branch)
           -> DPO                  (See the legacy/main branch)
```

> Pre-training related code and tools live in the `main` branch of this repository; the training scripts for SFT and DPO are in the `legacy/main` branch.

## Quick Start

### Operating Environment

- OS: Windows 11
- CPU: 11th Gen Intel(R) Core(TM) i7-11800H @ 2.30GHz
- RAM: 16 GB
- GPU: NVIDIA GeForce RTX 3060 Laptop (6GB)
- CUDA: 13.2
- Python: 3.12.9

### Step 1: Clone and Install

```bash
git clone https://github.com/Python-Lettle/MiniSnail.git
cd MiniSnail
pip install -e .
```

### Step 2: Dataset

The project uses pre-training data in **JSONL** format, one sample per line:

```
{"text": "如何才能摆脱拖延症？治愈拖延症并不容易，但以下建议可能有所帮助。"}
{"text": "清晨的阳光透过窗帘洒进房间，桌上的书页被风轻轻翻动。"}
{"text": "Transformer 通过自注意力机制建模上下文关系，是现代大语言模型的重要基础结构。"}
```

The data is lazily loaded row by row via `LazyPretrainDataset` (only the row-offset index is built; the data is not fully read into memory), so **there is no need to pre-tokenize it into bin files**.

SFT data follows a conversation format:

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

It is recommended to place the dataset in the `./dataset` directory.

### Step 3: Pre-training

1. Generate the configuration file (or edit `config.json` directly):

```bash
python scripts/get_config.py
```

2. Run the training script. The data path, the model save directory, and the train/validation split ratio are passed as command-line arguments:

```bash
python trainer/train_pretrain.py \
    --config ./config.json \
    --data_path ./dataset/full/pretrain_t2t.jsonl \
    --save_model_dir ./output/new_pretrain \
    --train_ratio 0.95
```

- After an interruption, you can resume training via `training.use_checkpoint` / `training.from_checkpoint` in `config.json` (the checkpoint contains the model, optimizer, GradScaler, and the complete RNG state);
- If `training.use_wandb` is enabled, run `wandb login` first.

3. Test generation (specify the weight file with `--model`; the number of steps is encoded in the file name):

```bash
python tests/test_pretrain_lm.py --model ./model/new_pretrain/pretrain_lm_XXXX.pt
```

> Note: During pre-training, every sequence starts with `<|im_start|>`. The test script prepends this prefix before generation to avoid empty outputs or degenerate repetitions.

4. Export the raw model weights from a checkpoint. A training checkpoint also contains the optimizer, scaler, and other states, so the weights need to be exported separately:

```bash
python scripts/get_pretrain_model_from_cpt.py \
    --checkpoint ./output/new_pretrain/checkpoint.pt \
    --output ./model/new_pretrain
```

### Step 4: Model Evaluation

- Perplexity evaluation: iterate over all weights in the model directory, reporting average loss / perplexity together with a significance test:

```bash
python scripts/eval_perplexity.py
```

- Generative evaluation on a self-built prompt test set: generate and save the full outputs for each prompt, making them easy to review manually:

```bash
python scripts/eval_generation.py
```

### Step 5: SFT and DPO

The training scripts for SFT and DPO live in the `legacy/main` branch; this branch does not include them yet.
