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
           -> SFT                  (Completed)
           -> DPO                  (In progress)
```

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
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
```

The project requires Python 3.12 and pins the direct dependency versions used by the verified training environment. The `test` extra installs the automated test dependency. To reproduce the CUDA 12.6 training environment, install the matching PyTorch wheel before installing the project:

```powershell
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -e ".[compile,test]"
```

After installation, run the fast regression suite. It does not require a GPU, dataset, or model weights:

```powershell
python -m pytest
```

`tests/unit` contains the pytest regression suite. Existing files such as `tests/test_pretrain_lm.py` are interactive evaluation scripts that require model weights and user input, so pytest does not collect them.

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

DPO data contains one preference sample per line (`chosen` and `rejected` share the same prompt, differing only in the response):

```json
{
    "chosen":   [{"role": "user", "content": "Question"}, {"role": "assistant", "content": "Better response"}],
    "rejected": [{"role": "user", "content": "Question"}, {"role": "assistant", "content": "Worse response"}]
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

**SFT** requires preprocessing the conversation data first, encoding variable-length conversations into fixed-length int16 shards (read via mmap during training, not fully loaded into memory):

```bash
python scripts/preprocess_sft_data.py \
    --data_path ./dataset/full/sft_t2t.jsonl \
    --output_dir ./dataset/full
```

For training, pass the shard directory via `--data_dir`; `training.from_weight` should point to the exported pre-training weights:

```bash
python trainer/train_sft.py \
    --config ./config.json \
    --data_dir ./dataset/full \
    --save_model_dir ./output/new_sft \
    --valid_ratio 0.005
```

Outputs: `sft_best.pt` (lowest validation loss) and `sft_final.pt` (final weights).

**DPO** uses the SFT model as its base (`training.from_weight` points to `sft_final.pt`). The initial loss should be close to ln(2) ≈ 0.693 (policy and reference start with identical weights); during training, the margin turning from negative to positive and accuracy rising slowly indicate normal progress:

```bash
python trainer/train_dpo.py \
    --config ./config.json \
    --data_path ./dataset/dpo.jsonl \
    --save_model_dir ./output/new_dpo
```

Output: `dpo_new.pt`.

> Checkpoint resume for SFT and DPO works the same way as pre-training: restore via `training.use_checkpoint` / `training.from_checkpoint` in `config.json` (the checkpoint contains the dual models, optimizer, GradScaler, and the complete RNG state).
