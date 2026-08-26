# MiniSnail

MiniSnail is a lightweight language model project, aiming to **simulate the core capabilities of large models with far fewer parameters, and complete the entire training process of large language models** - from data, pre-training, to instruction fine-tuning (SFT) and preference alignment (DPO).

## Project Features

- **Small** — Smaller model size
- **Efficient** — More efficient training and inference
- **Capable** — Try to retain the core capabilities of language models as much as possible
- **Experimental** — Used to explore the implementation methods of small language models
- **Simple** — Maintain a simple code structure, easy to understand and modify

## Project Progress

```
Raw Data -> MiniMind Tokenizer   (vocab_size = 6400)
-> Pre-training         (Completed)
-> SFT                  (See the legacy/main branch)
-> DPO                  (See the legacy/main branch)
```

The pre-training related code and tools are located in the `main` branch of this repository; the training scripts for SFT and DPO are in the `legacy/main` branch.

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

The project uses pre-trained data in the **JSONL** format, with one sample per line:

```
How can one get rid of procrastination? Healing from procrastination is not easy, but the following suggestions might be helpful. }
The morning sunlight filtered through the curtains and streamed into the room. The pages of the books on the table were gently turned by the wind. }
The Transformer model models the context relationships through the self-attention mechanism, which is an important foundational structure of modern large language models. }
```

The data is lazily loaded row-by-row through `LazyPretrainDataset` (only establishing row offset indices, without fully reading into memory), so **there is no need to pre-phrase into bin files**.

SFT data is in conversation format:

```json
{
"conversations": [
"Hello!" },
"Goodbye!" }
]
}
```

It is recommended to place the dataset in the `./dataset` directory.

### Step 3: Pre-training

1. Generate the configuration file (or directly edit `config.json`):

```bash
python scripts/get_config.py
```

2. Run the training script. The data path, model save directory, and the ratio of training/validation are passed as command-line parameters:

```bash
python trainer/train_pretrain.py \
--config ./config.json \
--data_path ./dataset/full/pretrain_t2t.jsonl \
--save_model_dir ./output/new_pretrain \
--train_ratio 0.95
```

- After training is interrupted, you can resume training using `config.json`'s `training.use_checkpoint` / `training.from_checkpoint` (the checkpoint contains the model, optimizer, GradScaler, and complete RNG state);
- If `training.use_wandb` is enabled, please first execute `wandb login`.

3. Test the generated results (specify the weight file with `--model`, and the number of steps is reflected in the file name):

```bash
python tests/test_pretrain_lm.py --model ./model/new_pretrain/pretrain_lm_XXXX.pt
```

> Note: During pre-training, each sequence starts with `

```bash
python scripts/get_pretrain_model_from_cpt.py \
--checkpoint ./output/new_pretrain/checkpoint.pt \
--output ./model/new_pretrain
```

### Step 4: Model Evaluation

- Perplexity Evaluation: Traverse all the weights in the model directory and provide the average loss / perplexity along with a significance test:

```bash
python scripts/eval_perplexity.py
```

- Generative evaluation on the self-built prompt test set: Generate and save complete outputs for each question, facilitating manual interpretation:

```bash
python scripts/eval_generation.py
```

### Step 5: SFT and DPO

The training scripts for SFT and DPO are located in the `legacy/main` branch. Currently, this branch does not include the corresponding implementations.

