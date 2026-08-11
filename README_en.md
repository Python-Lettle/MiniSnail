# Introduction

Lettle's small language model is designed to simulate all the capabilities of a large model with a much smaller parameter model.

[中文](./README.md) | English

# Features

- **Small** - Smaller model size

- **Efficient** - More efficient training and inference

- **Capable** - Retaining the core capabilities of the language model as much as possible

- **Experimental** - For exploring the implementation methods of small language models

- **Simple** - Maintaining a simple code structure that is easy to understand and modify



# Quick Start

## Environment

- OS: Windows 11
- CPU: 11th Gen Intel(R) Core(TM) i7-11800H @ 2.30GHz
- RAM: 16 GB
- GPU: NVIDIA GeForce RTX 3060 Laptop (6GB)
- CUDA==13.2
- Python==3.12.9



## Step 1: Clone

Clone the project:

```bash
git clone https://github.com/Python-Lettle/MiniSnail.git
cd MiniSnail
```

Install the project:

```bash
pip install -e .
```



## Step 2: Dataset

Make sure you have a dataset in a format similar to the following:

- Pretrain Dataset:

```
{"text": "如何才能摆脱拖延症？治愈拖延症并不容易，但以下建议可能有所帮助。"}
{"text": "清晨的阳光透过窗帘洒进房间，桌上的书页被风轻轻翻动。"}
{"text": "Transformer 通过自注意力机制建模上下文关系，是现代大语言模型的重要基础结构。"}
```

- SFT Dataset:

```
{
    "conversations": [
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "你好！"},
        {"role": "user", "content": "再见"},
        {"role": "assistant", "content": "再见！"}
    ]
}
{
    "conversations": [
        {"role": "system", "content": "# Tools ...", "tools": "[...]"},
        {"role": "user", "content": "把'你好世界'翻译成english"},
        {"role": "assistant", "content": "", "tool_calls": "[{\"name\":\"translate_text\",\"arguments\":{\"text\":\"你好世界\",\"target_language\":\"english\"}}]"},
        {"role": "tool", "content": "{\"translated_text\":\"Hello World\"}"},
        {"role": "assistant", "content": "Hello World"}
    ]
}
```

It is recommended to save these data sets to the directory `./data`



## Step 3: Preprocess the dataset

Navigate to the project root directory and run the following script:

```bash
python scripts/data_tokenize.py --tokenizer_path "./model/minimind" --data_path "./data/pretrain.jsonl" --train_output_path "./data/train_dataset.bin" --valid_output_path "./data/valid_dataset.bin" --train_ratio 0.95 --chunk_size 2000
```



## Step 4: Pre-training

Edit the parameters you need to use in `config.json`, you can generate the file by:

```bash
python scripts/generate_config.py
```

Configure your wandb:

```bash
wandb login
```

Run the training script:

```bash
python scripts/train_lm.py --config config.json
```

You can test your model by:

```bash
python tests/test_lm.py
```



## Step 5: SFT

Run the script:

```bash
python scripts/train_lm.py --config config.json
```

Also, you can test your model by:

```bash
python tests/test_sft_lm.py
```

