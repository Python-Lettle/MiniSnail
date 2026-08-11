# 介绍

Lettle 的小型语言模型，旨在用参数量远少于大型模型的方式，模拟出大型模型的所有能力。



# 特点

- **Small** — 更小的模型规模

- **Efficient** — 更高效的训练与推理
- **Capable** — 尽可能保留语言模型的核心能力
- **Experimental** — 用于探索小型语言模型的实现方式
- **Simple** — 保持代码结构简单、易于理解和修改



# 快速开始

## 运行环境

- OS: Windows 11
- CPU: 11th Gen Intel(R) Core(TM) i7-11800H @ 2.30GHz
- RAM: 16 GB
- GPU: NVIDIA GeForce RTX 3060 Laptop (6GB)
- CUDA==13.2
- Python==3.12.9



## Step 1: 下载项目

使用 git 进行 clone 操作:

```bash
git clone https://github.com/Python-Lettle/MiniSnail.git
cd MiniSnail
```

安装项目：

```bash
pip install -e .
```



## Step 2: 数据集

确保你有如下格式的数据集：

- 预训练数据集:

```
{"text": "如何才能摆脱拖延症？治愈拖延症并不容易，但以下建议可能有所帮助。"}
{"text": "清晨的阳光透过窗帘洒进房间，桌上的书页被风轻轻翻动。"}
{"text": "Transformer 通过自注意力机制建模上下文关系，是现代大语言模型的重要基础结构。"}
```

- SFT 数据集:

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

建议你将数据集存放在目录 `./data` 下



## Step 3: 预处理数据集

在项目根目录下运行如下脚本：

```bash
python scripts/data_tokenize.py --tokenizer_path "./model/minimind" --data_path "./data/pretrain.jsonl" --train_output_path "./data/train_dataset.bin" --valid_output_path "./data/valid_dataset.bin" --train_ratio 0.95 --chunk_size 2000
```



## Step 4: 预训练

在 `config.json` 中编辑你所需要用到的参数，你可以通过运行下面的脚本生成这个配置文件：

```bash
python scripts/generate_config.py
```

配置好 wandb：

```bash
wandb login
```

运行训练脚本：

```bash
python scripts/train_lm.py --config config.json
```

你可以通过如下脚本测试你训练的模型：

```bash
python tests/test_lm.py
```



## Step 5: SFT

运行训练脚本：

```bash
python scripts/train_lm.py --config config.json
```

同样的，你可以通过如下脚本测试你训练的模型：

```bash
python tests/test_sft_lm.py
```

