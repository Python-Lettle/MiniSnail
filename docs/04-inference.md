> [返回首页](README.md) | [上一篇: 训练流程](03-training.md) | [下一篇: 项目运行方式](05-how-to-run.md)

# 训练流程④：推理与生成

## 1. 推理入口概览

MiniSnail 的推理链路分为三层，自底向上逐层封装：

1. 底层：[SnailModel.generate](file:///workspace/src/minisnail/model.py#L220-L267) —— token 级别的自回归采样生成，支持温度缩放、top-k 截断、重复惩罚与贪婪解码，是所有推理的最终落点。
2. 中层：[SnailModel.chat](file:///workspace/src/minisnail/model.py#L270-L297) —— 基于 chat template 的多轮对话封装，在 generate 之上拼装对话上下文。
3. 顶层：[generate_text](file:///workspace/src/minisnail/generate.py#L8-L43) —— 一次性便利函数，完成「编码 prompt → 调 generate → 解码 → 拼接展示」全流程。

## 2. SnailModel.generate（自回归采样生成）

参考 [generate](file:///workspace/src/minisnail/model.py#L220-L267)。该方法被 `@torch.no_grad()` 装饰，整个生成过程不计算梯度。方法签名为：

```python
def generate(self, X,
             max_tokens=8192,
             temperature=0.85,
             repetition_penalty=1.2,
             top_k=50,
             eos_token_id=2,
             do_sample=True,
             skip_prompt=True):
```

参数说明：`X` 为 prompt 的 token id 张量；`max_tokens` 为最大生成长度；`temperature` 控制采样锐度；`repetition_penalty` 对已生成 token 打折；`top_k` 限定候选集大小；`eos_token_id` 触发提前停止；`do_sample` 切换采样/贪婪；`skip_prompt` 决定返回值是否仅含新生成部分。

逐步流程：

- 若 `X` 为 1 维，调用 `X.unsqueeze(0)` 升为 `[1, seq]`；随后 `X = X.long()` 转为 long 类型；记录 `original_length = X.size(-1)`，供返回时切片使用。
- 进入 `for _ in range(max_tokens)` 循环，每轮执行：
  1. 滑动窗口截断：`X = X[:, -self.config.model.context_length:]`，只保留末尾 `context_length`（默认 512）个 token，避免超过 RoPE 缓存长度。
  2. 前向计算：`logits = self.forward(X)`，取最后一个位置的 logits 并做温度缩放 `next_token_logits = logits[:, -1] / temperature`。
  3. 采样分支（`do_sample=True` 时）：
     - 若 `repetition_penalty > 1.0`，遍历已生成 token（`X[0].tolist()`），将它们的 logits 除以 `repetition_penalty` 做惩罚（即「打折」）。
     - 若 `top_k` 非零，用 `torch.topk` 取最大的 `min(top_k, next_token_logits.size(-1))` 个 logits，以其中最小值作为阈值，把低于阈值的 logits 用 `masked_fill(..., float("-inf"))` 屏蔽。
     - `probs = F.softmax(next_token_logits, dim=-1)` 归一化，再 `next_token_id = torch.multinomial(probs, 1)` 抽样。
  4. 贪婪分支（`do_sample=False`）：`next_token_id = next_token_logits.argmax(dim=-1, keepdim=True)`。
  5. 若 `eos_token_id is not None` 且 `next_token_id.item() == eos_token_id`，跳出循环。
  6. `X = torch.cat((X, next_token_id), dim=-1)` 把新 token 追加到序列末尾。
- 返回值：若 `skip_prompt=True`，返回 `X[:, original_length:]`（仅新生成的 token）；否则返回完整 `X`。

注意：方法签名里的默认值（`max_tokens=8192`、`temperature=0.85`、`top_k=50`）与 [GenerationConfig](file:///workspace/src/minisnail/config.py#L71-L79) 的默认值（`max_tokens=512`、`temperature=0.8`、`top_k=40`）并不一致。顶层 `generate_text` 在调用时显式传入 `GenerationConfig` 的值，因此实际推理使用的是配置文件的参数；只有直接调用 `model.generate` 而不传参时才会用上方法签名的默认值。`repetition_penalty=1.2` 与 `eos_token_id=2` 在两处保持一致。

## 3. SnailModel.chat（对话生成）

参考 [chat](file:///workspace/src/minisnail/model.py#L270-L297)。该方法是多轮对话的封装：

- 将 `history`（默认空列表）作为已有消息，追加当前 `{"role": "user", "content": message}`。
- 调用 `tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)` 把对话渲染为字符串，但**不**让模板自动添加 generation prompt。
- 随后手动在末尾拼接 `"<|im_start|>assistant\n"`，作为 assistant 回复的起始标记；但**不**追加模板在 `add_generation_prompt=True` 时本会注入的 `<think>` 推理标签，让模型直接生成正文。
- 将 prompt 经分词器编码为 `input_ids`，并 `.to(self.config.system.device)` 移动到目标设备。
- 调用 `self.generate(input_ids, eos_token_id=tokenizer.eos_token_id, **kwargs)`，其余生成参数通过 `kwargs` 透传。
- 对 `output_ids[0]` 调用 `tokenizer.decode(..., skip_special_tokens=True)` 得到可读的 `response` 文本并返回。

通过外部维护 `history` 列表即可实现多轮对话：每轮把上一轮的 user/assistant 消息追加进去再调用 `chat`。注意 `chat` 内部 `append` 的是当前 user 消息，并未把模型回复写回 history，调用方需自行把回复加入 `history` 以维持上下文。

## 4. generate_text（顶层封装）

参考 [generate_text](file:///workspace/src/minisnail/generate.py#L8-L43)。这是一站式推理入口：

- 解析设备：`device = torch.device(config.system.device)`（除非显式传入 `device`）；随后 `model.to(device)`、`model.eval()`。
- 编码 prompt：`prompt_ids = tokenizer.encode(prompt)`；`prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)`。
- 在 `with torch.no_grad():` 块中调用：

  ```python
  output_ids_tensor = model.generate(
      prompt_tensor,
      max_tokens=config.generation.max_tokens,
      temperature=config.generation.temperature,
      top_k=config.generation.top_k,
      eos_token_id=config.tokenizer.eos_token_id,
      repetition_penalty=config.generation.repetition_penalty,
  )
  ```

- 把输出张量转回 list：`output_ids = output_ids_tensor[0].cpu().numpy().tolist()`。
- **拼接逻辑（关键细节）**：由于 `generate` 的 `skip_prompt` 默认为 `True`，`output_ids_tensor` 只包含新生成的 token，并不含原始 prompt。因此这里手动做 `full_ids = prompt_ids + output_ids`，再 `text = tokenizer.decode(full_ids)`，得到「prompt + 生成内容」的完整文本用于展示。
- 通过 `console.print` 打印 prompt、生成的文本，并输出 `Generation time: {end - start:.2f} seconds`。

> **微妙之处（务必留意）**：`generate_text` 并没有向 `model.generate` 显式传入 `skip_prompt`，于是沿用方法默认值 `True`，`output_ids` 只是新生成部分；正因如此，函数内部才需要把 `prompt_ids` 重新拼回去再 decode。如果将来有人把 `skip_prompt` 改成 `False`（返回完整序列），这里的 `prompt_ids + output_ids` 就会重复一遍 prompt，导致展示文本出现两份提示词。这是 `generate` 与 `generate_text` 之间隐含的契约，修改任一方时务必同步。

## 5. 分词器与推理配置

- [get_tokenizer](file:///workspace/src/minisnail/tokenizer.py#L4-L7)：直接 `AutoTokenizer.from_pretrained(config.tokenizer.tokenizer_root)` 加载分词器，默认路径 `./model/minimind`。
- [GenerationConfig](file:///workspace/src/minisnail/config.py#L71-L79) 默认值：`model_path="./output/model_best.pt"`、`max_tokens=512`、`temperature=0.8`、`top_k=40`、`device="cuda"`、`repetition_penalty=1.2`。注意 `device` 字段在 `generate_text` 中实际取的是 `config.system.device` 而非 `config.generation.device`，`GenerationConfig.device` 在推理路径上并未被读取。
- [TokenizerConfig](file:///workspace/src/minisnail/config.py#L7-L14)：`eos_token_id=2`（对应 `<|im_end|>`），用于在 `generate` 中提前终止生成；`bos_token_id=1`（对应 `<|im_start|>`）；`vocab_size=6400`。

## 6. 推理示例（tests/test_lm.py）

参考 [test_lm.py](file:///workspace/tests/test_lm.py)。这是一段预训练后的推理冒烟测试，在 `__main__` 中：

- `config = SnailConfig.from_json("./config.json")` 读取配置。
- `tokenizer = get_tokenizer(config)` 加载分词器。
- `model = init_model(config)` 构造空模型。
- `checkpoint = load_checkpoint("./output/checkpoint.pt")`，再 `model.load_state_dict(checkpoint["model_state_dict"])` 载入预训练权重。
- `model.eval()`、`model.to(torch.device(config.system.device))`。
- 定义中文 prompt：`"我开发了一款小型语言模型，但是它的重复率太高"`。
- 调用 `generate_text(model, tokenizer, prompt, config, device=...)` 完成生成并打印。

这是训练完成后验证模型是否正常产出的最小可用流程；若输出含合理中文且重复惩罚生效，即说明预训练流程闭环。

## 7. 生成超参速查表

| 参数 | generate 默认 | GenerationConfig 默认 | 含义 |
| --- | --- | --- | --- |
| max_tokens | 8192 | 512 | 最大生成 token 数 |
| temperature | 0.85 | 0.8 | 采样温度，越大越随机 |
| top_k | 50 | 40 | 每步候选集大小 |
| repetition_penalty | 1.2 | 1.2 | 已生成 token 的 logits 惩罚系数 |
| eos_token_id | 2 | 2 | 命中则提前停止（`<|im_end|>`） |
| do_sample | True | — | True 采样 / False 贪婪 |
| skip_prompt | True | — | True 仅返回新 token / False 返回完整序列 |

说明：表中「—」表示 `GenerationConfig` 不含该字段，由 `generate` 方法签名直接提供默认值。`generate_text` 调用 `model.generate` 时只显式传入 `max_tokens`、`temperature`、`top_k`、`eos_token_id`、`repetition_penalty` 五项，`do_sample` 与 `skip_prompt` 均沿用方法默认值（`True`）。
