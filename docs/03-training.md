> [返回首页](README.md) | [上一篇: 模型架构](02-model.md) | [下一篇: 推理与生成](04-inference.md)

# 训练流程③：数据加载、训练函数、预训练、SFT、检查点与实验追踪

本篇覆盖 MiniSnail 训练管线的第三阶段：从原始 token 流到可加载的检查点与实验日志，逐层拆解数据加载、训练核心函数、预训练脚本（train_lm）、SFT 微调脚本（train_sft）、检查点机制以及实验追踪与监控。两个训练脚本共享同一套配置体系与核心函数，但在损失函数与数据格式上有关键差异，阅读时请特别注意区分。

---

## 1. 数据集与 DataLoader

参考 [dataset.py](file:///workspace/src/minisnail/dataset.py)。本模块定义了两种数据集类型：预训练用的流式 memmap 数据集与 SFT 用的预处理 npy 数据集，并由统一的 DataLoader 构造器包装。

### PretrainDataset

[PretrainDataset](file:///workspace/src/minisnail/dataset.py#L36-L78) 采用惰性内存映射（lazy memmap）加载，避免将整份预训练语料一次性读入内存。构造时通过 `np.memmap(data_path, dtype=np.int32, mode='r')` 把磁盘上的 int32 一维 token 数组映射为内存视图，随后计算样本数：

`num_samples = len(data) // (block_size + 1)`

即按 `block_size + 1` 长度切分为互不重叠（non-overlapping）的块。`__getitem__(index)` 取第 `index` 块连续的 `block_size + 1` 个 token 作为 chunk，并返回 `(x, y)` 对：

- `x = chunk[:-1]`，长度为 `block_size`，作为模型输入；
- `y = chunk[1:]`，长度为 `block_size`，向右错一位，作为下一 token 预测（next-token prediction）的标签。

chunk 在返回前会被 `.copy()` 转成 int64 张量，因为 memmap 切片返回的是子视图而非连续内存。由于每个 token 在单个 epoch 内恰好被访问一次，顺序的随机化完全交给 `DataLoader.shuffle`，dataset 自身不做随机采样。

### SFTDataset

[SFTDataset](file:///workspace/src/minisnail/dataset.py#L82-L92) 用于监督微调。它加载已经预处理好的两个 npy 文件：`np.load(input_path)` 得到 input_ids，`np.load(labels_path)` 得到 labels。`__getitem__` 直接将对应下标的样本转成 long 张量并返回 `(input_ids, labels)` 二元组。labels 中通常用 `-100` 标记被忽略的位置（如 prompt 部分），由损失函数的 `ignore_index` 处理，从而只对 assistant 回复计算损失。

### get_dataloader

[get_dataloader](file:///workspace/src/minisnail/dataset.py#L8-L32) 是为 `PretrainDataset` 服务的 DataLoader 构造器。`num_workers` 默认为 `cpu_count // 2`（至少为 1）。固定启用的选项包括 `pin_memory=True` 与 `drop_last=True`；其中验证集调用时显式传入 `shuffle=False, drop_last=False` 以遍历全部样本。当 `num_workers > 0` 时，额外启用 `persistent_workers=True` 与 `prefetch_factor=4` 以减少 worker 重启开销并预取批次。

---

## 2. 训练核心函数（functions.py）

参考 [functions.py](file:///workspace/src/minisnail/functions.py)。这里集中了训练过程中会用到的损失、调度、梯度裁剪以及若干手写的算子实现。

### cross_entropy_loss

[cross_entropy_loss](file:///workspace/src/minisnail/functions.py#L38-L77) 是手写的交叉熵损失，仅用于预训练（train_lm）中。它采用数值稳定的 log_softmax 实现：先对 logits 沿最后一维减去最大值（`logits - logits_max`），再计算 `log_sum_exp = log(sum(exp(shifted_logits)))`，得到 `log_softmax = shifted_logits - log_sum_exp`。随后通过 `torch.gather` 沿类别维取出目标位置对应的 log 概率，再取负均值作为最终损失。该函数直接接收模型输出的完整 logits（含 batch 与 seq 维），不做标签错位——错位工作已由 `PretrainDataset` 在数据侧完成。

### cosine_schedule

[cosine_schedule](file:///workspace/src/minisnail/functions.py#L79-L110) 实现带 warmup 的余弦学习率调度，分三段：

1. `it < warmup_iters`：线性 warmup，返回 `it / warmup_iters * max_lr`；
2. `warmup_iters < it <= cosine_cycle_iters`：在 `max_lr` 与 `min_lr` 之间余弦退火，公式为 `min_lr + (max_lr - min_lr) * (1 + cos(...)) / 2`；
3. `it > cosine_cycle_iters`：恒定返回 `min_lr`。

默认参数（来自 SchedulerConfig）：`max_lr=0.0005`、`min_lr=0.00005`、`warmup_iters=600`、`cosine_cycle_iters=6000`。

### gradient_clipping

[gradient_clipping](file:///workspace/src/minisnail/functions.py#L113-L136) 实现按全局 L2 范数裁剪梯度。它先把所有非空参数的 `parameter.grad` 展平并 `torch.cat`，计算整体 L2 范数，再算 `scale = max_l2_norm / (norm + eps)`；当 `norm > max_l2_norm` 时，对每个参数的 grad 原地执行 `mul_(scale)` 完成缩放。

### scaled_dot_product_attention 与辅助函数

[scaled_dot_product_attention](file:///workspace/src/minisnail/functions.py#L16-L36) 是一个手写的 SDPA 实现，接收 Q/K/V 以及可选的 mask，通过 einsum 计算相似度、按 `sqrt(d_k)` 缩放、masked_fill、softmax、再加权求和。

> 注意：模型实际推理时使用的是 PyTorch 官方的 `F.scaled_dot_product_attention`，而非本文件中的这个手写版本。这里的手写实现主要用于教学与单元测试，并不进入生产前向路径。

附带两个辅助函数：[softmax](file:///workspace/src/minisnail/functions.py#L11-L14)（减最大值后归一化的数值稳定 softmax）与 [silu](file:///workspace/src/minisnail/functions.py#L8-L9)（`x * sigmoid(x)`）。

---

## 3. 预训练流程（scripts/train_lm.py）

参考 [train_lm.py](file:///workspace/scripts/train_lm.py)。这是从原始 token 流训练一个 base LM 的主脚本，使用上节的 `cross_entropy_loss`。

### 入口与初始化

入口函数为 [train_lm](file:///workspace/scripts/train_lm.py#L74-L303)，签名 `train_lm(config, wandb_run, checkpoint)`。初始化阶段（约 L75-L136）依次完成：

- `setup_seed(config.system.seed)` 固定随机性；
- 准备 train/valid 数据路径、save_model_dir 并 `os.makedirs`；
- 选取 `device`，调用 `config.get_torch_dtype()` 得到 `(model_dtype, amp_dtype)`，令 `use_amp = amp_dtype is not None`；
- 通过 `get_dataloader(block_size=context_length)` 构建 `train_loader` 与 `valid_loader`（验证集传入 `shuffle=False, drop_last=False`）；
- 经 `init_model` 构建模型；若设置了 `from_weight` 则加载已有权重；
- 构造优化器 `AdamW(lr, betas=(0.9, 0.95), eps=1e-8, weight_decay)`；
- 若传入 checkpoint，则恢复 `model_state_dict`、`optimizer_state_dict` 与 `global_step`。

同时创建两个 `LossMonitor(show_stats=False)`（headless）：`train_loss_monitor` 与 `valid_loss_monitor`。

### 训练循环

训练循环（约 L165-L264）以 epoch 为外层、以 `(inputs, targets)` 为内层。每个 step：

1. 把数据搬到 device，`global_step += 1`；
2. 在 `torch.autocast(amp)` 上下文中前向，`loss = cross_entropy_loss(logits, targets)`；
3. `scaled_loss = loss / accumulation_steps` 后 `backward()`；
4. 每达到 `accumulation_steps` 次：执行 `gradient_clipping`、把当前 `cosine_schedule(global_step, ...)` 写入每个 param_group 的 `lr`、`optimizer.step()`、`optimizer.zero_grad()`；
5. 把当前 loss 记入 `train_loss_monitor`，并向 wandb 上报 `train/loss` 与 `train/lr`；
6. 每隔 `print_interval`（默认 200）打印进度。

### 验证

每 `valid_interval`（默认 400）步触发验证（约 L231-L255）：切到 `model.eval()`，从 `valid_loader` 中随机采样最多 20 个 batch，对每个样本单独前向并计算 `cross_entropy_loss`，取均值。把 `valid/loss` 上报 wandb；若是新的最小 loss，则把 `model.state_dict()` 保存为 `model_best.pt`。验证结束后回到 `model.train()`。

### 异常处理与收尾

异常处理（约 L266-L285）覆盖 `AssertionError`、`KeyboardInterrupt` 与通用 `Exception` 三种情况：都会通过 `save_checkpoint` 落盘 `checkpoint.pt` 后 `return`，从而保住训练进度。

正常结束（约 L288-L303）：若有未消费的累积梯度（`has_pending_grads`），会再补一次裁剪与 `optimizer.step()`；然后保存 `model_new.pt`，并分别 `finalize` 出 `train_loss_curve.png` 与 `valid_loss_curve.png`。

### CLI

命令行入口（约 L305-L357）：`--config` 指定 JSON 配置文件路径，未提供时按 `config.json` → `DEFAULT_CONFIG` 顺序回退。若 `use_checkpoint` 为真，则通过 `load_checkpoint(from_checkpoint)` 恢复，并以 `id=checkpoint['wandb_id']`、`resume="allow"` 重启 wandb run；否则开一个全新的 wandb run。最终调用 `train_lm` 后 `run.finish()`。

---

## 4. SFT 微调流程（scripts/train_sft.py）

参考 [train_sft.py](file:///workspace/scripts/train_sft.py)。该脚本在预训练权重之上做监督微调，结构上与 train_lm 相似，但在数据、损失与恢复逻辑上有关键差异。

### 入口与初始化

入口为 [train_sft](file:///workspace/scripts/train_sft.py#L81-L311)，签名 `train_sft(config, run, checkpoint)`。初始化阶段（约 L82-L156）：`setup_seed` → 构造 `SFTDataset(input_ids_path, labels_path)` → 用原生 `DataLoader` 包装（`batch_size`、`shuffle=True`、`num_workers=0`、`pin_memory=False`、`drop_last=False`）→ `init_model` + 可选 `from_weight` → `AdamW` → 可选 checkpoint 恢复。

### 关键差异：损失函数

SFT 与预训练最重要的区别在于损失（约 L191-L197）：SFT 使用 PyTorch 官方的 `torch.nn.functional.cross_entropy`（**不是** functions.py 中的手写 `cross_entropy_loss`），并显式带上 `ignore_index=-100`：

```
loss = F.cross_entropy(
    logits[:, :-1, :].contiguous().view(-1, vocab_size),
    labels[:, 1:].contiguous().view(-1),
    ignore_index=-100,
)
```

这里在损失侧把 logits 与 labels 都做了一次错位（`[:, :-1]` 对 `[:, 1:]`），相当于让位置 t 的输出预测位置 t+1 的标签；同时 `ignore_index=-100` 使得 labels 中标记为 `-100` 的位置（通常对应 prompt / 用户输入部分）不参与损失，从而只对 assistant 回复计算梯度。验证阶段（约 L246-L276）使用完全一致的损失计算方式，并在新最小 loss 时保存 `sft_best.pt`。

### 恢复逻辑

恢复逻辑（约 L167-L184）按 step 粒度续训：

- `start_epoch = global_step // steps_per_epoch`：定位从哪个 epoch 开始；
- `start_step_in_epoch = global_step % steps_per_epoch`：定位该 epoch 内已完成的步数；
- 内层循环里 `if step < start_step_in_epoch: continue` 跳过已训练的步，从而实现 epoch 中途恢复。

注意：脚本在 L155 先用 `checkpoint['epoch']` 给 `start_epoch` 赋值，但随后 L168 又用 `global_step // steps_per_epoch` 重新覆盖，实际生效的是后者。

### 收尾与 CLI

正常结束（约 L294-L311）：补一次累积梯度后保存 `sft_new.pt`，并生成 `train_loss_curve.png` 与 `valid_loss_curve.png`。`KeyboardInterrupt` 时落盘 `checkpoint.pt`。

CLI（约 L313-L365）：注意 flag 名为 `--config_path`，与 train_lm 的 `--config` 不同。`use_checkpoint` 为真时同样以 `id + resume="allow"` 重启 wandb run。

---

## 5. 检查点机制

参考 [train_lm.py](file:///workspace/scripts/train_lm.py) 中的实现。

### save_checkpoint

[save_checkpoint](file:///workspace/scripts/train_lm.py#L16-L51) 把训练状态序列化到磁盘。落盘的 dict 包含五个键：`model_state_dict`、`optimizer_state_dict`、`global_step`、`epoch`、`wandb_id`（取自 `run.id`）。该函数在 train_sft.py 中有结构相同的副本（[save_checkpoint](file:///workspace/scripts/train_sft.py#L23-L57)）。

### load_checkpoint

[load_checkpoint](file:///workspace/scripts/train_lm.py#L53-L72) 通过 `torch.load(src, weights_only=False)` 读取并返回整个 dict。`weights_only=False` 是为了能够加载包含 wandb_id 等 Python 对象的状态。

### get_model_from_checkpoint.py

[get_model_from_checkpoint.py](file:///workspace/scripts/get_model_from_checkpoint.py) 是一个独立的小工具：从一个完整 checkpoint 中取出 `model_state_dict`，再单独保存为一个纯权重文件（默认输出到 `./output/model_from_checkpoint.pt`），便于直接加载做推理。CLI 接收 `--checkpoint`，默认值为 `./output/checkpoint.pt`。

### 续训流程

续训由 [TrainingConfig](file:///workspace/src/minisnail/config.py#L28-L42) 的 `use_checkpoint` 与 `from_checkpoint` 两个字段共同触发：脚本启动时若 `use_checkpoint=True`，则调用 `load_checkpoint(from_checkpoint)` 恢复状态，并以 `id=checkpoint['wandb_id']`、`resume="allow"` 重启对应的 wandb run，使日志曲线无缝接续。

---

## 6. 实验追踪与监控

### Wandb

脚本启动时调用 `wandb.init(entity, project, config)` 开启一个 run；训练中按 step 上报 `train/loss`、`train/lr`、`valid/loss`；结束时 `run.finish()`。续训时通过 `id + resume="allow"` 接续同一 run。相关配置见 [WandbConfig](file:///workspace/src/minisnail/config.py#L81-L86)：默认 `entity="lettle-hong"`、`project="MiniSnail"`、`id=None`（由 wandb 自动分配）。

### LossMonitor

[LossMonitor](file:///workspace/src/minisnail/debug.py#L44-L220) 是一个轻量级的 loss 监视器，按 step 记录 loss 序列与移动平均，并维护一组运行统计量（`min_loss`、`max_loss`、`sum_loss`、`sum_squared_loss`、`count`、`first_loss`）。核心方法：

- `add_loss(epoch, loss) -> bool`：记入一条样本，返回 `is_min_loss`（是否为新的最小 loss）。
- `finalize(save_path)`：调用 [plot_loss_curve_basic](file:///workspace/src/minisnail/debug.py#L9-L42) 把整条 loss 曲线渲染成 PNG 保存到 `save_path`。

在两个训练脚本中均以 `show_stats=False` 实例化（headless 模式），不启用 matplotlib 的交互窗口，只在结束时输出 PNG。

### setup_seed 与 console

[setup_seed](file:///workspace/src/minisnail/util.py#L8-L16) 同时 seed `random`、`numpy`、`torch` 以及 CUDA（含 `manual_seed_all`），并把 `cudnn.deterministic=True`、`cudnn.benchmark=False` 以追求可复现。[read_memmap_data](file:///workspace/src/minisnail/util.py#L18-L30) 封装了 `np.memmap(train_data_path, dtype=np.int32, mode="r")`，用于把预训练 token 二进制文件以内存映射方式加载为只读的一维 int32 数组（与 `PretrainDataset` 内部的 memmap 用法一致）。[console](file:///workspace/src/minisnail/debug.py#L7) 是基于 rich 的 `Console` 实例，负责训练过程中格式化打印，并在 util.py 中重新导出。

---

## 7. 训练配置速查

下表汇总 [TrainingConfig](file:///workspace/src/minisnail/config.py#L28-L42) 与 [SchedulerConfig](file:///workspace/src/minisnail/config.py#L44-L50) 的默认值，供快速查阅。

| 配置项 | 默认值 | 所属 | 说明 |
| --- | --- | --- | --- |
| epochs | 6000 | TrainingConfig | 总训练 epoch 数 |
| batch_size | 32 | TrainingConfig | 每个 batch 的样本数 |
| lr | 0 | TrainingConfig | 优化器初始学习率，运行时由 scheduler 覆盖 |
| betas | (0.9, 0.95) | TrainingConfig | AdamW 的 beta1/beta2 |
| weight_decay | 0.001 | TrainingConfig | AdamW 权重衰减 |
| valid_interval | 400 | TrainingConfig | 每隔多少 step 做一次验证 |
| gradient_clip | 1.0 | TrainingConfig | 梯度 L2 范数裁剪阈值 |
| accumulation_steps | 1 | TrainingConfig | 梯度累积步数 |
| print_interval | 200 | TrainingConfig | 每隔多少 step 打印一次进度 |
| from_weight | None | TrainingConfig | 可选的初始权重路径 |
| use_checkpoint | False | TrainingConfig | 是否从检查点续训 |
| from_checkpoint | None | TrainingConfig | 检查点文件路径 |
| max_learning_rate | 0.0005 | SchedulerConfig | 余弦调度峰值学习率 |
| min_learning_rate | 0.00005 | SchedulerConfig | 余弦调度最小学习率 |
| warmup_iters | 600 | SchedulerConfig | 线性 warmup 迭代数 |
| cosine_cycle_iters | 6000 | SchedulerConfig | 余弦退火总迭代数 |

> 提示：`lr` 在配置中默认为 0，但训练循环每步都会用 `cosine_schedule(global_step, ...)` 重写每个 param_group 的 `lr`，因此配置中的 `lr` 只作为优化器构造时的占位值，实际生效的学习率完全由 SchedulerConfig 决定。
