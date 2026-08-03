# Checklist

## 文档完整性
- [x] `docs/` 目录下存在 README.md 及 01~05 共 6 个 Markdown 文件
- [x] README.md 包含项目概览、整体架构图、模块职责表、依赖关系、配置系统总览、子文档导航
- [x] 5 篇子文档分别覆盖：分词器与数据预处理、模型架构、训练流程、推理生成、运行方式

## 主线串联（大模型训练流程）
- [x] 文档顺序严格遵循：准备 → 数据 → 模型 → 训练 → 推理
- [x] README 含「训练流程主线串联图」，把 5 个阶段串成端到端链路
- [x] 每篇子文档顶部有「返回首页」与「上/下一篇」导航链接

## 关键类与函数覆盖
- [x] 配置类全部覆盖：SnailConfig 及 8 个子配置 + from_json/to_json/get_torch_dtype
- [x] 模型类全部覆盖：SnailModel、SnailBlock、MultiHeadSelfAttention、RotaryPositionalEmbedding、PWFFN、init_model、generate、chat
- [x] 数据类全部覆盖：PretrainDataset、SFTDataset、get_dataloader
- [x] 训练函数全部覆盖：cross_entropy_loss、cosine_schedule、gradient_clipping、silu、softmax、scaled_dot_product_attention
- [x] 训练脚本全部覆盖：train_lm、train_sft、save_checkpoint、load_checkpoint
- [x] 推理全部覆盖：generate_text、get_tokenizer
- [x] 工具全部覆盖：setup_seed、read_memmap_data、LossMonitor、console

## 代码定位准确性
- [x] 所有关键类/函数引用均使用 file:/// 绝对路径链接
- [x] 链接指向的文件路径与行号区间在源码中真实存在且正确
- [x] 链接文本使用文件 basename，未用反引号包裹

## 依赖与运行方式
- [x] README 列出全部第三方依赖（torch/numpy/transformers/einops/jaxtyping/rich/matplotlib/wandb/datasets/tqdm）及用途
- [x] README 给出项目内部模块依赖关系
- [x] 05-how-to-run.md 给出 8 步完整可执行命令
- [x] 每条运行命令对应到 scripts/ 下真实脚本及其参数名

## 准确性与质量
- [x] 文档描述与源码实际行为一致（如 vocab_size=6400、bos=1、eos=2、pad=0、d_model=512、num_layers=4、num_heads=16、d_ff=1344、context_length=512）
- [x] SFT 标签 -100 mask 机制说明准确（仅 assistant 回复部分参与 loss）
- [x] 梯度累积 / AMP / cosine warmup / 梯度裁剪逻辑说明准确
- [x] 未引入源码中不存在的功能或臆造 API
