# Tasks

- [x] Task 1: 创建 `docs/README.md` 导航首页与项目概览
  - [x] SubTask 1.1: 编写项目一句话简介与定位（MiniSnail = 小参数模拟大模型全流程能力的训练框架）
  - [x] SubTask 1.2: 绘制整体架构图（数据→模型→训练→推理的文字/mermaid 流程图）
  - [x] SubTask 1.3: 编写模块职责表（src/minisnail 各文件 + scripts 各脚本的职责一句话说明，附 file:/// 链接）
  - [x] SubTask 1.4: 编写第三方依赖清单与用途、项目内部模块依赖关系图
  - [x] SubTask 1.5: 编写配置系统总览（SnailConfig 8 个子配置一览表 + from_json/to_json/get_torch_dtype 说明）
  - [x] SubTask 1.6: 编写子文档导航目录链接 + 「训练流程主线串联图」

- [x] Task 2: 创建 `docs/01-tokenizer-data.md`（训练流程①：分词器与数据预处理）
  - [x] SubTask 2.1: 说明分词器加载（get_tokenizer + minimind tokenizer_config.json 关键 special token：bos=<|im_start|>=1, eos=<|im_end|>=2, pad=<|endoftext|>=0，vocab=6400，ChatML 模板）
  - [x] SubTask 2.2: 说明预训练数据 tokenize（scripts/data_tokenize.py：多进程 fork、_init_worker、_encode_chunk、按字节比例 train/valid 切分、memmap int32 输出）
  - [x] SubTask 2.3: 说明 SFT 数据预处理（scripts/preprocess_sft.py：apply_name_filter、pre_processing_chat、post_processing_chat、_clean_turn、_render_safely、generate_labels 的 -100 mask 机制）
  - [x] SubTask 2.4: 标注每个关键函数/类的 file:/// 定位链接

- [x] Task 3: 创建 `docs/02-model.md`（训练流程②：模型架构与关键类/函数）
  - [x] SubTask 3.1: 绘制 SnailModel 整体前向数据流图（Embedding → blocks → norm → output）
  - [x] SubTask 3.2: 说明 SnailBlock（Pre-Norm + MultiHeadSelfAttention + 残差 + Pre-Norm + PWFFN + 残差）
  - [x] SubTask 3.3: 说明 MultiHeadSelfAttention（W_Q/K/V/O 线性投影、rearrange 多头、RoPE 应用、F.scaled_dot_product_attention is_causal）
  - [x] SubTask 3.4: 说明 RotaryPositionalEmbedding（init_cache 计算 cos/sin、forward 奇偶切片旋转）
  - [x] SubTask 3.5: 说明 PWFFN（SwiGLU：w2(SiLU(w1·x) ⊙ (w3·x))）
  - [x] SubTask 3.6: 说明 RMSNorm、init_model 工厂函数
  - [x] SubTask 3.7: 标注每个关键类/方法的 file:/// 定位链接与默认超参（d_model=512, num_layers=4, num_heads=16, d_ff=1344, context_length=512）

- [x] Task 4: 创建 `docs/03-training.md`（训练流程③：数据加载、训练函数、预训练、SFT、检查点、实验追踪）
  - [x] SubTask 4.1: 说明数据加载（PretrainDataset memmap 懒加载非重叠 chunk、SFTDataset npy 加载、get_dataloader 参数）
  - [x] SubTask 4.2: 说明训练核心函数（cross_entropy_loss 自实现、cosine_schedule warmup+退火、gradient_clipping L2、scaled_dot_product_attention/softmax/silu）
  - [x] SubTask 4.3: 说明预训练流程 train_lm（训练循环结构、梯度累积、AMP、cosine LR、验证采样、model_best.pt、异常保存 checkpoint）
  - [x] SubTask 4.4: 说明 SFT 流程 train_sft（F.cross_entropy + ignore_index=-100、start_step_in_epoch 断点续训、sft_best.pt）
  - [x] SubTask 4.5: 说明检查点机制（save_checkpoint/load_checkpoint 字段、get_model_from_checkpoint.py 提取权重、use_checkpoint 续训 wandb resume）
  - [x] SubTask 4.6: 说明实验追踪（wandb init/log/finish、LossMonitor 统计与可视化、loss 曲线保存）
  - [x] SubTask 4.7: 标注每个关键函数/脚本的 file:/// 定位链接

- [x] Task 5: 创建 `docs/04-inference.md`（训练流程④：推理与生成）
  - [x] SubTask 5.1: 说明 SnailModel.generate（temperature 缩放、repetition_penalty 打折、top-k 截断、multinomial 采样、eos 停止、skip_prompt、context_length 滑窗）
  - [x] SubTask 5.2: 说明 SnailModel.chat（apply_chat_template 构造 prompt、加 assistant 标记、调用 generate、decode）
  - [x] SubTask 5.3: 说明 generate_text 顶层封装（encode → generate → decode → 合并 prompt）
  - [x] SubTask 5.4: 标注 file:/// 定位链接

- [x] Task 6: 创建 `docs/05-how-to-run.md`（项目运行方式）
  - [x] SubTask 6.1: 环境与依赖安装步骤（pip install -e . 及第三方依赖）
  - [x] SubTask 6.2: 生成默认配置（generate_config.py → config.json）
  - [x] SubTask 6.3: 预训练数据预处理命令与参数（data_tokenize.py）
  - [x] SubTask 6.4: SFT 数据预处理命令与参数（preprocess_sft.py）
  - [x] SubTask 6.5: 预训练运行命令（train_lm.py --config）
  - [x] SubTask 6.6: SFT 微调运行命令（train_sft.py --config_path）
  - [x] SubTask 6.7: 从检查点提取模型（get_model_from_checkpoint.py）
  - [x] SubTask 6.8: 推理运行方式（tests/test_lm.py 或 generate_text/chat）
  - [x] SubTask 6.9: 每条命令标注对应脚本的 file:/// 链接与默认参数

# Task Dependencies
- Task 1 应最先完成（README 为其他文档提供导航与术语基线）
- Task 2 / Task 3 / Task 4 / Task 5 互相独立，可并行
- Task 6 依赖 Task 2~5 已产出（运行方式需引用前面阶段），但可与其他文档并行起草
