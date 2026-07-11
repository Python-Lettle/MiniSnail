from minisnail.tokenizer import get_tokenizer
from minisnail.config import SnailConfig
from transformers.tokenization_utils_tokenizers import TokenizersBackend
import rich

if __name__ == "__main__":
    config = SnailConfig()
    tokenizer: TokenizersBackend = get_tokenizer(config)
    messages: list[dict[str, str]] = [
        {"role": "system", "content": "你是一个优秀的聊天机器人，总是给我正确的回应！"},
        {"role": "user", "content": '你来自哪里？'},
        {"role": "assistant", "content": '我来自月球'},
        {"role": "user", "content": '你到底来自哪里？'},
        {"role": "assistant", "content": '我来自地球'}
    ]
    new_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False
    )

    # Show the chat template
    rich.print('-'*100)
    rich.print(new_prompt)

    # Show the tokenizer parameters
    rich.print('-'*100)
    rich.print('Tokenizer vocab size:', len(tokenizer))
    model_inputs = tokenizer(new_prompt)
    rich.print('Encoder length:', len(model_inputs['input_ids']))
    response = tokenizer.decode(model_inputs['input_ids'], skip_special_tokens=False)
    rich.print('Decoder consistency:', response == new_prompt, "\n")
    
    # Test the tokenizer
    rich.print('-'*100)
    rich.print('Compression test (Chars/Tokens):')
    test_texts = [
        # 1-2. 中文样本 (约200字)
        "人工智能是计算机科学的一个分支，它企图了解智能的实质，并生产出一种新的能以人类智能相似的方式做出反应的智能机器，该领域的研究包括机器人、语言识别、图像识别、自然语言处理和专家系统等。人工智能从诞生以来，理论和技术日益成熟，应用领域也不断扩大，可以设想，未来人工智能带来的科技产品，将会是人类智慧的“容器”。人工智能可以对人的意识、思维的信息过程的模拟。人工智能不是人的智能，但能像人那样思考、也可能超过人的智能。",
        "星际航行是指在星系内甚至星系间的空间中进行的航行。由于宇宙空间极其广阔，传统的化学火箭动力在恒星间航行时显得力不从心。科学家们提出了多种方案，包括离子推进器、核热火箭、甚至是利用反物质作为能源的设想。此外，曲率驱动和虫洞旅行等科幻概念也在理论物理研究中被反复探讨。尽管目前人类的足迹仅限于月球，但随着核聚变技术和材料科学的突破，前往火星乃至更遥远的太阳系边缘将成为可能。",
        # 3-4. 英文样本 (约200词/字符)
        "Large language models (LLMs) are a type of artificial intelligence (AI) trained on vast amounts of text data to understand and generate human-like language. These models use deep learning techniques, specifically transformers, to process and predict the next word in a sequence. LLMs like GPT-4, Llama, and Claude have demonstrated remarkable capabilities in coding, translation, and creative writing. However, they also face challenges such as hallucinations, where the model generates factually incorrect information, and the need for significant computational resources.",
        "The development of sustainable energy is crucial for the future of our planet. As climate change continues to impact global weather patterns, transitioning from fossil fuels to renewable sources like solar, wind, and hydroelectric power has become an urgent priority. Innovations in battery storage technology and smart grid management are essential to ensure a reliable energy supply. International cooperation and policy frameworks are also necessary to drive the global shift towards a greener economy and reduce carbon emissions.",
        # 5. 混合样本
        "Python 是一种高级编程语言，以其简洁的语法和强大的生态系统而闻名。It is widely used in data science, machine learning, and web development. 开发者可以利用 NumPy, Pandas, and PyTorch 等库快速构建复杂的应用。学习 Python 的过程非常愉快，因为它的代码读起来就像英语一样。Whether you are a beginner or an expert, Python offers something for everyone.",
        # 6. 超长中文段落 (约500字)
        "在自然语言处理领域，分词（Tokenization）是最基础也是最关键的预处理步骤之一。分词质量的好坏直接影响后续的词嵌入、句法分析、语义理解乃至最终模型的性能表现。现代分词算法主要分为基于词典的方法、基于统计的方法和基于神经网络的方法三大类。基于词典的方法简单高效，但难以处理未登录词和歧义切分；基于统计的方法如条件随机场（CRF）能够利用上下文信息改善歧义消解，但需要大量标注语料；基于神经网络的方法如BERT的WordPiece和BPE（Byte Pair Encoding）则通过学习数据中的子词模式实现了良好的平衡。值得注意的是，中文分词与英文分词存在显著差异：英文单词之间有天然空格分隔，而中文文本是连续的字符流，必须依靠算法自主判断词边界。这也是为什么中文分词被认为是“汉语信息处理的第一个门槛”。近年来，预训练语言模型的兴起对分词技术提出了新的挑战和机遇。一方面，模型需要使用固定大小的词表，这迫使研究者探索更高效的字词表示方法；另一方面，模型的自注意力机制可以在一定程度上弥补分词错误带来的信息损失。",

        # 7. 长英文段落 (约400词)
        "Tokenization is the process of converting a sequence of text into smaller units called tokens. These tokens can be words, subwords, characters, or even punctuation marks, depending on the specific tokenization strategy employed. In the context of large language models (LLMs), tokenization plays a critical role in determining how effectively the model can represent and process language. One of the most widely used tokenization algorithms is Byte Pair Encoding (BPE), which starts with individual characters and iteratively merges the most frequent pairs to form larger tokens. Another popular approach is WordPiece, which uses a likelihood-based criterion to decide which subword units to create. SentencePiece, developed by Google, offers a more flexible framework that can handle multiple languages without requiring pre-tokenization. The choice of tokenizer has profound implications: a good tokenizer can capture meaningful linguistic patterns while keeping the vocabulary size manageable; a poor tokenizer may fragment words too aggressively, leading to inefficient processing and degraded model performance. For multilingual models, the challenge is even greater, as the tokenizer must balance coverage across languages with very different morphological structures. For instance, while English words can often be represented with one or two tokens, agglutinative languages like Turkish may require many more tokens to encode equivalent meanings. This imbalance—known as the tokenization disparity—can cause underrepresented languages to suffer from higher computational costs and poorer model quality.",

        # 8. 混合中英文+代码
        '''在机器学习项目中，数据预处理通常占据80%的时间。例如，使用Pandas进行数据清洗时，我们经常需要处理缺失值：df.fillna(method='ffill', inplace=True) 可以向前填充缺失数据。对于文本数据，我们还需要special_characters = ['@', '#', '$', '%'] 列表来过滤噪声。有趣的是，CNN（卷积神经网络）和RNN（循环神经网络）在20世纪90年代就已提出，但由于当时算力不足，直到2012年AlexNet的出现才真正引爆了深度学习浪潮。如今的Transformer架构更是让这一切发生了质的飞跃。Attention Is All You Need 这篇论文提出的自注意力机制（Self-Attention），通过计算Q、K、V矩阵的点积来捕捉序列内部的长距离依赖关系。代码实现如下：
        import torch.nn as nn
        class SelfAttention(nn.Module):
            def __init__(self, embed_dim):
                super().__init__()
                self.q = nn.Linear(embed_dim, embed_dim)
                self.k = nn.Linear(embed_dim, embed_dim)
                self.v = nn.Linear(embed_dim, embed_dim)
            def forward(self, x):
                scores = torch.matmul(self.q(x), self.k(x).transpose(-2, -1)) / (embed_dim ** 0.5)
                weights = torch.softmax(scores, dim=-1)
                return torch.matmul(weights, self.v(x))
        这个简单的注意力机制已经成为现代NLP模型的基石。''',

        # 9. 数学公式与专业术语
        "在量子力学中，薛定谔方程描述了微观粒子的运动规律：iℏ∂ψ/∂t = Ĥψ。其中ℏ是约化普朗克常数，ψ是波函数，Ĥ是哈密顿算符。对于一维无限深势阱，本征能量可表示为E_n = n²π²ℏ²/(2mL²)，n=1,2,3,...。在统计学习理论中，VC维（Vapnik-Chervonenkis dimension）衡量了假设空间的复杂度。对于一个线性分类器在R^d空间中的VC维是d+1。支持向量机（SVM）通过最大化间隔(Margin)来寻找最优分类超平面，其优化目标为：min ||w||²/2 subject to y_i(w·x_i+b) ≥ 1。这些数学公式在学术论文的文本中频繁出现，Tokenizer需要能够正确处理希腊字母、上下标和特殊符号的组合。此外，化学方程式如 H₂SO₄ + 2NaOH → Na₂SO₄ + 2H₂O 也包含了丰富的下标和箭头符号。",

        # 10. 多语言混合（日韩俄法西+中英）
        "在全球化背景下，多语言处理能力成为衡量Tokenizer质量的重要标准。例如，日语「人工知能は未来を変える」（人工智能改变未来）包含了汉字与假名的混合；韩语「머신러닝은 데이터 과학의 핵심입니다」（机器学习是数据科学的核心）使用了谚文；俄语「Искусственный интеллект — это будущее」（人工智能是未来）采用了西里尔字母；法语「L'apprentissage profond a révolutionné le traitement du langage naturel」（深度学习彻底改变了自然语言处理）包含了大量缩合冠词和变音符号；西班牙语「La inteligencia artificial está transformando el mundo」（人工智能正在改变世界）则有其独特的语法结构。一个理想的Tokenizer应该能够对这些不同书写系统的文本进行统一的、无损的分割，同时保持语义完整性。混合文本如 AI技术の進歩は速いですね！ESTO ES INCREÍBLE! そして、Pythonプログラミングはとても面白い。 对分词器的编码能力提出了极高的要求。",

        # 11. 极端情况：重复字符与特殊格式
        "这是一个测试重复字符的极端情况：AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA 和 BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB 以及 1234567890123456789012345678901234567890。接下来测试特殊格式：①列表项一 ②列表项二 ③列表项三。时间戳示例：2026-07-05T15:30:00+08:00。URL示例：https://example.com/api/v2/models/tokenizer?param1=value1&param2=中文参数。邮箱示例：user.name+tag@company-domain.org。表情符号：😊🎉🚀💯🔥。制表符和换行符测试：\t第一列\t第二列\t第三列\n\t数据A\t数据B\t数据C\n\t数据D\t数据E\t数据F。HTML标签：<div class=\"content\" style=\"color:red;\">这段文本包含HTML标记</div>。JSON格式：{\"name\": \"张三\", \"age\": 30, \"scores\": [98.5, 87.0, 92.3], \"active\": true}。",

        # 12. 文学性描述（富有修辞和成语）
        "春风又绿江南岸，明月何时照我还？这句千古流传的诗句，展现了汉语在简练中蕴含深远意境的独特魅力。在文学创作中，排比、对偶、比喻等修辞手法的运用，使得文章如行云流水般自然流畅。譬如，“人生如逆旅，我亦是行人”道尽了聚散无常的禅意；“落霞与孤鹜齐飞，秋水共长天一色”则描绘出一幅动静相宜的绝美画卷。人工智能要真正理解这样的语言，不仅需要掌握语法规则，还要领悟文化背景和情感内涵。当算法面对“举头望明月，低头思故乡”这样的句子时，它需要识别出“明月”不仅是天文现象，更是思乡之情的文化符号。同样，成语“画蛇添足”“守株待兔”“亡羊补牢”等蕴含着丰富的寓言故事，模型需要在有限的上下文中准确捕捉这些固化表达的指代意义。这正是自然语言理解从‘形式’走向‘意义’的关键挑战。",

        # 13. 技术文档式文本（含路径、版本号、命令行）
        '''在配置深度学习环境时，我们通常需要安装特定版本的CUDA和cuDNN。例如，使用以下命令创建Conda环境并安装依赖：
            conda create -n pytorch_env python=3.10
            conda activate pytorch_env
            pip install torch==2.1.0+cu121 torchvision==0.16.0+cu121 --index-url https://download.pytorch.org/whl/cu121
            在训练模型时，配置文件通常采用YAML格式：
            model:
            name: bert-base-chinese
            vocab_size: 21128
            hidden_size: 768
            num_hidden_layers: 12
            num_attention_heads: 12
            training:
            batch_size: 32
            learning_rate: 2e-5
            epochs: 3
            output_dir: ./outputs/experiment_001/
            文件路径中的反斜杠、冒号、点和下划线，以及版本号中的小数点，都对Tokenizer的分词粒度有重要影响。''',
    ]
    total_compression = 0
    for i, text in enumerate(test_texts):
        encoded = tokenizer.encode(text)
        token_count = len(encoded)
        char_count = len(text)
        compression_ratio = char_count / token_count
        total_compression += compression_ratio
        print(f"样本 {i+1} | 字符数: {char_count:4} | Tokens: {token_count:3} | 压缩率: {compression_ratio:.2f}")
    
    print(f"平均压缩率: {total_compression / len(test_texts):.2f}")