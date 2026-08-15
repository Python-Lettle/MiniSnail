import os
import torch
from typing import IO, BinaryIO
from transformers import BatchEncoding, PreTrainedTokenizer
from minisnail.config import SnailConfig
from minisnail.model import init_model, SnailModel
from minisnail.tokenizer import get_tokenizer
from minisnail.generate import generate_text
from minisnail.debug import console

def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
) -> dict[str, any]:
    """
    Given a serialized checkpoint (path or file-like object), restore the
    serialized state to the given model and optimizer.
    Return the checkpoint state.

    Args:
        src (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialized checkpoint.
    Returns:
        dict[str, any]: A dictionary of the checkpoint state.
    """
    # Load the checkpoint from the file or object
    if isinstance(src, str) or isinstance(src, os.PathLike):
        src = open(src, 'rb')
    # Load the model state from the checkpoint
    checkpoint = torch.load(src, weights_only=False)
    
    return checkpoint

prompts: list[str] = [
    "请介绍一下你自己。",
    "什么是人工智能？",
    "请列出人工智能的三个优点，每一点用一句话说明。",
    '''小明有一只猫，这只猫叫雪球。\n\n雪球每天喜欢吃鱼。\n\n请问，这只猫叫什么？''',
    '''如果所有猫都是动物，小花是一只猫。\n\n请问小花是什么？''',
    '''如果今天下雨，那么地面会湿。\n\n今天下雨了。\n\n请问地面怎么样？''',
    "请用列表形式介绍学习Python的三个理由。",
    "请比较Python和C++的区别，用表格回答。",
    "请分步骤说明如何训练一个语言模型。",
    "你是一名数学老师，请解释什么是质数。",
    "你是一名Python工程师，请解释什么是装饰器。",
    "中国的首都是",
    "中华人民共和国成立于",
    "Transformer模型的核心结构包括",
    "小明有5个苹果，又买了3个苹果，请问一共有多少个苹果？",
    "如果一个三角形三个角分别是60度、60度、60度，这是什么三角形？",
    "请写一个Python函数，计算两个数字的和。",
    '''解释下面代码：\nfor i in range(5):\n\tprint(i)''',
    "假如人类突然失去互联网，会发生什么？",
    "请写一首关于秋天的小诗。",
]

if __name__ == '__main__':
    config = SnailConfig.from_json("./config.json")
    tokenizer: PreTrainedTokenizer = get_tokenizer(config)
    # The model will load the weight from config.training.from_weight
    model: SnailModel = init_model(config)

    # Load the checkpoint
    # model_dir = "./output/checkpoint.pt"
    # checkpoint = load_checkpoint(model_dir)
    # model.load_state_dict(checkpoint["model_state_dict"])

    # Load the model weight
    model_dir = "./model/local_sft_new_model_epo2/sft_new.pt"
    model.load_state_dict(torch.load(model_dir, weights_only=False))
    
    console.print("[yellow]Loading model from weight:", model_dir)
    
    model.eval()
    model.to(device=torch.device(config.system.device))
    
    test_type = int(input("请输入测试类型（0：自动测试，1：手动测试）："))
    if test_type == 0:
        # Auto Test
        test_num: int = 0
        for prompt in prompts:
            test_num += 1
            console.print(f"[yellow]Test {test_num}")
            print("Prompt:")
            print(prompt)
            response = model.chat(prompt, tokenizer, repetition_penalty=config.generation.repetition_penalty, top_k=config.generation.top_k, top_p=config.generation.top_p, max_tokens=config.generation.max_tokens, do_sample=not config.generation.greedy)
            print("Response:")
            print(response)
    else:
        # Manual Test
        while True:
            prompt: str = input("👤: ")
            response = model.chat(prompt, tokenizer, repetition_penalty=config.generation.repetition_penalty, top_k=config.generation.top_k, top_p=config.generation.top_p, max_tokens=config.generation.max_tokens, do_sample=not config.generation.greedy)
            print("🤖:", response)

    
    
    
    
       
