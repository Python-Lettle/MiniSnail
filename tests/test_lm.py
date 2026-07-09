import torch
from transformers import BatchEncoding
from transformers.tokenization_utils_tokenizers import TokenizersBackend
from minisnail.config import SnailConfig
from minisnail.model import init_model, SnailModel
from minisnail.tokenizer import get_tokenizer
from minisnail.generate import generate_text
if __name__ == '__main__':
    config = SnailConfig()
    tokenizer: TokenizersBackend = get_tokenizer(config)
    model: SnailModel = init_model(config)
    
    prompt: str = "你好，这是测试提示词。"
    generate_text(model, tokenizer, prompt, config)
       
