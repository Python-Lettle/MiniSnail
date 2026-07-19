import torch
from transformers import BatchEncoding
from transformers.tokenization_utils_tokenizers import TokenizersBackend
from minisnail.config import SnailConfig
from minisnail.model import init_model, SnailModel
from minisnail.tokenizer import get_tokenizer
from minisnail.generate import generate_text
from minisnail.debug import console

if __name__ == '__main__':
    config = SnailConfig.from_json("./config.json")
    tokenizer: TokenizersBackend = get_tokenizer(config)
    # The model will load the weight from config.training.from_weight
    model: SnailModel = init_model(config)
    model.load_state_dict(torch.load(config.generation.model_path))
    console.print("[yellow]Loading model from weight:", config.generation.model_path)

    model.eval()
    model.to(device=torch.device(config.system.device))
    
    prompt: str = "我想知道中国的历史，"
    generate_text(model, tokenizer, prompt, config)
       
