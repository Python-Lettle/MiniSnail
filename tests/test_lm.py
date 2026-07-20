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
    # model.load_state_dict(torch.load(config.generation.model_path))
    model.load_state_dict(torch.load("./output/sft_best.pt"))
    console.print("[yellow]Loading model from weight:", "./output/sft_best.pt")

    model.eval()
    model.to(device=torch.device(config.system.device))
    
    prompt: str = "你都有什么功能？"
    response = model.chat(prompt, tokenizer, repetition_penalty=1.2, top_k=40, max_tokens=1024)
    console.print("Prompt:")
    console.print(prompt)
    console.print("Response:")
    console.print(response)
       
