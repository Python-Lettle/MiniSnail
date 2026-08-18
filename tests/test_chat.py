import os
import torch
from typing import IO, BinaryIO
from transformers import BatchEncoding, PreTrainedTokenizer
from minisnail.config import SnailConfig
from minisnail.model import init_model, SnailModel
from minisnail.tokenizer import get_tokenizer
from minisnail.debug import console

if __name__ == '__main__':
    config = SnailConfig.from_json("./config.json")
    device = torch.device(config.system.device)
    console.print("[yellow]Using device:", device)
    
    tokenizer: PreTrainedTokenizer = get_tokenizer(config)
    # The model will load the weight from config.training.from_weight
    model: SnailModel = init_model(config)

    # Load the model weight
    model_dir = "./model/local_dpo_model_epo2/dpo_new.pt"
    model.load_state_dict(torch.load(model_dir, weights_only=False))
    
    console.print("[yellow]Loading model from weight:", model_dir)
    
    model.eval()
    model.to(device=device)

    # Manual Test
    while True:
        prompt: str = input("👤: ")
        prompt_tensor = tokenizer(prompt, return_tensors="pt")
        prompt_tensor = prompt_tensor.to(device=device)
        prompt_tensor = prompt_tensor["input_ids"]
        
        print("🤖: ", end="")
        for token in model.streaming_generate(
            prompt_tensor,
            max_tokens=config.generation.max_tokens,
            temperature=config.generation.temperature,
            repetition_penalty=config.generation.repetition_penalty,
            top_k=config.generation.top_k,
            top_p=config.generation.top_p,
            do_sample=not config.generation.greedy,
            ):
            print(tokenizer.decode(token), end="", flush=True)
        print()
