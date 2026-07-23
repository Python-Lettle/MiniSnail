import os
import torch
from typing import IO, BinaryIO
from transformers import BatchEncoding
from transformers.tokenization_utils_tokenizers import TokenizersBackend
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
    checkpoint = torch.load(src)
    
    return checkpoint

if __name__ == '__main__':
    config = SnailConfig.from_json("./config.json")
    tokenizer: TokenizersBackend = get_tokenizer(config)
    # The model will load the weight from config.training.from_weight
    model: SnailModel = init_model(config)

    # Load the checkpoint
    # checkpoint = load_checkpoint("./output/checkpoint.pt")
    # model.load_state_dict(checkpoint["model_state_dict"])

    model.load_state_dict(torch.load("./output/sft_64500.pt"))
    console.print("[yellow]Loading model from weight:", "./output/sft_64500.pt")
    
    model.eval()
    model.to(device=torch.device(config.system.device))
    
    prompt: str = "为我介绍什么是电脑"
    response = model.chat(prompt, tokenizer, repetition_penalty=1.2, top_k=40, max_tokens=512)
    console.print("Prompt:")
    console.print(prompt)
    console.print("Response:")
    console.print(response)
       
