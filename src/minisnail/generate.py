import torch
import time
from transformers.tokenization_utils_tokenizers import TokenizersBackend
from minisnail.model import SnailModel
from minisnail.config import SnailConfig
from minisnail.util import console

def generate_text(model: SnailModel, tokenizer: TokenizersBackend, prompt: str, config: SnailConfig = None, device: torch.device = None):
    '''Generate text output by the model.
    '''
    device = torch.device(config.system.device) if device is None else device
    
    model.to(device)
    model.eval()

    prompt_ids: list[int] = tokenizer.encode(prompt)
    prompt_tensor: torch.Tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    start_time = time.time()
    with torch.no_grad():
        logits = model.generate(
            prompt_tensor,
            max_tokens=config.generation.max_tokens,
            temperature=config.generation.temperature,
            top_k=config.generation.top_k,
            eos_token_id=config.tokenizer.eos_token_id,
        )
        
        console.print("Logits:")
        console.print(logits)

        output_ids: list[int] = logits[0].cpu().numpy().tolist()

        # ==== Merge the original prompt and the generated content ====
        full_ids = prompt_ids + output_ids
        text = tokenizer.decode(full_ids)
        console.print("Prompt:")
        console.print(prompt)
        console.print("Generated Text:")
        console.print(text)
    end_time = time.time()
    console.print(f"Generation time: {end_time - start_time:.2f} seconds")