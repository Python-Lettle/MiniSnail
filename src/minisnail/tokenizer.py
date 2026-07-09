from minisnail.config import SnailConfig
from transformers import AutoTokenizer
from transformers.tokenization_utils_tokenizers import TokenizersBackend

def get_tokenizer(config: SnailConfig) -> TokenizersBackend:
    '''Get a tokenizer using the options in config.
    '''
    return AutoTokenizer.from_pretrained(config.tokenizer_path)
    
