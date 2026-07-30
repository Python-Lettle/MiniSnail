from minisnail.config import SnailConfig
from transformers import AutoTokenizer, PreTrainedTokenizer

def get_tokenizer(config: SnailConfig) -> PreTrainedTokenizer:
    '''Get a tokenizer using the options in config.
    '''
    return AutoTokenizer.from_pretrained(config.tokenizer.tokenizer_root)
    
