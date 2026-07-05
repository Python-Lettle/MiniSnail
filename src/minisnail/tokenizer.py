from transformers import AutoTokenizer

def get_tokenizer(config: SnailConfig):
    '''Get a tokenizer using the options in config.
    '''
    return AutoTokenizer.from_pretrained(config.tokenizer_path)
    
