import os
import math
from transformers import PretrainedConfig

class SnailConfig(PretrainedConfig):
    model_type = "minisnail"
    def __init__(self, **kwargs):
        '''Initialize a new configuration.
        '''
        # Extract PretrainedConfig-specific keys before they're consumed by kwargs.get
        self.bos_token_id: int = kwargs.get("bos_token_id", 1)
        self.eos_token_id: int = kwargs.get("eos_token_id", 2)
        super().__init__(bos_token_id=self.bos_token_id, eos_token_id=self.eos_token_id, **kwargs)

        # Path configuration
        self.save_dir: str | os.PathLike | None = kwargs.get("save_dir", "./output")
        self.model_name: str | None = kwargs.get("model_name", None)
        self.data_path: str | os.PathLike | None = kwargs.get("data_path", "./data/pretrain_t2t_mini.jsonl")

        # Training configuration
        self.epochs: int = kwargs.get("epochs", 2)
        self.batch_size: int = kwargs.get("batch_size", 32)
        self.learning_rate: float = kwargs.get("learning_rate", 0.001)
        self.device: str = torch.device(kwargs.get("device", "cuda:0"))
        self.dtype: str = kwargs.get("dtype", "bfloat16")
        self.num_workers: int = kwargs.get("num_workers", 8)
        self.accumulation_steps: int = kwargs.get("accumulation_steps", 8)
        self.grad_clip: float = kwargs.get("grad_clip", 1.0)
        self.log_interval: int = kwargs.get("log_interval", 100)
        self.save_interval: int = kwargs.get("save_interval", 1000)
        
        # Tokenizer configuration
        self.tokenizer_path: str = kwargs.get("tokenizer_path", "./model")
        self.vocab_size: int = kwargs.get("vocab_size", 6400)

        # Model configuration
        self.hidden_size: int = kwargs.get("hidden_size", 768)                      # Model dimension
        self.num_hidden_layers: int = kwargs.get("num_hidden_layers", 8)            # Number of layers
        self.num_attention_heads: int = kwargs.get("num_attention_heads", 8)        # Number of attention heads
        self.intermediate_size: int = kwargs.get("intermediate_size", math.ceil(self.hidden_size * math.pi / 64) * 64)           # Feedforward dimension = 8/3 * hidden_size
        self.max_seq_len: int = kwargs.get("max_seq_len", 340)                      # Maximum sequence length of the model
        self.hidden_act: str = kwargs.get("hidden_act", 'silu')
        self.max_position_embeddings: int = kwargs.get("max_position_embeddings", 32768)
        self.rms_norm_eps: float = kwargs.get("rms_norm_eps", 1e-6)
        self.rope_theta: float = kwargs.get("rope_theta", 1e6)
        self.q_k_dim: int = kwargs.get("q_k_dim", 64)
        self.attention_dropout: float = kwargs.get("attention_dropout", 0.0)

        # MOE configuration (Not used)
        self.use_moe: int = kwargs.get("use_moe", 0)                    # Whether to use MOE, 0 for no, 1 for yes
        
        # Wandb configuration
        self.use_wandb: int = kwargs.get("use_wandb", 0)             # Whether to use Wandb, 0 for no, 1 for yes
        self.wandb_project: str | None = kwargs.get("wandb_project", "minisnail")
        
