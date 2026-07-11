from dataclasses import dataclass, field
from typing import Optional, List
import json
import os

@dataclass
class TokenizerConfig:
    """Tokenizer configuration"""
    vocab_size: int = 6400
    tokenizer_name: str = "minimind"
    tokenizer_root: str = "./model/minimind"
    bos_token_id: int = 1
    eos_token_id: int = 2

@dataclass
class ModelConfig:
    """Model architecture configuration"""
    vocab_size: int = 6400
    context_length: int = 256
    d_model: int = 512
    num_layers: int = 4
    num_heads: int = 16
    d_ff: int = 1344
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-6

@dataclass
class TrainingConfig:
    """Training configuration"""
    epochs: int = 6000
    batch_size: int = 32
    lr: float = 1e-5
    betas: tuple[float, float] = (0.9, 0.999)
    weight_decay: float = 0.001
    valid_interval: int = 150
    gradient_clip: float = 1.0
    print_interval: int = 20
    from_weight: Optional[str] | None = None

@dataclass
class SchedulerConfig:
    """Learning rate scheduler configuration"""
    max_learning_rate: float = 3e-4
    min_learning_rate: float = 1e-5
    warmup_iters: int = 500
    cosine_cycle_iters: int = 10000

@dataclass
class DataConfig:
    """Data configuration"""
    train_data_path: str = "./data/tinystories_train.npy"
    valid_data_path: str = "./data/tinystories_valid.npy"
    save_model_dir: str = "./output/"
    dataset_name: str = "TinyStoriesV2-GPT4"

@dataclass
class SystemConfig:
    """System configuration"""
    device: str = "cuda"
    seed: int = 42

@dataclass
class GenerationConfig:
    """Generation configuration"""
    max_tokens: int = 5000
    temperature: float = 0.8
    top_k: int = 5
    eos_token: str = "<|endoftext|>"
    device: str = "cuda"

@dataclass
class WandbConfig:
    """Wandb configuration"""
    entity: str = "lettle-hong"
    project: str = "MiniSnail"

@dataclass
class SnailConfig:
    """Complete training configuration"""
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    data: DataConfig = field(default_factory=DataConfig)
    system: SystemConfig = field(default_factory=SystemConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> "SnailConfig":
        """Create configuration from dictionary"""
        return cls(
            tokenizer=TokenizerConfig(**config_dict.get("tokenizer", {})),
            model=ModelConfig(**config_dict.get("model", {})),
            training=TrainingConfig(**config_dict.get("training", {})),
            scheduler=SchedulerConfig(**config_dict.get("scheduler", {})),
            data=DataConfig(**config_dict.get("data", {})),
            system=SystemConfig(**config_dict.get("system", {})),
            generation=GenerationConfig(**config_dict.get("generation", {})),
            wandb=WandbConfig(**config_dict.get("wandb", {})),
        )
    
    @classmethod
    def from_json(cls, json_path: str) -> "SnailConfig":
        """Load configuration from JSON file"""
        with open(json_path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)
    
    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            "tokenizer": self.tokenizer.__dict__,
            "model": self.model.__dict__,
            "training": self.training.__dict__,
            "scheduler": self.scheduler.__dict__,
            "data": self.data.__dict__,
            "system": self.system.__dict__,
            "generation": self.generation.__dict__,
            "wandb": self.wandb.__dict__,
        }
    
    def to_json(self, json_path: str):
        """Save configuration to JSON file"""
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

DEFAULT_CONFIG = SnailConfig()