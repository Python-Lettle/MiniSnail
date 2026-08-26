from dataclasses import dataclass, field
from typing import Optional
import json
import torch

@dataclass
class TokenizerConfig:
    """Tokenizer configuration"""
    vocab_size: int = 6400
    tokenizer_name: str = "minimind"
    tokenizer_root: str = "./model/minimind"

@dataclass
class ModelConfig:
    """Model architecture configuration"""
    vocab_size: int = 6400
    context_length: int = 512
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
    lr: float = 0
    betas: tuple[float, float] = (0.9, 0.95)
    weight_decay: float = 0.001
    dpo_beta: float = 0.1
    valid_interval: int = 400
    valid_batches: int = 20
    gradient_clip: float = 1.0
    accumulation_steps: int = 1
    print_interval: int = 200
    from_weight: Optional[str] | None = None
    use_checkpoint: bool = False
    from_checkpoint: str | None = None
    use_wandb: bool = True
    use_compile: bool = True
    use_amp: bool = True

@dataclass
class SchedulerConfig:
    """Learning rate scheduler configuration"""
    max_learning_rate: float = 0.0005
    min_learning_rate: float = 0.00005
    warmup_iters: int = 600
    cosine_cycle_iters: int = 6000

@dataclass
class SystemConfig:
    """System configuration"""
    device: str = "cuda"
    seed: int = 42
    dtype: str = "float32"  # "float32" | "bfloat16" | "float16"

@dataclass
class GenerationConfig:
    """Generation configuration"""
    model_path: str = "./output/model_best.pt"
    max_tokens: int = 512
    temperature: float = 0.8
    top_k: int = 40
    top_p: float = 0.9
    device: str = "cuda"
    repetition_penalty: float = 1.2
    greedy: bool = False

@dataclass
class WandbConfig:
    """Wandb configuration"""
    entity: str = "lettle-hong"
    project: str = "MiniSnail"
    id: str | None = None

@dataclass
class SnailConfig:
    """Complete training configuration"""
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    system: SystemConfig = field(default_factory=SystemConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)

    def get_torch_dtype(self):
        """Get torch.dtype from system.dtype string. Returns (model_dtype, amp_dtype).

        标准 AMP 语义: 模型权重恒为 fp32, autocast 只降低前向计算精度。
        (纯 fp16/bf16 权重的梯度无法被 GradScaler 处理, 且数值稳定性差)
        """
        dtype_map = {
            "float32": (torch.float32, None),
            "bfloat16": (torch.float32, torch.bfloat16),
            "float16": (torch.float32, torch.float16),
        }
        return dtype_map.get(self.system.dtype, (torch.float32, None))

    @classmethod
    def from_dict(cls, config_dict: dict) -> "SnailConfig":
        """Create configuration from dictionary"""
        return cls(
            tokenizer=TokenizerConfig(**config_dict.get("tokenizer", {})),
            model=ModelConfig(**config_dict.get("model", {})),
            training=TrainingConfig(**config_dict.get("training", {})),
            scheduler=SchedulerConfig(**config_dict.get("scheduler", {})),
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
            "system": self.system.__dict__,
            "generation": self.generation.__dict__,
            "wandb": self.wandb.__dict__,
        }
    
    def to_json(self, json_path: str):
        """Save configuration to JSON file"""
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

DEFAULT_CONFIG = SnailConfig()