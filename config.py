# ===========================================================
# Project SENTINEL — Fine-tuning Pipeline Configuration
# ===========================================================
# All configuration for LoRA / QLoRA fine-tuning of Qwen 0.5B
# ===========================================================

import os
import torch
from dataclasses import dataclass, field
from typing import List, Optional

from dotenv import load_dotenv
load_dotenv()

# Detect CPU vs GPU at import time
_CPU_MODE = not torch.cuda.is_available()


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------
@dataclass
class DatabaseConfig:
    """Connection details for Supabase / PostgreSQL."""
    host: str = os.getenv("SUPABASE_DB_HOST", "")
    port: int = int(os.getenv("SUPABASE_DB_PORT", "5432"))
    dbname: str = os.getenv("SUPABASE_DB_NAME", "postgres")
    user: str = os.getenv("SUPABASE_DB_USER", "")
    password: str = os.getenv("SUPABASE_DB_PASSWORD", "")
    # Supabase REST API fallback
    api_url: str = os.getenv("SUPABASE_URL", "")
    api_key: str = os.getenv("SUPABASE_KEY", "")

    @property
    def postgres_dsn(self) -> str:
        if not all([self.host, self.user, self.password]):
            return ""
        return (
            f"postgresql://{self.user}:{self.password}"
            f"@{self.host}:{self.port}/{self.dbname}"
        )

    def has_credentials(self) -> bool:
        return bool(self.postgres_dsn) or bool(self.api_url)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
@dataclass
class ModelConfig:
    """Base model and adapter configuration."""
    # Base model — Qwen 0.5B
    model_name_or_path: str = "Qwen/Qwen2.5-0.5B"
    # CPU: use float16 (bfloat16 needs GPU). GPU: use bfloat16 if available.
    torch_dtype: str = "float16" if _CPU_MODE else ("bfloat16" if torch.cuda.is_bf16_supported() else "float16")
    device_map: str = "cpu" if _CPU_MODE else "auto"

    # LoRA parameters
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])
    bias: str = "none"

    # QLoRA 4-bit quantisation — only enabled when explicitly requested AND GPU is present
    load_in_4bit: bool = False
    bnb_4bit_compute_dtype: str = field(default="float16")  # matches torch_dtype; updated in apply_mode()
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
@dataclass
class DataConfig:
    """Data loading and serialisation configuration."""
    # Source
    use_supabase: bool = True
    api_url: str = os.getenv("SUPABASE_URL", "")
    api_key: str = os.getenv("SUPABASE_KEY", "")
    table_name: str = "transactions"
    local_data_dir: str = "./data"
    local_train_file: str = "./data/transactions.csv"

    # Time-based split (ISO format YYYY-MM-DD)
    train_end_date: str = "2026-01-01"    # train on Jan 2025 – Dec 2025
    val_start_date: str = "2026-01-01"    # validate on Jan 2026
    val_end_date: str = "2026-02-28"      # up to Feb 2026
    holdout_start_date: str = "2026-01-01"  # untouched holdout

    # Serialisation
    max_records: int = int(os.getenv("MAX_RECORDS", "50000"))
    max_history_transactions: int = 20
    max_prompt_length: int = 1024
    chunk_size: int = 50000              # rows per Supabase fetch


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    num_epochs: int = int(os.getenv("NUM_EPOCHS", "3"))
    per_device_train_batch_size: int = int(os.getenv("BATCH_SIZE", "4"))
    gradient_accumulation_steps: int = 4
    learning_rate: float = float(os.getenv("LR", "2e-4"))
    weight_decay: float = 0.01
    max_steps: int = -1                      # -1 means full epochs
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.05

    # Logging & saving
    logging_steps: int = 50
    save_steps: int = 500
    eval_steps: int = 250
    save_total_limit: int = 3

    # Output
    output_dir: str = os.getenv("OUTPUT_DIR", "./output/sentinel-lora")
    report_to: str = "none"                  # or "tensorboard"

    # Class imbalance
    fraud_weight: float = 10.0               # weight for fraud class in loss

    # Dataset
    dataset_cache_dir: str = "./data/cache"


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
@dataclass
class EvalConfig:
    """Evaluation configuration."""
    eval_batch_size: int = 8
    block_threshold: float = 0.80
    review_threshold: float = 0.30

    # Metrics
    target_recall: float = 0.85
    target_precision: float = 0.70
    target_pr_auc: float = 0.80


# ---------------------------------------------------------------------------
# Full Pipeline Config
# ---------------------------------------------------------------------------
@dataclass
class PipelineConfig:
    """Master configuration combining all sub-configs."""
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    # High-level switches
    mode: str = "lora"       # "lora" or "qlora"
    do_train: bool = True
    do_eval: bool = True

    def apply_mode(self):
        """Apply mode-specific settings."""
        if self.mode == "qlora":
            if _CPU_MODE:
                print("[⚠️] QLoRA (4-bit quantization) requires a GPU. Falling back to LoRA (full fp16).")
                self.mode = "lora"
            else:
                self.model.load_in_4bit = True
                self.training.per_device_train_batch_size = 2
                self.training.gradient_accumulation_steps = 8
                self.training.output_dir = "./output/sentinel-qlora"
        elif self.mode == "lora":
            self.model.load_in_4bit = False
            self.training.output_dir = os.getenv("OUTPUT_DIR", "./output/sentinel-lora")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def get_config(mode: str = "lora") -> PipelineConfig:
    """Load the default config and apply the requested mode."""
    cfg = PipelineConfig(mode=mode)
    cfg.apply_mode()
    return cfg
