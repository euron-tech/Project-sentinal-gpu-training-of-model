"""
Pipeline Model - QLoRA Model Loading for Qwen2.5-0.5B

Imports transformers/peft lazily to avoid NumPy/pyarrow
compatibility issues on machines without the right dependencies
installed.
"""

import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


def setup_model(config: Any = None):
    """Configure the model for QLoRA training.

    Accepts a Config object (from config.py). Returns a dict with
    keys: model, tokenizer, peft_model (if LoRA was applied).
    """
    logger.info("Setting up model with QLoRA configuration...")

    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    # Extract settings from config object
    if config is not None and hasattr(config, "model"):
        model_name = config.model.model_name_or_path
        device_map = config.model.device_map
        load_in_4bit = config.model.load_in_4bit
        r = config.model.lora_r
        lora_alpha = config.model.lora_alpha
        lora_dropout = config.model.lora_dropout
    else:
        model_name = "Qwen/Qwen2.5-0.5B-Instruct"
        device_map = "auto"
        load_in_4bit = True
        r = 8
        lora_alpha = 16
        lora_dropout = 0.1

    logger.info(f"Loading base model: {model_name}")
    if device_map == "auto":
        logger.info("Device map: auto (will use all available GPUs)")
    else:
        logger.info(f"Device map: {device_map}")

    # Set up 4-bit quantization for QLoRA
    if load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        logger.info("4-bit quantization (NF4) enabled")
    else:
        bnb_config = None
        logger.info("Quantization disabled (full precision)")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Set pad_token to eos_token")

    # Load model
    model_kwargs = {"trust_remote_code": True, "device_map": device_map}
    if bnb_config:
        model_kwargs["quantization_config"] = bnb_config

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

    logger.info(f"Base model loaded. Params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    # Apply LoRA
    peft_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = prepare_model_for_kbit_training(model)
    peft_model = get_peft_model(model, peft_config)
    peft_model.print_trainable_parameters()

    logger.info("Model setup complete")
    return {
        "model": peft_model,
        "tokenizer": tokenizer,
        "peft_model": peft_model,
    }


# Backwards-compatible alias
load_model = setup_model
