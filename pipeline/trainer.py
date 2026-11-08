"""
SENTINEL — Fine-tuning Pipeline

Loads Qwen 0.5B, applies LoRA adapters, and trains on synthetic-transaction
instruction-tuning data.
"""

import os
import json
import torch
from typing import List

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model

from config import PipelineConfig
import logging
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Model & Tokenizer Loading
# ---------------------------------------------------------------------------
def load_model_and_tokenizer(config: PipelineConfig):
    """
    Load the base model and tokenizer.

    * CPU mode  → full fp16 fine-tuning with LoRA (no quantization)
    * GPU mode  → 4-bit QLoRA if `load_in_4bit=True`
    """
    _cfg = config.model
    logger.info(f"Loading model: {_cfg.model_name_or_path}")

    # BitsAndBytes config (only used on GPU)
    bnb_config = None
    if _cfg.load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=_cfg.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=getattr(torch, _cfg.torch_dtype, torch.float16),
            bnb_4bit_use_double_quant=_cfg.bnb_4bit_use_double_quant,
        )
        logger.info("QLoRA 4-bit quantisation enabled.")
    else:
        logger.info("LoRA (full precision) — no quantization.")

    device_map = _cfg.device_map

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(_cfg.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    logger.info("Tokenizer ready.")

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        _cfg.model_name_or_path,
        quantization_config=bnb_config,
        torch_dtype=getattr(torch, _cfg.torch_dtype, torch.float16),
        device_map=device_map,
        trust_remote_code=True,
    )
    logger.info(f"Model loaded on device: {next(model.parameters()).device}")

    return model, tokenizer


# ---------------------------------------------------------------------------
# 2. Apply LoRA Adapters
# ---------------------------------------------------------------------------
def apply_lora(model, config: PipelineConfig):
    """Wrap the base model with LoRA adapter layers."""
    _cfg = config.model

    peft_config = LoraConfig(
        r=_cfg.lora_r,
        lora_alpha=_cfg.lora_alpha,
        target_modules=_cfg.target_modules,
        lora_dropout=_cfg.lora_dropout,
        bias=_cfg.bias,
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # IMPORTANT: only call prepare_model_for_kbit_training when using 4-bit
    if _cfg.load_in_4bit:
        model = prepare_model_for_kbit_training(
            model,
            cast_dtype=torch.bfloat16,
            gradient_checkpointing=True,
        )
        logger.info("Prepared for 4-bit training.")
    else:
        logger.info("Skipping kbit preparation (full precision LoRA).")

    return model


# ---------------------------------------------------------------------------
# 3. Data Collator
# ---------------------------------------------------------------------------
class FraudDataCollator:
    """Dynamic padding + fraud-weighted loss masking."""

    def __init__(self, tokenizer, fraud_weight: float = 10.0):
        self.tokenizer = tokenizer
        self.fraud_weight = fraud_weight

    def __call__(self, batch):
        input_ids = [item["input_ids"] for item in batch]
        labels = [item["labels"] for item in batch]
        is_fraud = [item.get("label", item.get("is_fraud", 0)) for item in batch]

        max_len = max(len(x) for x in input_ids)
        pad = self.tokenizer.pad_token_id

        padded_ids = []
        padded_labels = []
        for ids, lbl, fraud in zip(input_ids, labels, is_fraud):
            cur_ids = ids + [pad] * (max_len - len(ids))
            cur_labels = lbl + [pad] * (max_len - len(lbl))
            # Mark padding tokens so cross-entropy ignores them
            cur_labels = [
                x if x != pad else -100
                for x in cur_labels
            ]
            padded_ids.append(cur_ids)
            padded_labels.append(cur_labels)

        return {
            "input_ids": torch.tensor(padded_ids),
            "labels": torch.tensor(padded_labels),
            "attention_mask": torch.tensor(
                [[1] * len(x) + [0] * (max_len - len(x)) for x in input_ids]
            ),
            # Per-sample fraud flag for weighted loss
            "is_fraud": torch.tensor(is_fraud),
        }


# ---------------------------------------------------------------------------
# 4. Compute Metrics
# ---------------------------------------------------------------------------
def compute_metrics(eval_pred, config: PipelineConfig):
    """Run zero-shot classification on generated text."""
    logits, labels = eval_pred
    predictions = logits.argmax(axis=1) if isinstance(logits, list) else logits

    # Simple accuracy (full metrics need inference pass)
    correct = (predictions == labels).mean()
    return {"eval_accuracy": correct}


# ---------------------------------------------------------------------------
# 5. Training Loop
# ---------------------------------------------------------------------------
def run_training(config: PipelineConfig, model, tokenizer, train_dataset,
                 eval_dataset=None, compute_metrics_fn=None, data_collator=None):
    """Set up and run the HuggingFace Trainer."""
    tc = config.training

    args = TrainingArguments(
        output_dir=tc.output_dir,
        num_train_epochs=tc.num_epochs,
        per_device_train_batch_size=tc.per_device_train_batch_size,
        gradient_accumulation_steps=tc.gradient_accumulation_steps,
        learning_rate=tc.learning_rate,
        weight_decay=tc.weight_decay,
        lr_scheduler_type=tc.lr_scheduler_type,
        warmup_ratio=tc.warmup_ratio,
        logging_steps=tc.logging_steps,
        save_steps=tc.save_steps,
        evaluation_strategy="steps",
        eval_steps=tc.eval_steps,
        save_strategy="steps",
        save_total_limit=tc.save_total_limit,
        report_to=tc.report_to,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        fp16=not torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
        bf16=torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
        dataloader_pin_memory=not torch.cuda.is_available(),  # False on CPU
        remove_unused_columns=False,
    )

    # Use provided collator/metrics or fall back to defaults
    if data_collator is None:
        data_collator = FraudDataCollator(tokenizer, tc.fraud_weight)
    if compute_metrics_fn is None:
        compute_metrics_fn = lambda ep: compute_metrics(ep, config)

    trainer_kwargs = dict(
        model=model,
        args=args,
        train_dataset=train_dataset,
        data_collator=None,  # SFTTrainer handles tokenization internally
        compute_metrics=compute_metrics_fn,
    )
    if eval_dataset is not None:
        trainer_kwargs["eval_dataset"] = eval_dataset

    trainer = Trainer(**trainer_kwargs)

    trainer.train()

    # Save the LoRA adapter
    adapter_dir = os.path.join(tc.output_dir, "final_adapter")
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    logger.info(f"LoRA adapter saved to {adapter_dir}")
    return trainer


# ---------------------------------------------------------------------------
# 6. Export adapter to ONNX / TorchScript (placeholder)
# ---------------------------------------------------------------------------
def export_model(model, tokenizer, path: str):
    """Export the fine-tuned adapter for deployment."""
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    logger.info(f"Model exported to {path}")


def prepare_model_for_kbit_training(model, cast_dtype, gradient_checkpointing=True):
    """
    From the `peft` library: prepare a quantized model for training.
    Reimplemented inline so we only call it when load_in_4bit=True
    (i.e. when bitsandbytes is available on GPU).
    """
    from peft import prepare_model_for_kbit_training as _orig
    return _orig(model, cast_dtype=cast_dtype, gradient_checkpointing=gradient_checkpointing)
