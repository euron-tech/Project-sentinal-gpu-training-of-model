"""
SENTINEL — Main orchestration entry point
Fine-tunes Qwen2.5-0.5B for fraud classification via QLoRA.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from config import PipelineConfig, get_config
from pipeline.loader import SupabaseDatasource
from pipeline.dataset import FraudInstructionDataset, prepare_dataset
from pipeline.model import setup_model
from pipeline.trainer import run_training as train_model
from pipeline.evaluator import evaluate_model

logger = logging.getLogger("sentinel")


def parse_args():
    """Parse CLI arguments that override config."""
    p = argparse.ArgumentParser(
        description="SENTINEL: Fine-tune Qwen2.5-0.5B for fraud detection"
    )
    p.add_argument(
        "--mode",
        choices=["train", "eval", "full"],
        default="full",
        help="Run only training, only evaluation, or both (default: full)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Stop after data pipeline; skip model training/evaluation",
    )
    p.add_argument(
        "--data-file",
        type=str,
        default=None,
        help="Path to a CSV file as an alternative data source",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory",
    )
    p.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Override base model name",
    )
    p.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of training epochs",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override per-device batch size",
    )
    return p.parse_args()


def setup_logging():
    """Configure root logging to file + console."""
    log_dir = Path("./logs")
    log_dir.mkdir(exist_ok=True)

    fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"

    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        handlers=[
            logging.FileHandler(log_dir / "training.log", mode="w"),
            logging.StreamHandler(sys.stdout),
        ],
    )


def load_data(cfg):
    """Load raw records from the configured datasource."""
    logger.info("Loading data from Supabase ...")
    ds = SupabaseDatasource(
        api_url=cfg.data.api_url,
        api_key=cfg.data.api_key,
        table_name=cfg.data.table_name,
    )
    records = ds.fetch(limit=cfg.data.max_records, offset=0)
    logger.info(f"Fetched {len(records)} raw records from Supabase")
    return records


def run():
    """Main pipeline: load → prepare → train → evaluate."""
    args = parse_args()
    setup_logging()

    # Load environment variables from .env (if present)
    load_dotenv()

    # Merge CLI overrides into the config
    cfg = get_config()
    if args.model_name:
        cfg.model.model_name_or_path = args.model_name
    if args.epochs:
        cfg.training.num_epochs = args.epochs
    if args.batch_size:
        cfg.training.per_device_train_batch_size = args.batch_size
    if args.output_dir:
        cfg.training.output_dir = args.output_dir

    logger.info("=" * 60)
    logger.info("  SENTINEL — QLoRA Fraud Classification Pipeline")
    logger.info("=" * 60)
    logger.info(f"Model   : {cfg.model.model_name_or_path}")
    logger.info(f"LoRA r  : {cfg.model.lora_r}")
    logger.info(f"LoRA alpha : {cfg.model.lora_alpha}")
    logger.info(f"Bits    : {cfg.model.load_in_4bit}")
    logger.info(f"Epochs  : {cfg.training.num_epochs}")
    logger.info(f"BatchSz : {cfg.training.per_device_train_batch_size}")
    logger.info(f"Output  : {cfg.training.output_dir}")
    logger.info("-" * 60)

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    records = load_data(cfg)

    # 2. Prepare instruction-tuning dataset
    logger.info("Preparing instruction-tuning dataset ...")
    ds = prepare_dataset(records, cfg)
    # Split into train/validation
    split_ratio = getattr(cfg.training, "eval_split", 0.2)
    split_idx = int(len(ds) * (1 - split_ratio))
    train_ds = ds[:split_idx]
    val_ds = ds[split_idx:]
    logger.info(
        f"Dataset split -> train: {len(train_ds)}, val: {len(val_ds)}"
    )

    if args.dry_run:
        print("\n" + "=" * 60)
        logger.info("[Dry-run: data pipeline validated. Stopping here.]")
        logger.info("[Remove --dry-run to proceed with training & evaluation.]")
        print("=" * 60)
        sys.exit(0)

    # 3. Set up model
    logger.info("Loading base model & configuring LoRA ...")
    model_result = setup_model(cfg)
    model = model_result["model"]
    tokenizer = model_result["tokenizer"]

    # ------------------------------------------------------------------
    # 4. Train
    # ------------------------------------------------------------------
    if args.mode in ("train", "full"):
        trainer = train_model(
            config=cfg,
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_ds,
            eval_dataset=val_ds,
        )
        logger.info("Training complete — saving adapter ...")
        save_path = Path(cfg.training.output_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(save_path / "adapter")
        tokenizer.save_pretrained(save_path / "adapter")
        logger.info(f"Adapter saved to {save_path / 'adapter'}")
    else:
        # Load a pre-trained adapter for eval-only mode
        adapter_path = Path(cfg.training.output_dir) / "adapter"
        if adapter_path.exists():
            logger.info(f"Loading adapter from {adapter_path} ...")
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, adapter_path)
        else:
            logger.warning(
                f"No adapter found at {adapter_path}; "
                "evaluating the base model as-is."
            )

    # ------------------------------------------------------------------
    # 5. Evaluate
    # ------------------------------------------------------------------
    if args.mode in ("eval", "full"):
        logger.info("Running evaluation ...")
        metrics = evaluate_model(
            model=model,
            tokenizer=tokenizer,
            test_dataset=[{"instruction": r["instruction"], "label": r["label"]} for r in val_ds],
            output_path=str(Path(cfg.training.output_dir) / "evaluation_report.json"),
        )

        logger.info("=" * 60)
        logger.info("  EVALUATION RESULTS")
        logger.info("=" * 60)
        for k, v in metrics.items():
            logger.info(f"  {k:<30s} = {v}")
        logger.info("=" * 60)

        # Save metrics to JSON
        import json
        metrics_path = Path(cfg.training.output_dir) / "metrics.json"
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2, default=str)
        logger.info(f"Metrics saved to {metrics_path}")

    logger.info("SENTINEL pipeline finished.")


if __name__ == "__main__":
    run()