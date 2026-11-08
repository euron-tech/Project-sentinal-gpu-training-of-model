"""Data loading utilities for the fraud detection pipeline."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from config import PipelineConfig
from pipeline.ports import SupabaseDatasource

logger = logging.getLogger(__name__)


def save_to_jsonl(data: List[Dict[str, Any]], path: str) -> None:
    """Save records to JSON Lines format."""
    with open(path, "w", encoding="utf-8") as f:
        for record in data:
            f.write(json.dumps(record, default=str) + "\n")
    logger.info(f"Saved {len(data)} records to {path}")


def load_from_jsonl(path: str) -> List[Dict[str, Any]]:
    """Load records from JSON Lines format."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    logger.info(f"Loaded {len(records)} records from {path}")
    return records


def create_datasource(cfg: PipelineConfig) -> SupabaseDatasource:
    """Create a synchronous Supabase datasource from config."""
    return SupabaseDatasource(
        api_url=cfg.data.api_url,
        api_key=cfg.data.api_key,
        table_name=cfg.data.table_name,
    )
