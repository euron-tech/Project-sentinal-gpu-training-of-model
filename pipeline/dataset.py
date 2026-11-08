"""Dataset preparation: convert raw transaction dicts into instruction-tuning format."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from config import DataConfig, PipelineConfig

logger = logging.getLogger(__name__)

# Instruction template for fraud classification
SYSTEM_PROMPT = (
    "You are SENTINEL, PayVista's fraud detection intelligence. "
    "Analyze the transaction context and return a JSON decision with fields: "
    '"fraud_probability" (0-1), "decision" (ALLOW/REVIEW/BLOCK), '
    '"confidence" (0-1), and "reasoning" (plain text explanation).'
)

USER_TEMPLATE = (
    "Analyze this transaction for fraud risk:\n"
    "Merchant: {merchant_name}\n"
    "Amount: {transaction_amount} {currency}\n"
    "Channel: {channel}\n"
    "Customer device: {device_risk_level}\n"
    "Customer location: {geo_risk_level}\n"
    "Transaction hour: {transaction_hour}\n"
    "Failed attempts (last 24h): {failed_attempts_24h}\n"
    "Customer account age (days): {customer_age_days}\n"
    "Recent avg transaction amount: {recent_avg_amount}\n"
    "Amount spike ratio: {amount_spike_ratio}\n"
    "Device fingerprint: {device_fingerprint}\n"
    "IP risk score: {ip_risk_score}\n"
    "Geo mismatch: {geo_mismatch}\n"
    "New device: {new_device}\n"
    "Velocity (transactions last hour): {velocity_last_hour}\n"
    "---\n"
    "Customer's last {max_history} transactions:\n"
    "{history}\n"
    "---\n"
    "Return ONLY valid JSON with keys: fraud_probability, decision, confidence, reasoning"
)

ASSISTANT_TEMPLATE = (
    '{{\n  "fraud_probability": {fraud_probability},\n'
    '  "decision": "{label}",\n'
    '  "confidence": {confidence},\n'
    '  "reasoning": "{reasoning}"\n'
    "}}"
)


def _label_to_decision(label: Optional[str]) -> str:
    """Map the database label to the output decision."""
    if label is None:
        return "ALLOW"
    label_upper = label.upper().strip()
    if label_upper in ("FRAUD", "FRAUDULENT", "FRAUD_CONFIRMED", "FALSE_POSITIVE"):
        return "BLOCK"
    if label_upper in ("SUSPICIOUS", "REVIEW", "FLAGGED", "FALSE_POSITIVE"):
        return "REVIEW"
    return "ALLOW"


def _build_history_text(history_rows: List[Dict[str, Any]], max_n: int = 20) -> str:
    """Format the customer's recent transactions as a concise text block."""
    lines = []
    for row in history_rows[:max_n]:
        lines.append(
            f"- {row.get('transaction_time', '?')}: "
            f"{row.get('transaction_amount', '?')} {row.get('currency', 'INR')} "
            f"at {row.get('merchant_name', 'unknown')} "
            f"via {row.get('channel', '?')} "
            f"[{row.get('transaction_status', '?')}]"
        )
    return "\n".join(lines) if lines else "(no recent transaction history)"


def _compute_features(row: Dict[str, Any]) -> Dict[str, Any]:
    """Extract and compute features from a raw transaction row."""
    amount = float(row.get("transaction_amount", 0) or 0)
    recent_avg = float(row.get("recent_avg_amount", 0) or 0)
    # Amount spike ratio
    spike = round(amount / recent_avg, 2) if recent_avg > 0 else 1.0
    return {
        "merchant_name": row.get("merchant_name", "unknown"),
        "transaction_amount": amount,
        "currency": row.get("currency", "INR"),
        "channel": row.get("channel", "unknown"),
        "device_risk_level": row.get("device_risk_level", "low"),
        "geo_risk_level": row.get("geo_risk_level", "low"),
        "transaction_hour": row.get("transaction_hour", "12"),
        "failed_attempts_24h": int(row.get("failed_attempts_24h", 0) or 0),
        "customer_age_days": int(row.get("customer_age_days", 365) or 365),
        "recent_avg_amount": recent_avg,
        "amount_spike_ratio": spike,
        "device_fingerprint": row.get("device_fingerprint", "unknown"),
        "ip_risk_score": float(row.get("ip_risk_score", 0) or 0),
        "geo_mismatch": bool(row.get("geo_mismatch", False)),
        "new_device": bool(row.get("new_device", False)),
        "velocity_last_hour": int(row.get("velocity_last_hour", 0) or 0),
    }


def prepare_dataset(
    raw_records: List[Dict[str, Any]],
    cfg: PipelineConfig,
    split: str = "train",
) -> List[Dict[str, Any]]:
    """Convert raw DB rows into instruction-tuning messages for SFTTrainer.

    Returns a list of dicts, each with:
      - messages: [{"role": "system", ...}, {"role": "user", ...}, {"role": "assistant", ...}]
      - label: the ground-truth label (for evaluation)
    """
    data_cfg = cfg.data
    dataset = []
    skipped = 0

    for row in raw_records:
        txn_id = row.get("transaction_id")
        txn_time = row.get("transaction_time", "")
        label = row.get("label") or "legitimate"

        # --- time-based split ---
        if split == "train":
            if txn_time >= data_cfg.train_end_date:
                skipped += 1
                continue
        elif split == "val":
            if not (data_cfg.val_start_date <= txn_time < data_cfg.val_end_date):
                skipped += 1
                continue
        elif split == "holdout":
            if txn_time < data_cfg.holdout_start_date:
                skipped += 1
                continue

        # Build context
        features = _compute_features(row)
        # Get history from the row (should be pre-joined by loader)
        history_raw = row.get("recent_transactions", [])
        if isinstance(history_raw, str):
            try:
                history_raw = json.loads(history_raw)
            except (json.JSONDecodeError, TypeError):
                history_raw = []
        history_text = _build_history_text(history_raw, data_cfg.max_history_transactions)

        # Fill user prompt
        user_prompt = USER_TEMPLATE.format(
            max_history=data_cfg.max_history_transactions,
            history=history_text,
            **features,
        )

        # Build assistant response from label
        # Reasoning adapted from Finguard (MIT) — AML-specific patterns: structuring, layering,
        # smurfing, rapid fund movement. Source: Rishi-Is-Cool/Finguard
        decision = _label_to_decision(label)
        if label.upper().strip() in ("FRAUD", "FRAUDULENT", "FRAUD_CONFIRMED"):
            fraud_prob = 0.95
            confidence = 0.90
            reasoning = (
                f"This transaction shows patterns consistent with fraudulent activity. "
                f"Amount spike ratio of {features['amount_spike_ratio']} and "
                f"device risk level {features['device_risk_level']} indicate elevated risk. "
                f"Transaction patterns suggest potential structuring, rapid fund movement, "
                f"or layering activity consistent with money laundering indicators."
            )
        elif label.upper().strip() in ("SUSPICIOUS", "REVIEW", "FLAGGED"):
            fraud_prob = 0.55
            confidence = 0.60
            reasoning = (
                f"This transaction shows some anomalous patterns requiring manual review. "
                f"Factors: {features['device_risk_level']} device risk, "
                f"{features['failed_attempts_24h']} failed attempts in last 24h. "
                f"Transaction pattern may indicate smurfing or account takeover activity."
            )
        else:
            fraud_prob = 0.05
            confidence = 0.85
            reasoning = (
                f"This transaction appears legitimate. "
                f"Consistent with customer's typical behavior and merchant profile."
            )

        assistant_text = ASSISTANT_TEMPLATE.format(
            fraud_probability=fraud_prob,
            label=decision,
            confidence=confidence,
            reasoning=reasoning.replace('"', "'"),
        )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": assistant_text},
        ]

        dataset.append({
            "instruction": user_prompt,
            "response": assistant_text,
            "label": label,
            "transaction_id": txn_id,
            "transaction_time": txn_time,
        })

    logger.info(
        f"Prepared {len(dataset)} samples for split={split} "
        f"({skipped} skipped by date filter)"
    )
    return dataset


def save_to_jsonl(dataset: List[Dict[str, Any]], path: str):
    """Save a prepared dataset to JSONL format for Hugging Face."""
    import json
    with open(path, "w", encoding="utf-8") as f:
        for item in dataset:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    logger.info(f"Saved {len(dataset)} records to {path}")


def load_from_jsonl(path: str) -> List[Dict[str, Any]]:
    """Load a prepared dataset from JSONL."""
    dataset = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                dataset.append(json.loads(line))
    logger.info(f"Loaded {len(dataset)} records from {path}")
    return dataset


class FraudInstructionDataset:
    """HuggingFace-compatible instruction dataset for QLoRA fine-tuning.

    Wraps prepared JSONL instruction data so it can be passed directly to
    ``Trainer`` / ``transformers.Trainer`` as a Dataset.
    """

    def __init__(self, path: str):
        self._records = load_from_jsonl(path)

    def __len__(self):
        return len(self._records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self._records[idx]
