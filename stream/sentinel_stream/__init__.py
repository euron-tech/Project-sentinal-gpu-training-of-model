"""SENTINEL streaming: Amazon Kinesis producer -> real-time scorer (champion + shadow models) -> actions + analytics.

Modules
  bus        - Amazon Kinesis Data Streams producer (PutRecords) and shard reader (GetRecords)
  generator  - synthetic transactions with injected fraud rings, or replay of the labelled SFT dataset
  producer   - auto-mode traffic generator: publishes transactions to Kinesis while switched on (button in the UI)
  processor  - consumes, scores on XGBoost (champion) + LoRA/QLoRA (shadow), decides, publishes actions
  store      - SQLite event store + analytics queries
  api        - FastAPI: animated flow page, analytics API, health (runs the processor in a thread)
"""

import os

# Kinesis data stream names (Terraform creates sentinel-<env>-transactions / -decisions)
STREAM_TX = os.getenv("STREAM_TRANSACTIONS", "sentinel-transactions")
STREAM_DECISIONS = os.getenv("STREAM_DECISIONS", "sentinel-decisions")

# What happens after each decision. "further action" hooks: the action fields of every record on the decisions stream + SNS for BLOCK.
ACTIONS = {
    "ALLOW": {"action": "APPROVE", "label": "Non-fraud", "text": "Non-fraud: payment approved, no further action needed."},
    "REVIEW": {"action": "STEP_UP_AUTH", "label": "Suspicious",
               "text": "Suspicious: OTP step-up sent to the customer and case queued for an analyst."},
    "BLOCK": {"action": "BLOCK_AND_ALERT", "label": "Fraud",
              "text": "Fraud: payment blocked, card frozen, customer and fraud team alerted."},
}


def action_for(decision: str) -> dict:
    return ACTIONS.get(decision, ACTIONS["REVIEW"])
