"""SENTINEL streaming: Kafka producer -> real-time scorer (champion + shadow models) -> actions + analytics.

Modules
  generator  - synthetic transactions with injected fraud rings, or replay of the labelled SFT dataset
  producer   - auto-mode traffic generator: publishes transactions to Kafka while switched on (button in the UI)
  processor  - consumes, scores on XGBoost (champion) + LoRA/QLoRA (shadow), decides, publishes actions
  store      - SQLite event store + analytics queries
  api        - FastAPI: animated flow page, analytics API, health (runs the processor in a thread)
"""

import os

TOPIC_TX = os.getenv("TOPIC_TRANSACTIONS", "sentinel.transactions")
TOPIC_DECISIONS = os.getenv("TOPIC_DECISIONS", "sentinel.decisions")
TOPIC_ACTIONS = os.getenv("TOPIC_ACTIONS", "sentinel.actions")

# What happens after each decision. "further action" hooks: Kafka topic sentinel.actions + SNS for BLOCK.
ACTIONS = {
    "ALLOW": {"action": "APPROVE", "label": "Non-fraud", "text": "Non-fraud: payment approved, no further action needed."},
    "REVIEW": {"action": "STEP_UP_AUTH", "label": "Suspicious",
               "text": "Suspicious: OTP step-up sent to the customer and case queued for an analyst."},
    "BLOCK": {"action": "BLOCK_AND_ALERT", "label": "Fraud",
              "text": "Fraud: payment blocked, card frozen, customer and fraud team alerted."},
}


def action_for(decision: str) -> dict:
    return ACTIONS.get(decision, ACTIONS["REVIEW"])


def kafka_config(**extra) -> dict:
    """librdkafka settings. Confluent Cloud: KAFKA_API_KEY / KAFKA_API_SECRET -> SASL_SSL + PLAIN.
    Local docker-compose broker: just KAFKA_BOOTSTRAP (plaintext)."""
    cfg = {"bootstrap.servers": os.getenv("KAFKA_BOOTSTRAP", "localhost:9092")}
    if os.getenv("KAFKA_API_KEY"):
        cfg.update({"security.protocol": "SASL_SSL", "sasl.mechanisms": "PLAIN",
                    "sasl.username": os.environ["KAFKA_API_KEY"], "sasl.password": os.getenv("KAFKA_API_SECRET", "")})
    cfg.update(extra)
    return cfg
