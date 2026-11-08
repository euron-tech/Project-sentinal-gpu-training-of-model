# Inference utilities for fraud classification
# Parts adapted from Finguard (MIT): _extract_risk_level pattern
# Source: Rishi-Is-Cool/Finguard
# https://github.com/Rishi-Is-Cool/Finguard/blob/924e5aaa4f11af24163d8801f78aa24cc2a2f5f5/week

import re
from typing import Optional


def extract_risk_level(text: str) -> Optional[str]:
    """Extract risk level from model output text.

    Adapted from Finguard (MIT) - handles both structured JSON outputs
    and natural language responses. Source: Rishi-Is-Cool/Finguard
    """
    text_upper = text.upper()

    # Check for high risk indicators (AML-specific patterns)
    high_indicators = [
        "FRAUD", "FRAUDULENT", "FRAUD_CONFIRMED", "HIGH RISK", "HIGH-RISK",
        "STRUCTURING", "LAYERING", "SMURFING", "RAPID FUND MOVEMENT",
        "MONEY LAUNDERING", "AML ALERT"
    ]
    for indicator in high_indicators:
        if indicator in text_upper:
            return "HIGH"

    # Check for medium risk indicators
    medium_indicators = [
        "SUSPICIOUS", "REVIEW", "FLAGGED", "MEDIUM RISK", "MEDIUM-RISK",
        "ACCOUNT TAKEOVER", "ANOMALOUS"
    ]
    for indicator in medium_indicators:
        if indicator in text_upper:
            return "MEDIUM"

    # Default to low risk
    return "LOW"


def parse_fraud_probabilities(text: str) -> dict:
    """Parse fraud probability scores from model output.

    Expects either JSON-formatted output or natural language containing
    numeric probabilities.
    """
    result = {"fraud_probability": 0.5, "confidence": 0.5}

    # Try to extract from JSON-like output
    fraud_match = re.search(r'"?fraud_probability"?\s*:\s*([0-9.]+)', text)
    if fraud_match:
        result["fraud_probability"] = float(fraud_match.group(1))

    confidence_match = re.search(r'"?confidence"?\s*:\s*([0-9.]+)', text)
    if confidence_match:
        result["confidence"] = float(confidence_match.group(1))

    return result


def classify_transaction(risk_level: str, fraud_prob: float) -> str:
    """Convert risk level and fraud probability to final classification.

    Returns: 'FRAUD', 'SUSPICIOUS', or 'LEGITIMATE'
    """
    if risk_level == "HIGH" and fraud_prob > 0.7:
        return "FRAUD"
    elif risk_level == "MEDIUM" or (risk_level == "HIGH" and 0.4 < fraud_prob <= 0.7):
        return "SUSPICIOUS"
    else:
        return "LEGITIMATE"
