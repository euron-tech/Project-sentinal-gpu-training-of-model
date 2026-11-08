"""
Pipeline Evaluator - Fraud Classification Evaluation

Imports datasets lazily to avoid NumPy/pyarrow compatibility
issues on machines without the right dependencies installed.
"""

import json
import logging
import os
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


def _lazy_import_dataset():
    """Lazy import to avoid NumPy/pyarrow issues locally."""
    from datasets import Dataset
    return Dataset


def evaluate_model(
    model: Any,
    tokenizer: Any,
    test_dataset: List[Dict[str, Any]],
    output_path: str = "./results/evaluation_report.json",
):
    """Evaluate the model on a test dataset and save results to JSON."""
    logger.info(f"Evaluating model on {len(test_dataset)} test examples")

    Dataset = _lazy_import_dataset()

    hf_test = Dataset.from_list(test_dataset)

    # Tokenize
    def tokenize_fn(examples):
        return tokenizer(
            examples["instruction"],
            truncation=True,
            max_length=256,
            padding=False,
        )

    tokenized_test = hf_test.map(tokenize_fn, batched=True)

    # Evaluate (basic accuracy for fraud classification)
    import torch
    model.eval()
    correct = 0
    total = 0
    predictions = []

    logger.info("Running evaluation...")
    with torch.no_grad():
        for i, example in enumerate(tokenized_test):
            inputs = tokenizer(
                example["instruction"],
                return_tensors="pt",
                truncation=True,
                max_length=256,
            )

            # Move to device
            device = next(model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Generate response
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False,
            )

            # Decode
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Check if model correctly identified fraud
            is_fraud = test_dataset[i]["label"] == 1
            model_predict_fraud = any(
                keyword in response.lower()
                for keyword in ["fraud", "suspicious", "fraudulent"]
            )

            if is_fraud == model_predict_fraud:
                correct += 1
            total += 1

            predictions.append(
                {
                    "instruction": example["instruction"],
                    "prediction": response,
                    "actual_label": test_dataset[i]["label"],
                    "correct": is_fraud == model_predict_fraud,
                }
            )

            if (i + 1) % 50 == 0:
                logger.info(
                    f"Evaluated {i + 1}/{total} examples "
                    f"(accuracy: {correct / (i + 1):.2f})"
                )

    # Calculate metrics
    accuracy = correct / total if total > 0 else 0
    report = {
        "accuracy": accuracy,
        "total_examples": total,
        "correct_predictions": correct,
        "predictions": predictions,
    }

    # Save report
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    logger.info(
        f"Evaluation complete! Accuracy: {accuracy:.2f}, "
        f"Results saved to {output_path}"
    )

    return report
