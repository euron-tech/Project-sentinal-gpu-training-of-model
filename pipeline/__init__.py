"""
Pipeline package — strict modularity.

Exports (modules): model, trainer, evaluator, dataset
Exports (functions): setup_model, train_model, evaluate_model, FraudInstructionDataset
"""

# Module-level exports (strict modularity)
import pipeline.model
import pipeline.trainer
import pipeline.evaluator
import pipeline.dataset

# Function-level exports (for main.py convenience)
from pipeline.model import setup_model
from pipeline.trainer import run_training as train_model
from pipeline.evaluator import evaluate_model
from pipeline.dataset import FraudInstructionDataset, prepare_dataset

__all__ = [
    "model",
    "trainer",
    "evaluator",
    "dataset",
    "setup_model",
    "train_model",
    "evaluate_model",
    "FraudInstructionDataset",
]
