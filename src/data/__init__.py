"""
Data loading and preprocessing utilities for Titanic dataset.

Exports:
    - load_train_data: Load training data from CSV
    - load_test_data: Load test data from CSV
    - TitanicPreprocessor: Sklearn-compatible preprocessor for imputation and encoding
"""

from .load_data import load_train_data, load_test_data
from .preprocess import TitanicPreprocessor

__all__ = [
    "load_train_data",
    "load_test_data",
    "TitanicPreprocessor",
]
