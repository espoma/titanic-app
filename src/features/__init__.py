"""
Feature engineering transformers for Titanic dataset.

All transformers are sklearn-compatible (BaseEstimator + TransformerMixin)
and can be used directly in sklearn Pipelines.

Usage:
    from features import FamilySizeTransformer
    
    pipeline = Pipeline([
        ("family", FamilySizeTransformer(drop_original=True)),
        ("preprocessor", TitanicPreprocessor(...)),
        ("classifier", RandomForestClassifier()),
    ])
"""

from .family import FamilySizeTransformer

__all__ = [
    "FamilySizeTransformer",
]
