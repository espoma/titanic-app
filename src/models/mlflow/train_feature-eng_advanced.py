"""
MLflow training script with feature engineering.

Trains multiple models with FamilySize feature and logs to MLflow.
Outputs are organized by experiment name for cleaner structure.
"""

import os
import sys
import warnings

# Add src to path BEFORE importing local modules
sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))

from sklearn.ensemble import (
    ExtraTreesClassifier,
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
import mlflow
import mlflow.sklearn
import dagshub

# Initialize DagsHub
dagshub.init(repo_owner="espoma", repo_name="titanic-app", mlflow=True)

import models.mlflow.mlflow_utils as mlflow_utils
from data import load_train_data, TitanicPreprocessor
from features import FamilySizeTransformer
from config import MODELS_DIR, MLFLOW_TRACKING_URI

warnings.filterwarnings("ignore")

# ============================================================================
# CONFIG
# ============================================================================

EXPERIMENT_NAME = "titanic-feature-eng-family-size"

# Create experiment-specific output directory
EXPERIMENT_MODELS_DIR = os.path.join(MODELS_DIR, EXPERIMENT_NAME)
os.makedirs(EXPERIMENT_MODELS_DIR, exist_ok=True)

# ============================================================================
# DATA LOADING
# ============================================================================

train_raw = load_train_data()
print(f"\n📊 Raw train shape: {train_raw.shape}")

X = train_raw.drop(columns=["Survived"])
y = train_raw["Survived"]
print(f"   Features: {X.columns.tolist()}")
print(f"   Target distribution:\n{y.value_counts(normalize=True)}")

# ============================================================================
# MODEL & PREPROCESSING CONFIGS
# ============================================================================

METHODS = ["basic", "median_impute", "knn_impute"]

MODELS = [
    ExtraTreesClassifier(n_estimators=100, max_depth=10, random_state=42),
    RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
    GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42),
    AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=1),
        n_estimators=100,
        random_state=42,
    ),
]


def main():
    experiment_id = mlflow_utils.get_or_create_experiment(
        EXPERIMENT_NAME, MLFLOW_TRACKING_URI
    )

    total_runs = len(MODELS) * len(METHODS)
    completed = 0
    skipped = 0

    for clf in MODELS:
        model_name = clf.__class__.__name__

        for method in METHODS:
            run_name = f"{model_name}_{method}"
            
            # Check if already exists
            if mlflow_utils.run_already_exists(experiment_id, run_name):
                print(f"⏭️  Run '{run_name}' already exists. Skipping...")
                skipped += 1
                continue

            print("\n" + "=" * 70)
            print(f"🚀 RUNNING: {model_name} | Method: {method}")
            print("=" * 70)

            # Feature transformer (FamilySize)
            feature_transformer = FamilySizeTransformer(
                drop_original=True, 
                include_self=True
            )

            # Preprocessor
            # Note: FamilySize is treated as ORDINAL as requested
            preprocessor = TitanicPreprocessor(
                method=method,
                keep_name=False,
                numeric_features=["Age", "Fare"],
                ordinal_features=["Pclass", "FamilySize"],  # FamilySize is ordinal
                categorical_features=["Sex", "Embarked"],
            )

            # Run experiment
            mlflow_utils.run_experiment(
                experiment_name=EXPERIMENT_NAME,
                run_name=run_name,
                X=X,
                y=y,
                clf=clf,
                preprocessor=preprocessor,
                TRACKING_URI=MLFLOW_TRACKING_URI,
                MODELS_DIR=EXPERIMENT_MODELS_DIR,  # Use experiment-specific dir
                feature_transformer=feature_transformer,
            )
            completed += 1

    # Summary
    print("\n" + "=" * 70)
    print("✅ TRAINING COMPLETE")
    print("=" * 70)
    print(f"\n   📈 Completed: {completed}")
    print(f"   ⏭️  Skipped:   {skipped}")
    print(f"   📁 Models saved to: {EXPERIMENT_MODELS_DIR}")
    print(f"\n   To view MLflow UI, run:")
    print(f"      mlflow ui --backend-store-uri {MLFLOW_TRACKING_URI}")
    print("   Then open: http://localhost:5000")


if __name__ == "__main__":
    main()
