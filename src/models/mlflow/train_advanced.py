import pandas as pd
import numpy as np
import os
import sys
import warnings

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    ExtraTreesClassifier,
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
)
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow
import mlflow.sklearn
import joblib
from sklearn.pipeline import Pipeline
import mlflow_utils as mlflow_utils
import dagshub

# Initialize DagsHub
dagshub.init(repo_owner="espoma", repo_name="titanic-app", mlflow=True)

sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
from data import load_train_data, TitanicPreprocessor
from config import MODELS_DIR, MLFLOW_TRACKING_URI

warnings.filterwarnings("ignore")

# ============================================================================
# CONFIG
# ============================================================================

os.makedirs(MODELS_DIR, exist_ok=True)

train_raw = load_train_data()
print(f"\nRaw train shape: {train_raw.shape}")

X = train_raw.drop(columns=["Survived"])
y = train_raw["Survived"]
print(f"    Raw features: {X.columns.tolist()}")
print(f"    Target distribution:\n{y.value_counts(normalize=True)}")


def main():

    experiment_name = "titanic-model-comparison-advanced"
    experiment_id = mlflow_utils.get_or_create_experiment(experiment_name, MLFLOW_TRACKING_URI)

    methods = ["basic", "median_impute", "knn_impute"]

    models = [
        ExtraTreesClassifier(n_estimators=100, max_depth=10, random_state=42),
        RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
        # GradientBoostingClassifier(n_estimators=100, max_depth=10, random_state=42),
        AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=1),
            n_estimators=100,
            random_state=42,
        ),
    ]

    for clf in models:
        model_name = clf.__class__.__name__

        for method in methods:
            print("\n" + "=" * 70)
            run_name = f"{model_name}_{method}"
            if mlflow_utils.run_already_exists(experiment_id, run_name):
                print(f"Run {run_name} already exists. Skipping...")
                continue
            print(
                f"RUNNING EXPERIMENT: {experiment_name} | MODEL: {model_name} | METHOD: {method}"
            )
            preprocessor = TitanicPreprocessor(
                method=method,
                keep_name=False,
                numeric_features=["Age", "Fare"],
                ordinal_features=["Pclass", "Parch", "SibSp"],
                categorical_features=["Sex"],
            )
            print("=" * 70)
            mlflow_utils.run_experiment(
                experiment_name,
                run_name,
                X,
                y,
                clf,
                preprocessor,
                MLFLOW_TRACKING_URI,
                MODELS_DIR,
            )

        for method in methods:
            print("\n" + "=" * 70)
            run_name = f"{model_name}-{method}-no_ordinals"
            if mlflow_utils.run_already_exists(experiment_id, run_name):
                print(f"Run {run_name} already exists. Skipping...")
                continue
            print(
                f"RUNNING EXPERIMENT: {experiment_name} | MODEL: {model_name} | METHOD: {method} (No Ordinals)"
            )
            preprocessor = TitanicPreprocessor(
                method=method,
                keep_name=False,
                numeric_features=["Age", "Fare", "Pclass", "SibSp", "Parch"],
                ordinal_features=[],
                categorical_features=["Sex"],
            )
            print("=" * 70)
            mlflow_utils.run_experiment(
                experiment_name,
                run_name,
                X,
                y,
                clf,
                preprocessor,
                MLFLOW_TRACKING_URI,
                MODELS_DIR,
            )

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"\nTo view MLflow UI, run:")
    print(f"  mlflow ui --backend-store-uri {MLFLOW_TRACKING_URI}")
    print("\nThen open: http://localhost:5000")


if __name__ == "__main__":
    main()
