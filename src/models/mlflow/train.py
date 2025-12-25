"""
Train baseline model (Logistic Regression) with MLflow experiment tracking.

This script:
1. Loads processed train/test data
2. Uses 'basic' preprocessing method
3. Trains a Logistic Regression model with 5-fold cross-validation
4. Logs metrics, parameters, and model to MLflow
5. Serves as baseline for future experiments
"""

import pandas as pd
import numpy as np
import os
import sys
import warnings
import subprocess

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow
import mlflow.sklearn
import joblib
from sklearn.pipeline import Pipeline
import dagshub

# Initialize DagsHub
dagshub.init(repo_owner="espoma", repo_name="titanic-app", mlflow=True)

warnings.filterwarnings("ignore")

# Add src to path to import preprocess module and config
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from data import load_train_data, TitanicPreprocessor
from config import MODELS_DIR, MLFLOW_TRACKING_URI

# ============================================================================
# CONFIG
# ============================================================================

# Create models directory if it doesn't exist
os.makedirs(MODELS_DIR, exist_ok=True)

# Load raw train data (we apply the TitanicPreprocessor inside a pipeline)
train_raw = load_train_data()
print(f"   Raw train shape: {train_raw.shape}")

X = train_raw.drop(columns=["Survived"])
y = train_raw["Survived"]
print(f"   Raw features: {X.columns.tolist()}")
print(f"   Target distribution:\n{y.value_counts(normalize=True)}")


def run_experiment(experiment_name, run_name, X, y, clf, preprocessor):
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(experiment_name)

    # Build pipeline: preprocessor + classifier
    print(f"\n2. Building pipeline (preprocessor + {clf.__class__.__name__})...")
    pipeline = Pipeline([("preprocessor", preprocessor), ("clf", clf)])

    print(f"\n   Preprocessing info: {preprocessor.get_config()}")

    # Cross-validation (preprocessing will be fitted within each fold)
    print("\n3. Running 5-fold cross-validation on pipeline...")
    cv_results = cross_validate(
        pipeline,
        X,
        y,
        cv=5,
        scoring=["accuracy", "precision", "recall", "f1"],
        return_train_score=True,
    )

    # Compute CV stats
    cv_metrics = {
        "accuracy_mean": cv_results["test_accuracy"].mean(),
        "accuracy_std": cv_results["test_accuracy"].std(),
        "precision_mean": cv_results["test_precision"].mean(),
        "precision_std": cv_results["test_precision"].std(),
        "recall_mean": cv_results["test_recall"].mean(),
        "recall_std": cv_results["test_recall"].std(),
        "f1_mean": cv_results["test_f1"].mean(),
        "f1_std": cv_results["test_f1"].std(),
        "train_accuracy_mean": cv_results["train_accuracy"].mean(),
    }

    print("\n   Cross-Validation Results:")
    print(
        f"   Accuracy:  {cv_metrics['accuracy_mean']:.6f} (+/- {cv_metrics['accuracy_std']:.6f})"
    )
    print(
        f"   Precision: {cv_metrics['precision_mean']:.6f} (+/- {cv_metrics['precision_std']:.6f})"
    )
    print(
        f"   Recall:    {cv_metrics['recall_mean']:.6f} (+/- {cv_metrics['recall_std']:.6f})"
    )
    print(f"   F1:        {cv_metrics['f1_mean']:.6f} (+/- {cv_metrics['f1_std']:.6f})")

    # Start MLflow run
    print("\n4. Logging to MLflow...")
    with mlflow.start_run(run_name=run_name):
        # Link to Git Commit for DagsHub
        try:
            commit_id = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
            mlflow.set_tag("mlflow.source.git.commit", commit_id)
            # Descriptive tags for organization
            mlflow.set_tag("experiment_stage", "baseline")
            mlflow.set_tag("model_purpose", "initial_benchmark")
        except Exception:
            pass

        # Log parameters
        mlflow.log_params(
            {
                "model": clf.__class__.__name__,
                "preprocessing_method": preprocessor.method,
                "cv_folds": 5,
            }
        )
        
        # Log model specific params if available
        if hasattr(clf, "get_params"):
             mlflow.log_params(clf.get_params())

        # Log metrics
        mlflow.log_metrics(
            {
                "cv_accuracy_mean": cv_metrics["accuracy_mean"],
                "cv_accuracy_std": cv_metrics["accuracy_std"],
                "cv_precision_mean": cv_metrics["precision_mean"],
                "cv_precision_std": cv_metrics["precision_std"],
                "cv_recall_mean": cv_metrics["recall_mean"],
                "cv_recall_std": cv_metrics["recall_std"],
                "cv_f1_mean": cv_metrics["f1_mean"],
                "cv_f1_std": cv_metrics["f1_std"],
                "train_accuracy_mean": cv_metrics["train_accuracy_mean"],
            }
        )

        # Train final pipeline on full train set
        print("   Training final pipeline on full training set...")
        pipeline.fit(X, y)
        train_accuracy = accuracy_score(y, pipeline.predict(X))
        mlflow.log_metric("final_train_accuracy", train_accuracy)

        # Save preprocessor separately with joblib
        preproc_path = os.path.join(MODELS_DIR, f"preprocessor_{run_name}.joblib")
        joblib.dump(pipeline.named_steps["preprocessor"], preproc_path)
        mlflow.log_artifact(preproc_path, artifact_path="preprocessor")

        # Log the full pipeline (preprocessor + model) to MLflow
        mlflow.sklearn.log_model(pipeline, "model")

        # Log preprocessing config
        mlflow.log_dict(
            {
                "method": preprocessor.method,
                "keep_name": preprocessor.keep_name,
                "numeric_features": preprocessor.numeric_features,
                "ordinal_features": preprocessor.ordinal_features,
                "categorical_features": preprocessor.categorical_features,
            },
            "preprocessing_config.json",
        )

        print("   ✓ Model logged to MLflow")

        # Get run info
        run_id = mlflow.active_run().info.run_id
        print(f"   Run ID: {run_id}")


def main():

    experiment_name = "titanic-model-comparison"
    methods = ["basic", "median_impute", "knn_impute"]
    
    models = [
        LogisticRegression(random_state=42, solver="lbfgs", max_iter=1000),
        RandomForestClassifier(n_estimators=100, min_samples_split=2, max_depth=5, oob_score=True, random_state=10),
        SVC(probability=True, random_state=42)
    ]

    for clf in models:
        model_name = clf.__class__.__name__
        
        # 1. Standard features
        for method in methods:
            print("\n" + "=" * 70)
            run_name = f"{model_name}-{method}"
            print(f"RUNNING EXPERIMENT: {experiment_name} | MODEL: {model_name} | METHOD: {method}")
            preprocessor = TitanicPreprocessor(
                method=method,
                keep_name=False,
                numeric_features=["Age", "Fare"],
                ordinal_features=["Pclass", "Parch", "SibSp"],
                categorical_features=["Sex"],
            )
            print("=" * 70)
            run_experiment(experiment_name, run_name, X, y, clf, preprocessor)
            
        # 2. No ordinals (treating them as numeric or dropping? The original code put them in numeric_features)
        # Original code: numeric_features=["Age", "Fare", "Pclass", "SibSp", "Parch"], ordinal_features=[]
        for method in methods:
            print("\n" + "=" * 70)
            run_name = f"{model_name}-{method}-no_ordinals"
            print(f"RUNNING EXPERIMENT: {experiment_name} | MODEL: {model_name} | METHOD: {method} (No Ordinals)")
            preprocessor = TitanicPreprocessor(
                method=method,
                keep_name=False,
                numeric_features=["Age", "Fare", "Pclass", "SibSp", "Parch"],
                ordinal_features=[],
                categorical_features=["Sex"],
            )
            print("=" * 70)
            run_experiment(experiment_name, run_name, X, y, clf, preprocessor)

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"\nTo view MLflow UI, run:")
    print(f"  mlflow ui --backend-store-uri {MLFLOW_TRACKING_URI}")
    print("\nThen open: http://localhost:5000")


if __name__ == "__main__":
    main()
