import optuna
import pandas as pd
import os
import sys
import warnings
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from data import load_train_data, TitanicPreprocessor

warnings.filterwarnings("ignore")

train_raw = load_train_data()

X = train_raw.drop(columns=["Survived"])
y = train_raw["Survived"]
print(f"    Raw features: {X.columns.tolist()}")
print(f"    Target distribution:\n{y.value_counts(normalize=True)}")

def objective(trial, X, y):
    # Suggest hyperparameters for the classifier
    n_estimators = trial.suggest_int("n_estimators", 100, 1000)
    max_depth = trial.suggest_int("max_depth", 10, 50)
    min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
    # min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 20)
    
    # Suggest hyperparameters for preprocessing
    method = trial.suggest_categorical("method", ["basic", "median_impute", "knn_impute"])

    # Define the classifier with OOB enabled
    clf = ExtraTreesClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        # min_samples_leaf=min_samples_leaf,
        bootstrap=True,  # Required for oob_score
        oob_score=True,
        random_state=42,
        n_jobs=4,
    )

    # Feature strategy selection
    feature_strategy = trial.suggest_categorical("feature_strategy", ["numeric_only", "with_ordinals"])
    
    if feature_strategy == "numeric_only":
        # Treat everything as numeric
        numeric_features = ["Age", "Fare", "Pclass", "SibSp", "Parch"]
        ordinal_features = []
    else:
        # Treat Pclass, SibSp, Parch as ordinal
        numeric_features = ["Age", "Fare"]
        ordinal_features = ["Pclass", "Parch", "SibSp"]

    # Define the preprocessor
    preprocessor = TitanicPreprocessor(
        method=method,
        keep_name=False,
        numeric_features=numeric_features,
        ordinal_features=ordinal_features,
        categorical_features=["Sex"],
    )

    # Build the pipeline
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", clf),
    ])

    # Fit the pipeline on the FULL dataset
    pipeline.fit(X, y)

    # Access the fitted classifier to get the OOB score
    # The classifier is the last step named 'classifier'
    oob_score = pipeline.named_steps["classifier"].oob_score_

    return oob_score


# Create study and optimize
study = optuna.create_study(direction="maximize")
study.optimize(lambda trial: objective(trial, X, y), n_trials=50)

print(f"\nBest OOB Score: {study.best_value:.4f}")
print(f"Best parameters: {study.best_params}")

# Re-train the best model on full data (just to verify/inspect)
best_params = study.best_params

# Decode the best feature strategy
if best_params["feature_strategy"] == "numeric_only":
    best_numeric = ["Age", "Fare", "Pclass", "SibSp", "Parch"]
    best_ordinal = []
else:
    best_numeric = ["Age", "Fare"]
    best_ordinal = ["Pclass", "Parch", "SibSp"]

# Reconstruct the best pipeline
best_clf = ExtraTreesClassifier(
    n_estimators=best_params["n_estimators"],
    max_depth=best_params["max_depth"],
    min_samples_split=best_params["min_samples_split"],
    bootstrap=True,
    oob_score=True,
    random_state=42,
    n_jobs=-1
)

best_preprocessor = TitanicPreprocessor(
    method=best_params["method"],
    keep_name=False,
    numeric_features=best_numeric,
    ordinal_features=best_ordinal,
    categorical_features=["Sex"],
)

final_pipeline = Pipeline([
    ("preprocessor", best_preprocessor),
    ("classifier", best_clf),
])

final_pipeline.fit(X, y)
final_oob = final_pipeline.named_steps["classifier"].oob_score_
print(f"Final Re-trained OOB Score: {final_oob:.4f}")

# Save the best model and parameters
import joblib
import json
from config import MODELS_DIR

os.makedirs(MODELS_DIR, exist_ok=True)

# Save pipeline
model_path = os.path.join(MODELS_DIR, "best_extratrees_pipeline.joblib")
joblib.dump(final_pipeline, model_path)
print(f"\nSaved best model to: {model_path}")

# Save parameters
params_path = os.path.join(MODELS_DIR, "best_extratrees_params.json")
with open(params_path, "w") as f:
    json.dump(best_params, f, indent=4)
print(f"Saved best parameters to: {params_path}")
