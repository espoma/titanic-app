import optuna
import pandas as pd
import os
import sys
import warnings
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from data.load_data import load_train_data, load_test_data
from data.preprocess import TitanicPreprocessor

warnings.filterwarnings("ignore")

train_raw = load_train_data()

X = train_raw.drop(columns=["Survived"])
y = train_raw["Survived"]
print(f"    Raw features: {X.columns.tolist()}")
print(f"    Target distribution:\n{y.value_counts(normalize=True)}")

# Define consistent cross-validation strategy
CV_FOLDS = 5
cv_strategy = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)

def objective(trial, X, y):
    # Suggest hyperparameters for the classifier
    n_estimators = trial.suggest_int("n_estimators", 100, 1000)
    max_depth = trial.suggest_int("max_depth", 3, 10)  # GradientBoosting typically uses shallower trees
    min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
    learning_rate = trial.suggest_float("learning_rate", 0.01, 0.3, log=True)
    subsample = trial.suggest_float("subsample", 0.6, 1.0)
    
    # Suggest hyperparameters for preprocessing
    method = trial.suggest_categorical("method", ["basic", "median_impute", "knn_impute"])

    # Define the classifier
    clf = GradientBoostingClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        learning_rate=learning_rate,
        subsample=subsample,
        random_state=42,
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

    # Use cross-validation for consistent evaluation
    cv_scores = cross_val_score(pipeline, X, y, cv=cv_strategy, scoring="accuracy")
    
    return cv_scores.mean()


# Create study and optimize
study = optuna.create_study(direction="maximize")
study.optimize(lambda trial: objective(trial, X, y), n_trials=50)

print(f"\nBest CV Score: {study.best_value:.4f}")
print(f"Best parameters: {study.best_params}")

# Re-train the best model on full data
best_params = study.best_params

# Decode the best feature strategy
if best_params["feature_strategy"] == "numeric_only":
    best_numeric = ["Age", "Fare", "Pclass", "SibSp", "Parch"]
    best_ordinal = []
else:
    best_numeric = ["Age", "Fare"]
    best_ordinal = ["Pclass", "Parch", "SibSp"]

# Reconstruct the best pipeline
best_clf = GradientBoostingClassifier(
    n_estimators=best_params["n_estimators"],
    max_depth=best_params["max_depth"],
    min_samples_split=best_params["min_samples_split"],
    learning_rate=best_params["learning_rate"],
    subsample=best_params["subsample"],
    random_state=42,
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

# Fit on full data for final model
final_pipeline.fit(X, y)

# Verify with CV on final pipeline
final_cv_scores = cross_val_score(final_pipeline, X, y, cv=cv_strategy, scoring="accuracy")
print(f"Final CV Score (verification): {final_cv_scores.mean():.4f} (+/- {final_cv_scores.std():.4f})")

# Save the best model and parameters
import joblib
import json
from config import MODELS_DIR

os.makedirs(MODELS_DIR, exist_ok=True)

# Save pipeline
model_path = os.path.join(MODELS_DIR, "best_gradientboosting_pipeline.joblib")
joblib.dump(final_pipeline, model_path)
print(f"\nSaved best model to: {model_path}")

# Save parameters
params_path = os.path.join(MODELS_DIR, "best_gradientboosting_params.json")
with open(params_path, "w") as f:
    json.dump(best_params, f, indent=4)
print(f"Saved best parameters to: {params_path}")
