import optuna
import pandas as pd
import os
import sys
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from data import load_train_data, TitanicPreprocessor
from features import FamilySizeTransformer

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
    # Suggest hyperparameters for Random Forest
    n_estimators = trial.suggest_int("n_estimators", 100, 1000)
    max_depth = trial.suggest_int("max_depth", 5, 30)
    min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 10)
    
    # Suggest hyperparameters for KNN Imputer (if we wanted to tune it, but request says "knn impute")
    # We will stick to fixed knn_impute as requested, but could tune n_neighbors if desired.
    # method = "knn_impute" 

    # Define the classifier
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=42,
        n_jobs=4,
    )

    # Feature strategy selection
    # We always use FamilySize and Embarked as requested
    
    # Feature Engineering
    feature_transformer = FamilySizeTransformer(drop_original=True, include_self=True)

    # Define the preprocessor
    # FamilySize is ordinal, Embarked is categorical
    preprocessor = TitanicPreprocessor(
        method="knn_impute",  # Fixed as requested
        keep_name=False,
        numeric_features=["Age", "Fare"],
        ordinal_features=["Pclass", "FamilySize"],
        categorical_features=["Sex", "Embarked"],
    )

    # Build the pipeline
    pipeline = Pipeline([
        ("features", feature_transformer),
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

# Reconstruct the best pipeline
best_clf = RandomForestClassifier(
    n_estimators=best_params["n_estimators"],
    max_depth=best_params["max_depth"],
    min_samples_split=best_params["min_samples_split"],
    min_samples_leaf=best_params["min_samples_leaf"],
    random_state=42,
    n_jobs=-1
)

best_feature_transformer = FamilySizeTransformer(drop_original=True, include_self=True)

best_preprocessor = TitanicPreprocessor(
    method="knn_impute",
    keep_name=False,
    numeric_features=["Age", "Fare"],
    ordinal_features=["Pclass", "FamilySize"],
    categorical_features=["Sex", "Embarked"],
)

final_pipeline = Pipeline([
    ("features", best_feature_transformer),
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
model_path = os.path.join(MODELS_DIR, "best_rf_knn_fe_embarked_pipeline.joblib")
joblib.dump(final_pipeline, model_path)
print(f"\nSaved best model to: {model_path}")

# Save parameters
params_path = os.path.join(MODELS_DIR, "best_rf_knn_fe_embarked_params.json")
with open(params_path, "w") as f:
    json.dump(best_params, f, indent=4)
print(f"Saved best parameters to: {params_path}")
