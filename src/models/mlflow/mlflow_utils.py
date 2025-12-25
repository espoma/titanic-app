import mlflow
from mlflow.entities import ViewType
import os
import joblib
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate
from sklearn.metrics import accuracy_score
import subprocess


def get_or_create_experiment(experiment_name, tracking_uri):
    """
    Sets the tracking URI and retrieves the experiment ID, creating it if it doesn't exist.
    """
    mlflow.set_tracking_uri(tracking_uri)

    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_name)
        print(f"Created new experiment: {experiment_name} (ID: {experiment_id})")
    else:
        experiment_id = experiment.experiment_id
        print(f"Using existing experiment: {experiment_name} (ID: {experiment_id})")

    mlflow.set_experiment(experiment_name)
    return experiment_id


def run_already_exists(experiment_id, run_name, params=None):
    """
    Checks if a run with the given name (and optional params) already exists in the experiment.
    Returns True if it exists, False otherwise.
    """
    # Search for runs with the same name in this experiment
    query = f"tags.mlflow.runName = '{run_name}'"

    runs = mlflow.search_runs(
        experiment_ids=[experiment_id],
        filter_string=query,
        run_view_type=ViewType.ACTIVE_ONLY,
    )

    if len(runs) == 0:
        return False

    return True


def run_experiment(
    experiment_name, 
    run_name, 
    X, 
    y, 
    clf, 
    preprocessor, 
    TRACKING_URI, 
    MODELS_DIR,
    feature_transformer=None  # NEW: optional feature engineering step
):
    """
    Run a complete experiment with optional feature engineering.
    """
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(experiment_name)

    # Build pipeline dynamically
    print(f"\n2. Building pipeline...")
    steps = []
    
    # Add feature transformer if provided
    if feature_transformer is not None:
        steps.append(("features", feature_transformer))
        print(f"   Feature engineering: {feature_transformer.__class__.__name__}")
    
    # Add preprocessor and classifier
    steps.append(("preprocessor", preprocessor))
    steps.append(("clf", clf))
    
    pipeline = Pipeline(steps)
    print(f"   Preprocessor: {preprocessor.method}")
    print(f"   Classifier: {clf.__class__.__name__}")

    print(f"\n   Preprocessing config: {preprocessor.get_config()}")

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
        except Exception:
            pass

        # Log parameters
        params = {
            "model": clf.__class__.__name__,
            "preprocessing_method": preprocessor.method,
            "cv_folds": 5,
        }
        
        # Log feature transformer info if present
        if feature_transformer is not None:
            params["feature_transformer"] = feature_transformer.__class__.__name__
            if hasattr(feature_transformer, "get_params"):
                for key, value in feature_transformer.get_params().items():
                    params[f"feature_{key}"] = value
        
        mlflow.log_params(params)

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
        config_dict = {
            "method": preprocessor.method,
            "keep_name": preprocessor.keep_name,
            "numeric_features": preprocessor.numeric_features,
            "ordinal_features": preprocessor.ordinal_features,
            "categorical_features": preprocessor.categorical_features,
        }
        
        # Add feature transformer config if present
        if feature_transformer is not None:
            config_dict["feature_transformer"] = {
                "class": feature_transformer.__class__.__name__,
                "params": feature_transformer.get_params() if hasattr(feature_transformer, "get_params") else {}
            }
        
        mlflow.log_dict(config_dict, "pipeline_config.json")

        print("   ✓ Model logged to MLflow")

        # Get run info
        run_id = mlflow.active_run().info.run_id
        print(f"   Run ID: {run_id}")
