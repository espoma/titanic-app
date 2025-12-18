import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

RAW_TRAIN_PATH = os.path.join(RAW_DIR, "train.csv")
RAW_TEST_PATH = os.path.join(RAW_DIR, "test.csv")

MODELS_DIR = os.path.join(PROJECT_ROOT, "results", "models")
BEST_MODELS_DIR = os.path.join(MODELS_DIR, "best")
SUBMISSION_DIR = os.path.join(PROJECT_ROOT, "results", "submissions")

MLFLOW_TRACKING_URI = os.path.join(PROJECT_ROOT, "results", "mlflow")

