import os
import pandas as pd
import sys
import joblib
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
from config import MODELS_DIR, SUBMISSION_DIR
from data.load_data import load_test_data

best_model_name = "best_extratrees_pipeline.joblib"

best_model = joblib.load(os.path.join(MODELS_DIR, best_model_name))

test_raw = load_test_data()

y_pred = best_model.predict(test_raw)

y_submission = pd.DataFrame({"PassengerId": test_raw["PassengerId"], "Survived": y_pred})

y_submission.to_csv(os.path.join(SUBMISSION_DIR, f"submission_MODEL-{best_model_name.replace('.joblib', '')}.csv"), index=False)