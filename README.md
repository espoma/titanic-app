# Titanic Survival Prediction — POC

A playful proof-of-concept ML project: explore the Titanic dataset, build a flexible preprocessing pipeline, run experiments with MLflow, and eventually deploy a Streamlit app to predict survival chances.

## The Idea

You've done the exploratory work, built a preprocessing strategy, trained some models, and tracked experiments. Now comes the fun part: **share it with the world** via a simple app that anyone can use.

This project walks through that full journey—from raw data to a deployable prediction app. We're using the [Titanic dataset](https://www.kaggle.com/competitions/titanic) from Kaggle because it's approachable for beginners yet rich enough to explore real ML workflows. The question: **Would you have survived the Titanic?**

## What We're Building

1. **Exploratory Data Analysis** (`notebooks/eda.ipynb`)  
   Visualizations, statistical summaries, and feature investigation to understand Age, Fare, Pclass, Sex, and other patterns.

2. **Flexible Preprocessing** (`src/data/preprocess.py`)  
   A scikit-learn compatible `TitanicPreprocessor` class supporting multiple strategies:
   - `basic`: drop unnecessary columns, mean imputation for numeric, mode for categorical, one-hot encode.
   - (Planned) `median`, `knn`, `drop`: alternative imputation and feature handling strategies.
   - Deterministic and reusable via `joblib` serialization.

3. **Experiment Tracking** (`src/models/mlflow/train*.py`)  
   Baseline trainer using MLflow to log cross-validation metrics, model artifacts, and parameters. All experiments are recorded and hosted on **[DagsHub](https://dagshub.com/espoma/titanic-app)**.
   - **MLflow Tracking Dashboard**: [View Experiments on DagsHub](https://dagshub.com/espoma/titanic-app.mlflow)

4. **Model Fine-Tuning** (`src/models/tuning/tune*.py`)  
   Optuna-driven hyperparameter optimization. The best pipelines are saved as artifacts and can be compared via the DagsHub MLflow UI.

5. **Streamlit App** (`src/app/app.py`)  
   A lightweight UI where users input passenger info (age, class, sex, etc.) and get a survival prediction from the best model.
   - **Live Demo**: [titanic-app-espoma.streamlit.app](https://titanic-app-espoma.streamlit.app)

## Project Structure

```text
titanic/
├── data/                   # Raw and processed data
│   ├── raw/                # Original train.csv and test.csv from Kaggle
│   └── processed/          # Data after basic cleaning/transformation
├── notebooks/              # Jupyter notebooks for exploration
│   └── eda.ipynb           # Main Exploratory Data Analysis
├── src/                    # Source code
│   ├── app/                # Streamlit application
│   ├── data/               # Data loading and preprocessing logic
│   ├── features/           # Custom feature engineering transformers
│   ├── models/             # Training and prediction scripts
│   │   ├── mlflow/         # MLflow experiment tracking scripts
│   │   ├── tuning/         # Hyperparameter optimization (Optuna)
│   │   └── wandb/          # Weights & Biases experiment tracking
│   └── config.py           # Project-wide configuration and paths
├── results/                # Local storage for models and metrics
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

## Exploratory Data Analysis

The initial exploration in `notebooks/eda.ipynb` revealed several key insights that guided the modeling strategy:

- **Missing Values**: Significant missing data was found in `Age` (~20%), `Cabin` (>75%), and `Embarked` (2 records). `Cabin` was dropped due to sparsity, while `Age` required robust imputation.
- **Survival Correlations**:
    - **Sex**: Females had a significantly higher survival rate (~74%) compared to males (~19%).
    - **Pclass**: First-class passengers were much more likely to survive than those in third class.
    - **Age**: Children had higher survival rates, while the elderly were more vulnerable.
- **Feature Redundancy**: `PassengerId` was identified as non-informative for baseline models and removed.
- **Feature Drop**: `Ticket` may contain helpful signals related to the position of each passenger, but this information was dropped for simplicity.
- **Class Balance**: The target variable `Survived` is relatively balanced (~38% survival rate), making Accuracy a viable primary metric, though Precision and Recall are tracked for a deeper understanding of model performance.

## Tech Stack

- **Data & ML**: pandas, scikit-learn, numpy, scipy
- **Exploration**: Jupyter, seaborn, matplotlib
- **Experiment Tracking**: MLflow + **DagsHub** (Remote Tracking Server)
- **Optimization**: Optuna
- **Serialization**: joblib
- **Deployment**: Streamlit (app) + Streamlit Cloud (hosting)
