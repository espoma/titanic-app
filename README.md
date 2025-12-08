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

3. **Experiment Tracking** (`src/models/train.py`)  
   Baseline trainer using MLflow to log cross-validation metrics, model artifacts, and parameters. Compare different preprocessing strategies and models side-by-side.

4. **Streamlit App** (upcoming)  
   A lightweight UI where users input passenger info (age, class, sex, etc.) and get a survival prediction from the best model.

## Tech Stack

- **Data & ML**: pandas, scikit-learn, numpy, scipy
- **Exploration**: Jupyter, seaborn, matplotlib
- **Experiment Tracking**: MLflow (local file backend)
- **Serialization**: joblib
- **Deployment**: Streamlit (app) + Streamlit Cloud (hosting)

