import streamlit as st
import pandas as pd
import os
import joblib
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
from config import BEST_MODELS_DIR

st.set_page_config(page_title="Titanic Survival Predictor", page_icon="🚢", layout="centered")

# Custom CSS for a better look
st.markdown(
    """
    <style>
    .stApp {
        background-image: linear-gradient(rgba(0, 0, 0, 0.6), rgba(0, 0, 0, 0.6)), url("https://upload.wikimedia.org/wikipedia/commons/f/fd/RMS_Titanic_3.jpg");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        color: white;
    }
    h1 {
        color: #ffffff;
        text-align: center;
        text-shadow: 2px 2px 4px #000000;
        font-family: 'Georgia', serif;
    }
    h2, h3 {
        color: #f0f2f6;
        text-align: center;
        text-shadow: 1px 1px 2px #000000;
    }
    .stNumberInput > label, .stSelectbox > label {
        color: white !important;
        font-weight: bold;
        text-shadow: 1px 1px 2px #000000;
        font-size: 1.1rem;
    }
    p {
        color: #e0e0e0;
        font-size: 1.1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🚢 Would you have survived the Titanic?")

st.subheader("What's your sitch?")

# Model = st.selectbox("Model", [model.split(".")[0] for model in os.listdir(BEST_MODELS_DIR) if model.endswith(".joblib") and "pipeline" in model])

Model = "best_gradientboosting_pipeline"
col1, col2 = st.columns(2)

with col1:
    Age = st.number_input("Age", min_value=0, max_value=120, value=30)
    Pclass = st.number_input("Class", min_value=1, max_value=3, value=1)
    SibSp = st.number_input("Siblings/Spouses", min_value=0, max_value=10, value=0)
    # Embarked = st.selectbox("Embarked", ["S", "C", "Q"])

with col2:
    Fare = st.number_input("Fare", min_value=0, max_value=500, value=30)
    ParCh = st.number_input("Parents/Children", min_value=0, max_value=10, value=0)
    Sex = st.selectbox("Sex", ["Male", "Female"])

best_model = joblib.load(os.path.join(BEST_MODELS_DIR, Model + ".joblib"))

input_data = pd.DataFrame({
    "Age": [Age],
    "Pclass": [Pclass],
    "SibSp": [SibSp],
    "Fare": [Fare],
    "Parch": [ParCh],
    "Sex": [Sex]
})

prediction = best_model.predict(input_data)[0]
survival_prob = best_model.predict_proba(input_data)[0][1]

if prediction == 1:
    st.success(f"You would have survived! (Probability: {survival_prob:.2%})")
else:
    st.error(f"You would NOT have survived. (Probability: {survival_prob:.2%})")


