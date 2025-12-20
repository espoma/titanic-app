import streamlit as st
import pandas as pd
import os
import joblib
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../"))
from config import BEST_MODELS_DIR

st.set_page_config(page_title="Titanic Survival Predictor", page_icon="🚢", layout="wide")

# Custom CSS for a better look
st.markdown(
    """
    <style>
    .stApp {
        background-image: linear-gradient(rgba(0, 0, 0, 0.7), rgba(0, 0, 0, 0.7)), url("https://upload.wikimedia.org/wikipedia/commons/f/fd/RMS_Titanic_3.jpg");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        color: white;
    }
    .main-title {
        color: #ffffff;
        text-align: center;
        text-shadow: 3px 3px 6px #000000;
        font-family: 'Georgia', serif;
        font-size: 3.5rem;
        margin-bottom: 1rem;
    }
    .history-card {
        background: rgba(255, 255, 255, 0.1);
        padding: 20px;
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        backdrop-filter: blur(10px);
        margin-bottom: 20px;
    }
    .link-button {
        display: inline-block;
        padding: 10px 20px;
        margin: 10px 5px;
        border-radius: 8px;
        text-decoration: none;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .repo-link {
        background-color: #24292e;
        color: white !important;
        border: 1px solid #444;
    }
    .repo-link:hover {
        background-color: #444;
        transform: translateY(-2px);
    }
    .kaggle-link {
        background-color: #20beff;
        color: black !important;
    }
    .kaggle-link:hover {
        background-color: #00a0e0;
        transform: translateY(-2px);
    }
    .inflation-warning {
        font-size: 0.9rem;
        font-style: italic;
        color: #ffd700;
        margin-top: -15px;
        margin-bottom: 15px;
    }
    .project-note {
        font-size: 0.85rem;
        color: #cccccc;
        border-top: 1px solid rgba(255, 255, 255, 0.2);
        padding-top: 15px;
        margin-top: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<h1 class="main-title">🚢 Titanic Survival Predictor</h1>', unsafe_allow_html=True)

# Main layout: History on the left, App on the right
col_hist, col_spacer, col_app = st.columns([1, 0.1, 1.5])

with col_hist:
    st.markdown('<div class="history-card">', unsafe_allow_html=True)
    st.header("📜 A Brief History")
    st.markdown("""
    The **RMS Titanic** was a British passenger liner that sank in the North Atlantic Ocean on **April 15, 1912**, after striking an iceberg during her maiden voyage from Southampton to New York City.
    
    *   **The Tragedy:** Of the estimated **2,224** passengers and crew aboard, more than **1,500** died.
    *   **The Wreck:** The ship broke apart and foundered at 2:20 AM, approximately 370 miles southeast of Newfoundland.
    *   **The Legacy:** The disaster led to major changes in maritime safety regulations, including the requirement for enough lifeboats for everyone on board.
    """)
    
    st.markdown("### 🔗 Project Links")
    st.markdown(
        '<a href="https://github.com/espoma/titanic-app" class="link-button repo-link">⭐ View on GitHub</a>'
        '<a href="https://www.kaggle.com/competitions/titanic" class="link-button kaggle-link">📊 Kaggle Competition</a>',
        unsafe_allow_html=True
    )
    
    st.markdown('<div class="project-note">', unsafe_allow_html=True)
    st.markdown("""
    **About this project:**  
    This is a playful personal project by **[espoma](https://github.com/espoma)** to showcase a complete end-to-end ML lifecycle:
    - 🔍 **EDA:** Deep dive into passenger data.
    - 🛠️ **Preprocessing:** Custom pipelines for data cleaning.
    - 🧪 **Experiments:** Model comparison using MLflow.
    - 🎯 **Fine-tuning:** Hyperparameter optimization with Optuna.
    - 🚀 **Deployment:** This interactive Streamlit app.
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col_app:
    st.subheader("Would you have survived?")
    st.write("Enter your details below to see your chances.")

    # Model selection (hardcoded for now as per original script)
    Model = "best_gradientboosting_pipeline"
    
    form_col1, form_col2 = st.columns(2)
    
    with form_col1:
        Age = st.number_input("Age", min_value=0, max_value=120, value=30)
        Pclass = st.selectbox("Passenger Class", [1, 2, 3], index=0, help="1 = 1st, 2 = 2nd, 3 = 3rd")
        SibSp = st.number_input("Siblings/Spouses Aboard", min_value=0, max_value=10, value=0)

    with form_col2:
        Fare = st.number_input("Fare (1912 GBP)", min_value=0.0, max_value=600.0, value=30.0, step=1.0)
        
        # Inflation warning
        usd_val = Fare * 178
        eur_val = Fare * 168
        st.markdown(f'<p class="inflation-warning">⚠️ That\'s about <b>${usd_val:,.2f}</b> or <b>€{eur_val:,.2f}</b> in today\'s money! 💸</p>', unsafe_allow_html=True)
        
        ParCh = st.number_input("Parents/Children Aboard", min_value=0, max_value=10, value=0)
        Sex = st.selectbox("Sex", ["Male", "Female"])

    if st.button("🚢 Predict My Survival", use_container_width=True):
        try:
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
            
            st.markdown("---")
            if prediction == 1:
                st.success(f"### 🎉 You would have survived!\n**Survival Probability:** {survival_prob:.2%}")
                st.balloons()
            else:
                st.error(f"### 🧊 You would NOT have survived.\n**Survival Probability:** {survival_prob:.2%}")
        except Exception as e:
            st.error(f"Error loading model or making prediction: {e}")



