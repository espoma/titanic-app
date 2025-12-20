import streamlit as st
import pandas as pd
import os
import joblib
import sys
import numpy as np
import base64

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../"))
from config import BEST_MODELS_DIR

st.set_page_config(page_title="Titanic Adventure", page_icon="🚢", layout="wide")

def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_png_as_page_bg(bin_file):
    bin_str = get_base64_of_bin_file(bin_file)
    page_bg_img = '''
    <style>
    .stApp {
        background-image: linear-gradient(rgba(0, 20, 50, 0.4), rgba(0, 10, 30, 0.7)), url("data:image/png;base64,%s");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        color: white;
        font-family: 'Montserrat', sans-serif;
    }
    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)

# Set the custom background
bg_path = os.path.join(os.path.dirname(__file__), "assets/background.png")
if os.path.exists(bg_path):
    set_png_as_page_bg(bg_path)
else:
    # Fallback if image is missing
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #001432;
            color: white;
            font-family: 'Montserrat', sans-serif;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# Custom CSS for refined typography and components
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@300;400;600;700&display=swap');

    .main-title {
        color: #ffffff;
        text-align: center;
        text-shadow: 3px 3px 10px #003366;
        font-family: 'Montserrat', sans-serif;
        font-weight: 700;
        font-size: 4.5rem;
        margin-bottom: 0.5rem;
        padding-top: 2rem;
    }
    .sidebar .stMarkdown {
        color: #f0f8ff;
    }
    .history-section {
        background: rgba(255, 255, 255, 0.1);
        padding: 15px;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        backdrop-filter: blur(5px);
        margin-bottom: 20px;
    }
    .link-button {
        display: block;
        width: 100%;
        padding: 12px;
        margin: 10px 0;
        border-radius: 10px;
        text-decoration: none;
        font-weight: bold;
        text-align: center;
        transition: all 0.3s ease;
    }
    .repo-link {
        background: linear-gradient(135deg, #24292e 0%, #404448 100%);
        color: white !important;
        border: 2px solid #586069;
        font-size: 1.1rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
    .repo-link:hover {
        transform: scale(1.02);
        box-shadow: 0 6px 20px rgba(0,0,0,0.4);
        border-color: #ffffff;
    }
    .kaggle-link {
        background-color: rgba(32, 190, 255, 0.2);
        color: #20beff !important;
        border: 1px solid #20beff;
        font-size: 0.9rem;
    }
    .kaggle-link:hover {
        background-color: rgba(32, 190, 255, 0.3);
    }
    .inflation-warning {
        font-size: 0.95rem;
        font-style: italic;
        color: #00ffcc;
        margin-top: -10px;
        margin-bottom: 15px;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.5);
    }
    .percentile-info {
        font-size: 0.85rem;
        color: #ffd700;
        margin-top: -10px;
        margin-bottom: 10px;
    }
    /* Engaging Label Styling */
    .stNumberInput label, .stSelectbox label {
        color: #ffffff !important;
        font-weight: 600 !important;
        font-size: 1.2rem !important;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
        min-height: 2.5rem;
        display: flex;
        align-items: center;
    }
    /* Larger Dropdown Description */
    .port-description {
        font-size: 1.4rem !important;
        line-height: 1.6;
        color: #e0f2ff;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.8);
    }
    .port-schedule {
        font-size: 1.1rem;
        color: #00ffcc;
        font-style: italic;
    }
    /* Big Playful Button */
    .stButton > button {
        background-color: rgba(0, 100, 200, 0.4) !important;
        color: white !important;
        font-size: 1.8rem !important;
        font-weight: bold !important;
        padding: 20px 40px !important;
        border-radius: 15px !important;
        border: 2px solid #00ffcc !important;
        transition: all 0.4s ease !important;
        width: 100% !important;
        height: auto !important;
        text-transform: uppercase;
        letter-spacing: 2px;
        font-family: 'Montserrat', sans-serif;
    }
    .stButton > button:hover {
        background-color: #ff8c00 !important;
        color: white !important;
        border-color: #ffffff !important;
        transform: translateY(-5px);
        box-shadow: 0 10px 20px rgba(0,0,0,0.4);
    }
    /* Success/Error Contrast Fix */
    div[data-testid="stNotification"] {
        background-color: rgba(0, 0, 0, 0.8) !important;
        border: 2px solid #ffffff !important;
        color: white !important;
    }
    div[data-testid="stNotification"] h3 {
        color: white !important;
    }
    .insight-card {
        background: rgba(255, 255, 255, 0.05);
        padding: 20px;
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin-top: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

def get_fare_percentile(fare, pclass):
    # Approximate percentiles based on Titanic dataset
    thresholds = {
        1: [0, 35, 60, 100, 513],
        2: [0, 13, 15, 26, 74],
        3: [0, 7.75, 8.05, 15.5, 70]
    }
    p_values = [0, 25, 50, 75, 100]
    return np.interp(fare, thresholds[pclass], p_values)

# Sidebar Content
with st.sidebar:
    st.title("🚢 Titanic Archive")
    
    st.markdown('<div class="history-section">', unsafe_allow_html=True)
    st.subheader("📜 The Journey")
    st.markdown("""
    The **RMS Titanic** was a British passenger liner that sank in the North Atlantic Ocean on **April 15, 1912**.
    
    *   **Casualties:** Over **1,500** souls lost.
    *   **Passengers:** ~2,224 on board.
    *   **Location:** 370 miles SE of Newfoundland.
    *   **Cause:** Iceberg collision at 11:40 PM.
    """)
    st.markdown('</div>', unsafe_allow_html=True)

    st.subheader("📊 Data Source")
    st.markdown("""
    The data for this project is sourced from the famous **[Kaggle Titanic Competition](https://www.kaggle.com/competitions/titanic)**. 
    It's a classic dataset used to teach machine learning fundamentals.
    """)
    
    st.subheader("👨‍💻 Project Info")
    st.markdown("""
    This is a playful personal project by **[espoma](https://github.com/espoma)** showcasing a full ML pipeline:
    - 🔍 EDA & Cleaning
    - 🧪 MLflow Tracking
    - 🎯 Optuna Tuning
    - 🚀 Streamlit Deploy
    """)
    
    st.markdown("### 🔗 Quick Links")
    st.markdown(
        '<a href="https://github.com/espoma/titanic" class="link-button repo-link">🚀 View GitHub Repo</a>'
        '<a href="https://www.kaggle.com/competitions/titanic" class="link-button kaggle-link">Kaggle Competition</a>',
        unsafe_allow_html=True
    )

# Main Page
st.markdown('<h1 class="main-title">🌊 Your Titanic Adventure Awaits!</h1>', unsafe_allow_html=True)

col_left, col_mid, col_right = st.columns([1, 4, 1])

with col_mid:
    # Row 1
    r1c1, r1c2 = st.columns(2)
    with r1c1:
        Age = st.number_input("How old are you?", min_value=0, max_value=120, value=30, help="Age in years")
    with r1c2:
        Fare = st.number_input("What is your fare?", min_value=0.0, max_value=600.0, value=30.0, step=1.0, help="Fare in 1912 GBP")
        # Inflation warning
        usd_val = Fare * 178
        eur_val = Fare * 168
        st.markdown(f'<p class="inflation-warning">Watch out! This corresponds to <b>${usd_val:,.0f}</b> or <b>€{eur_val:,.0f}</b> in today\'s money! 💸</p>', unsafe_allow_html=True)

    # Row 2
    r2c1, r2c2 = st.columns(2)
    with r2c1:
        class_map = {"First": 1, "Second": 2, "Third": 3}
        Pclass_label = st.selectbox("Which class are you boarding?", list(class_map.keys()), index=0)
        Pclass = class_map[Pclass_label]
        
        # Percentile info - Only show if above 85%
        percentile = get_fare_percentile(Fare, Pclass)
        if percentile > 85:
            st.markdown(f'<p class="percentile-info">📊 Your fare is in the top <b>{100-percentile:.0f}%</b> of {Pclass_label} class.</p>', unsafe_allow_html=True)

    with r2c2:
        ParCh = st.number_input("How many parents/children are you boarding with?", min_value=0, max_value=10, value=0)

    # Row 3
    r3c1, r3c2 = st.columns(2)
    with r3c1:
        SibSp = st.number_input("How many siblings/spouses are you boarding with?", min_value=0, max_value=10, value=0)
    with r3c2:
        Sex = st.selectbox("Your gender?", ["Male", "Female"])

    # Row 4: Embarkation & Map
    st.markdown("---")
    
    # Port Data
    port_info = {
        "Southampton, UK": {
            "desc": "The Titanic's main departure point, bustling with activity as the world's largest ship prepared for her maiden voyage.",
            "schedule": "Reached: April 4, 12:00 AM | Embarked: April 10, 9:30 AM - 11:30 AM",
            "coords": (50.9097, -1.4044)
        },
        "Cherbourg, France": {
            "desc": "A brief stop in France to pick up wealthy continental passengers, many of whom arrived via the 'Titanic Special' train from Paris.",
            "schedule": "Reached: April 10, 6:30 PM | Embarked: 6:30 PM - 8:00 PM",
            "coords": (49.6337, -1.6221)
        },
        "Queenstown, Ireland": {
            "desc": "The final port of call before the open Atlantic. Many Irish emigrants boarded here, hoping for a new life in America.",
            "schedule": "Reached: April 11, 11:30 AM | Embarked: 11:30 AM - 1:30 PM",
            "coords": (51.8503, -8.2943)
        }
    }

    r4c1, r4c2 = st.columns([1, 1])
    with r4c1:
        Embarked = st.selectbox(
            "Where are you boarding from?",
            list(port_info.keys())
        )
        st.markdown(f'<p class="port-description">{port_info[Embarked]["desc"]}</p>', unsafe_allow_html=True)
        st.markdown(f'<p class="port-schedule">🕒 {port_info[Embarked]["schedule"]}</p>', unsafe_allow_html=True)
    
    with r4c2:
        lat, lon = port_info[Embarked]["coords"]
        map_url = f"https://maps.google.com/maps?q={lat},{lon}&z=10&output=embed"
        st.components.v1.iframe(map_url, height=250)

    st.markdown("<br>", unsafe_allow_html=True)
    
    if st.button("🚢 Calculate My Chances", use_container_width=True):
        Model = "best_gradientboosting_pipeline"
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

            # Survival Insights Section
            st.markdown('<div class="insight-card">', unsafe_allow_html=True)
            st.subheader("💡 Survival Insights")
            st.markdown(f"""
            Based on historical data and our machine learning model, several key factors played a crucial role in survival chances:
            
            *   **Gender & Age:** The "women and children first" policy was strictly enforced. Women had a significantly higher survival rate (~74%) compared to men (~19%).
            *   **Socio-Economic Status:** Passengers in **{Pclass_label} Class** had different survival rates. First-class passengers were prioritized for lifeboats and had better access to the upper decks.
            *   **Family Size:** Having a small family (1-3 members) often helped, as people could stay together. However, very large families sometimes struggled to coordinate during the chaos.
            *   **Location:** Proximity to the Boat Deck was life-saving. Those boarding from **{Embarked.split(',')[0]}** joined a diverse group of passengers with varying fates.
            
            *Your predicted probability of **{survival_prob:.2%}** reflects how these factors combined for your specific profile.*
            """)
            st.markdown('</div>', unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Error loading model: {e}")





