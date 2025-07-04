import streamlit as st
import pandas as pd
import joblib
from xgboost import XGBClassifier

# ------------------------
# Load model and label encoder
# ------------------------
model = joblib.load("xgb_los_model.pkl")
label_encoder = joblib.load("los_label_encoder.pkl")

# ------------------------
# Page configuration
# ------------------------
st.set_page_config(
    page_title="Hospital LOS Predictor",
    layout="centered",
    page_icon="🏥",
)

# ------------------------
# LIGHT THEME CONSTANTS (no dark mode toggle)
# ------------------------
BG_GRADIENT   = "linear-gradient(to bottom right, #e3f2fd, #fce4ec)"
CARD_COLOR    = "rgba(255, 255, 255, 0.6)"
TEXT_COLOR    = "#212121"
BOX_SHADOW    = "0 10px 30px rgba(0, 0, 0, 0.1)"
SUCCESS_BG    = "rgba(232, 245, 233, 0.6)"
SUCCESS_TEXT  = "#2e7d32"

# Apply global style overrides
st.markdown(
    f"""
    <style>
    html, body {{
        background: {BG_GRADIENT};
        color: {TEXT_COLOR};
    }}
    .stApp {{ padding: 1rem; }}
    h1, h2 {{ color: {TEXT_COLOR}; }}
    .stButton > button {{
        background: rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        color: {TEXT_COLOR};
        font-weight: 600;
        padding: 0.6rem 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.25);
        backdrop-filter: blur(12px);
        box-shadow: {BOX_SHADOW};
    }}
    .stSelectbox, .stCheckbox, .stSlider, .stNumberInput {{
        background: rgba(255, 255, 255, 0.2) !important;
        border-radius: 12px;
        padding: 0.5rem;
        color: {TEXT_COLOR};
        backdrop-filter: blur(10px);
    }}
    .block-container {{
        max-width: 800px;
        margin: auto;
        padding: 2rem;
        border-radius: 25px;
        background: {CARD_COLOR};
        backdrop-filter: blur(15px);
        box-shadow: {BOX_SHADOW};
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------
# App title/intro
# ------------------------
st.title("🏥 Hospital Length of Stay Predictor")
st.markdown("Use patient clinical data to predict whether their stay will be **Short**, **Medium**, or **Long**.")

# ------------------------
# Prediction form
# ------------------------
with st.form("predict_form"):
    st.subheader("🧾 Patient Information")
    col1, col2 = st.columns(2)
    with col1:
        rcount     = st.slider("Recent Admissions", 0, 10, 1)
        gender     = st.selectbox("Gender", ["F", "M"])
        diagnosis  = st.selectbox("Secondary Diagnosis", ["None", "DX1", "DX2", "DX3"])
    with col2:
        hemo         = st.slider("Hemoglobin", 5.0, 20.0, 13.5)
        hematocrit   = st.slider("Hematocrit", 20.0, 60.0, 40.0)
        neutrophils  = st.slider("Neutrophils", 20.0, 90.0, 50.0)

    st.subheader("🩺 Clinical Conditions")
    col3, col4, col5 = st.columns(3)
    with col3:
        dialysis = st.checkbox("Dialysis End Stage")
        asthma   = st.checkbox("Asthma")
        irondef  = st.checkbox("Iron Deficiency")
    with col4:
        pneum          = st.checkbox("Pneumonia")
        substance      = st.checkbox("Substance Dependence")
        psychdisorder  = st.checkbox("Psych Disorder")
    with col5:
        depress   = st.checkbox("Depression")
        psychother = st.checkbox("Psychotherapy")
        fibrosis  = st.checkbox("Fibrosis")

    st.subheader("📊 Vitals & Labs")
    col6, col7, col8 = st.columns(3)
    with col6:
        sodium   = st.slider("Sodium", 120.0, 160.0, 140.0)
        glucose  = st.slider("Glucose", 50.0, 300.0, 100.0)
    with col7:
        bun        = st.slider("BUN", 5.0, 50.0, 15.0)
        creatinine = st.slider("Creatinine", 0.5, 5.0, 1.2)
    with col8:
        bmi         = st.slider("BMI", 10.0, 50.0, 22.0)
        pulse       = st.slider("Pulse", 40, 150, 70)
        respiration = st.slider("Respiration", 10, 40, 18)

    submit = st.form_submit_button("Predict LOS")

# ------------------------
# Prediction + animated output
# ------------------------
if submit:
    features = {
        'rcount': rcount,
        'gender': 0 if gender == 'F' else 1,
        'dialysisrenalendstage': int(dialysis),
        'asthma': int(asthma),
        'irondef': int(irondef),
        'pneum': int(pneum),
        'substancedependence': int(substance),
        'psychologicaldisordermajor': int(psychdisorder),
        'depress': int(depress),
        'psychother': int(psychother),
        'fibrosisandother': int(fibrosis),
        'malnutrition': 0,
        'hemo': hemo,
        'hematocrit': hematocrit,
        'neutrophils': neutrophils,
        'sodium': sodium,
        'glucose': glucose,
        'bloodureanitro': bun,
        'creatinine': creatinine,
        'bmi': bmi,
        'pulse': pulse,
        'respiration': respiration,
    }

    for dx in ['DX1', 'DX2', 'DX3']:
        features[f"secondarydiagnosisnonicd9_{dx}"] = 1 if diagnosis == dx else 0

    df = pd.DataFrame([features])
    for col in model.get_booster().feature_names:
        if col not in df.columns:
            df[col] = 0
    df = df[model.get_booster().feature_names]

    pred   = model.predict(df)[0]
    result = label_encoder.inverse_transform([pred])[0]

    st.markdown(
        f"""
        <style>
        @keyframes fadeInUp {{
            0% {{ opacity: 0; transform: translateY(20px); }}
            100% {{ opacity: 1; transform: translateY(0); }}
        }}
        .result-card {{
            padding: 1rem;
            margin-top: 1rem;
            border-radius: 18px;
            background: {SUCCESS_BG};
            backdrop-filter: blur(10px);
            box-shadow: 0 4px 10px rgba(0,0,0,0.15);
            animation: fadeInUp 0.6s ease-out;
        }}
        </style>
        <div class='result-card'>
            <h3 style='color: {SUCCESS_TEXT};'>✅ Predicted Length of Stay: <strong>{result}</strong></h3>
        </div>
        """,
        unsafe_allow_html=True,
    )


Dark-mode code removed — your script now runs exclusively in the light theme with the same layout, form, prediction logic, and animated result card.

Launch the updated app.py; everything should work cleanly without the top-right toggle.

