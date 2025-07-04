import streamlit as st
import pandas as pd
import joblib
from xgboost import XGBClassifier

# ───────────────────────────────────────────────────────────────
# App Configuration
# ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Hospital LOS Predictor",
    layout="centered",
    page_icon="🏥"
)

# Load model and label encoder
model = joblib.load("xgb_los_model.pkl")
label_encoder = joblib.load("los_label_encoder.pkl")

# ───────────────────────────────────────────────────────────────
# Enhanced Neumorphic Styling
# ───────────────────────────────────────────────────────────────
st.markdown("""
    <style>
    html, body, .stApp {
        background: #e0e5ec;
        font-family: 'Segoe UI', sans-serif;
        color: #2c3e50;
    }

    .block-container {
        max-width: 860px;
        margin: 2rem auto;
        padding: 2.5rem;
        border-radius: 20px;
        background: #e0e5ec;
        box-shadow: 9px 9px 16px #a3b1c6,
                    -9px -9px 16px #ffffff;
    }

    h1 {
        font-size: 2.4rem;
        text-align: center;
        color: #3f51b5;
        margin-bottom: 1rem;
    }

    .stSubheader {
        color: #455a64;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }

    .stButton > button {
        background: #e0e5ec;
        color: #3f51b5;
        padding: 0.7rem 2rem;
        font-weight: bold;
        border-radius: 12px;
        border: none;
        box-shadow: 5px 5px 10px #babecc,
                    -5px -5px 10px #ffffff;
        transition: all 0.3s ease;
    }

    .stButton > button:hover {
        background: #d4dae3;
        transform: translateY(-2px);
        box-shadow: 3px 3px 6px #babecc,
                    -3px -3px 6px #ffffff;
    }

    .stSelectbox, .stCheckbox, .stSlider, .stNumberInput {
        background-color: #e0e5ec !important;
        border-radius: 12px;
        padding: 0.4rem;
        color: #2c3e50;
        font-size: 0.95rem;
        box-shadow: inset 3px 3px 6px #babecc,
                    inset -3px -3px 6px #ffffff;
    }

    .los-result {
        padding: 1.5rem;
        margin-top: 2rem;
        border-radius: 20px;
        background: #e0e5ec;
        color: #2e7d32;
        font-weight: bold;
        font-size: 1.3rem;
        text-align: center;
        box-shadow: inset 4px 4px 8px #babecc,
                    inset -4px -4px 8px #ffffff;
    }

    /* Form labels */
    label {
        font-weight: 500;
        color: #455a64;
    }
    </style>
""", unsafe_allow_html=True)

# ───────────────────────────────────────────────────────────────
# Title
# ───────────────────────────────────────────────────────────────
st.title("🏥 Hospital Length of Stay Predictor")
st.markdown("Use clinical inputs to predict if a patient’s hospital stay will be **Short**, **Medium**, or **Long**.")

# ───────────────────────────────────────────────────────────────
# Form
# ───────────────────────────────────────────────────────────────
with st.form("predict_form"):
    st.subheader("🧾 Patient Information")
    col1, col2 = st.columns(2)
    with col1:
        rcount = st.slider("Recent Admissions", 0, 10, 1)
        gender = st.selectbox("Gender", ["F", "M"])
        diagnosis = st.selectbox("Secondary Diagnosis", ["None", "DX1", "DX2", "DX3"])
    with col2:
        hemo = st.slider("Hemoglobin", 5.0, 20.0, 13.5)
        hematocrit = st.slider("Hematocrit", 20.0, 60.0, 40.0)
        neutrophils = st.slider("Neutrophils", 20.0, 90.0, 50.0)

    st.subheader("🩺 Clinical Conditions")
    col3, col4, col5 = st.columns(3)
    with col3:
        dialysis = st.checkbox("Dialysis End Stage")
        asthma = st.checkbox("Asthma")
        irondef = st.checkbox("Iron Deficiency")
    with col4:
        pneum = st.checkbox("Pneumonia")
        substance = st.checkbox("Substance Dependence")
        psychdisord = st.checkbox("Psych Disorder")
    with col5:
        depress = st.checkbox("Depression")
        psychother = st.checkbox("Psychotherapy")
        fibrosis = st.checkbox("Fibrosis")

    st.subheader("📊 Vitals & Labs")
    col6, col7, col8 = st.columns(3)
    with col6:
        sodium = st.slider("Sodium", 120.0, 160.0, 140.0)
        glucose = st.slider("Glucose", 50.0, 300.0, 100.0)
    with col7:
        bun = st.slider("BUN", 5.0, 50.0, 15.0)
        creatinine = st.slider("Creatinine", 0.5, 5.0, 1.2)
    with col8:
        bmi = st.slider("BMI", 10.0, 50.0, 22.0)
        pulse = st.slider("Pulse", 40, 150, 70)
        respiration = st.slider("Respiration", 10, 40, 18)

    submitted = st.form_submit_button("Predict LOS")

# ───────────────────────────────────────────────────────────────
# Prediction Logic
# ───────────────────────────────────────────────────────────────
if submitted:
    data = {
        "rcount": rcount,
        "gender": 0 if gender == "F" else 1,
        "dialysisrenalendstage": int(dialysis),
        "asthma": int(asthma),
        "irondef": int(irondef),
        "pneum": int(pneum),
        "substancedependence": int(substance),
        "psychologicaldisordermajor": int(psychdisord),
        "depress": int(depress),
        "psychother": int(psychother),
        "fibrosisandother": int(fibrosis),
        "malnutrition": 0,
        "hemo": hemo,
        "hematocrit": hematocrit,
        "neutrophils": neutrophils,
        "sodium": sodium,
        "glucose": glucose,
        "bloodureanitro": bun,
        "creatinine": creatinine,
        "bmi": bmi,
        "pulse": pulse,
        "respiration": respiration,
    }

    for dx in ["DX1", "DX2", "DX3"]:
        data[f"secondarydiagnosisnonicd9_{dx}"] = 1 if diagnosis == dx else 0

    input_df = pd.DataFrame([data])
    for feat in model.get_booster().feature_names:
        if feat not in input_df.columns:
            input_df[feat] = 0
    input_df = input_df[model.get_booster().feature_names]

    pred = model.predict(input_df)[0]
    result = label_encoder.inverse_transform([pred])[0]

    st.markdown(f"""
        <div class="los-result">
            ✅ Predicted Length of Stay: <strong>{result}</strong>
        </div>
    """, unsafe_allow_html=True)
