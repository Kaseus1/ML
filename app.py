import streamlit as st
import pandas as pd
import joblib
from xgboost import XGBClassifier

# Load model and label encoder
model = joblib.load("xgb_los_model.pkl")
label_encoder = joblib.load("los_label_encoder.pkl")

# Page configuration
st.set_page_config(page_title="Hospital LOS Predictor", layout="centered", page_icon="🏥")

# Modern Glassmorphism Style
st.markdown("""
    <style>
    html, body {
        background: linear-gradient(to right, #f0f4f8, #e8edf3);
        font-family: 'Segoe UI', sans-serif;
        color: #212121;
        transition: all 0.3s ease-in-out;
    }

    .stApp {
        padding: 1rem;
        animation: fadeIn 1.2s ease-in-out;
    }

    h1, h2 {
        color: #1a237e;
    }

    .stButton > button {
        background: linear-gradient(145deg, #64b5f6, #42a5f5);
        border-radius: 10px;
        border: none;
        color: white;
        padding: 0.6rem 1.5rem;
        font-weight: 600;
        box-shadow: 0 4px 8px rgba(66, 165, 245, 0.3);
        transition: all 0.3s ease;
    }

    .stButton > button:hover {
        background: linear-gradient(145deg, #42a5f5, #2196f3);
        transform: scale(1.03);
        box-shadow: 0 6px 12px rgba(33, 150, 243, 0.4);
    }

    .stSelectbox, .stCheckbox, .stSlider, .stNumberInput {
        background: rgba(255, 255, 255, 0.8) !important;
        border-radius: 12px;
        padding: 0.5rem;
        backdrop-filter: blur(8px);
        box-shadow: 0 4px 10px rgba(0,0,0,0.06);
        transition: all 0.2s ease-in-out;
    }

    .stSelectbox:hover, .stCheckbox:hover, .stSlider:hover {
        transform: scale(1.01);
    }

    .block-container {
        max-width: 800px;
        margin: auto;
        padding: 2rem;
        border-radius: 20px;
        background: rgba(255, 255, 255, 0.75);
        backdrop-filter: blur(10px);
        box-shadow: 0 10px 20px rgba(0,0,0,0.08);
        animation: fadeInUp 1s ease;
    }

    @keyframes fadeIn {
        0% { opacity: 0; transform: scale(0.98); }
        100% { opacity: 1; transform: scale(1); }
    }

    @keyframes fadeInUp {
        0% { opacity: 0; transform: translateY(20px); }
        100% { opacity: 1; transform: translateY(0); }
    }
    </style>
""", unsafe_allow_html=True)

# Title and description
st.title("🏥 Hospital Length of Stay Predictor")
st.markdown("Use patient clinical data to predict whether their stay will be **Short**, **Medium**, or **Long**.")

# Form input
with st.form("predict_form"):
    st.subheader("🧾 Patient Information")

    col1, col2 = st.columns(2)
    with col1:
        rcount = st.slider("Recent Admissions", 0, 10, 1)
        gender = st.selectbox("Gender", ["F", "M"])
        diagnosis = st.selectbox("Secondary Diagnosis", ['None', 'DX1', 'DX2', 'DX3'])

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
        psychdisorder = st.checkbox("Psych Disorder")

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

if submitted:
    data = {
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
        data[f"secondarydiagnosisnonicd9_{dx}"] = 1 if diagnosis == dx else 0

    input_df = pd.DataFrame([data])
    for col in model.get_booster().feature_names:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[model.get_booster().feature_names]

    pred = model.predict(input_df)[0]
    result = label_encoder.inverse_transform([pred])[0]

    # Styled prediction result
    st.markdown(f"""
        <div style='padding: 1rem; margin-top: 1rem; border-radius: 15px;
            background: rgba(232, 245, 233, 0.85);
            backdrop-filter: blur(6px);
            box-shadow: 0 4px 10px rgba(76, 175, 80, 0.3);
            animation: fadeIn 1s ease-in-out;'>
            <h3 style='color: #2e7d32;'>✅ Predicted Length of Stay: <strong>{result}</strong></h3>
        </div>
    """, unsafe_allow_html=True)
