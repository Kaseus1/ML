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
# Neumorphic Styling (Light-mode only, fixed colors)
# ───────────────────────────────────────────────────────────────
st.markdown("""
    <style>
    @keyframes gradientBG {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    html, body, .stApp {
        height: 100%;
        margin: 0;
        background: linear-gradient(120deg, #f0f4ff, #e0f7fa);
        background-size: 400% 400%;
        animation: gradientBG 20s ease infinite;
        font-family: 'Segoe UI', sans-serif;
        color: #2c3e50;
    }

    .block-container {
        max-width: 860px;
        margin: 2rem auto;
        padding: 2.5rem;
        border-radius: 20px;
        background: #ecf0f3;
        box-shadow: 8px 8px 20px #d1d9e6, -8px -8px 20px #ffffff;
    }

    h1 {
        font-size: 2.2rem;
        text-align: center;
        color: #34495e;
        margin-bottom: 0.5rem;
    }

    .stSubheader {
        color: #2c3e50;
    }

    .stButton > button {
        background: #4a90e2 !important;
        color: white !important;
        padding: 0.6rem 1.8rem;
        font-weight: 600;
        border-radius: 12px;
        border: none;
        box-shadow: 4px 4px 10px #c1c7d0, -4px -4px 10px #ffffff;
        transition: 0.3s ease;
    }

    .stButton > button:hover {
        background: #357ABD !important;
        transform: scale(1.05);
    }

    .stSelectbox, .stCheckbox, .stSlider, .stNumberInput {
        background-color: #ecf0f3 !important;
        border-radius: 10px;
        padding: 0.4rem;
        color: #333333;
        font-size: 0.95rem;
    }

    .los-result {
        padding: 1.2rem;
        margin-top: 2rem;
        border-radius: 18px;
        background: #ffffff;
        backdrop-filter: blur(10px);
        color: #2e7d32;
        font-weight: bold;
        font-size: 1.2rem;
        text-align: center;
        box-shadow: 6px 6px 12px #d1d9e6, -6px -6px 12px #ffffff;
    }
    </style>
""", unsafe_allow_html=True)

# ───────────────────────────────────────────────────────────────
# Title
# ───────────────────────────────────────────────────────────────
st.title("🏥 Hospital Length of Stay Predictor")
st.markdown("Use clinical data to predict if a patient’s hospital stay will be **Short**, **Medium**, or **Long**.")

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
# Prediction + Auto Scroll to Result
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

    # 👇 Scroll target + result display
    st.markdown('<a name="result"></a>', unsafe_allow_html=True)
    st.markdown(f"""
        <div class="los-result">
            ✅ Predicted Length of Stay: <strong>{result}</strong>
        </div>
    """, unsafe_allow_html=True)
    
    # 👇 Auto scroll to result
    st.markdown("""
        <script>
            window.location.href = "#result";
        </script>
    """, unsafe_allow_html=True)
