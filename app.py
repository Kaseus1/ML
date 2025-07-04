import streamlit as st
import pandas as pd
import joblib
from xgboost import XGBClassifier

# ───────────────────────────────────────────────────────────────
# Configuration
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
# Styling & Animations
# ───────────────────────────────────────────────────────────────
bg_gradient  = "linear-gradient(to bottom right, #e3f2fd, #fce4ec)"
card_color   = "rgba(255, 255, 255, 0.60)"
text_color   = "#212121"
box_shadow   = "0 10px 30px rgba(0, 0, 0, 0.10)"
success_bg   = "rgba(232, 245, 233, 0.60)"
success_text = "#2e7d32"

st.markdown(f"""
    <style>
    @keyframes fadeInDown {{
      0% {{opacity: 0; transform: translateY(-20px);}}
      100% {{opacity: 1; transform: translateY(0);}}
    }}
    @keyframes fadeInUp {{
      0% {{opacity: 0; transform: translateY(20px);}}
      100% {{opacity: 1; transform: translateY(0);}}
    }}
    @keyframes glowPulse {{
      0% {{ box-shadow: 0 0 0px #81c784; }}
      50% {{ box-shadow: 0 0 20px #81c784; }}
      100% {{ box-shadow: 0 0 0px #81c784; }}
    }}
    @keyframes slideInUp {{
      0% {{ opacity: 0; transform: translateY(30px); }}
      100% {{ opacity: 1; transform: translateY(0); }}
    }}

    html, body {{
        background: {bg_gradient};
        color: {text_color};
    }}
    .stApp {{
        padding: 1rem;
    }}
    .block-container {{
        max-width: 800px;
        margin: auto;
        padding: 2rem;
        border-radius: 25px;
        background: {card_color};
        backdrop-filter: blur(15px);
        box-shadow: {box_shadow};
    }}
    h1 {{
        text-align: center;
        animation: fadeInDown 0.8s ease-out;
    }}
    h4 {{
        margin-top: 2rem;
        animation: fadeInUp 0.6s ease-out;
    }}
    .stButton > button {{
        background: rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        color: {text_color};
        font-weight: 600;
        padding: 0.6rem 1.5rem;
        border: 1px solid rgba(0, 0, 0, 0.05);
        backdrop-filter: blur(12px);
        box-shadow: {box_shadow};
        transition: all 0.3s ease;
    }}
    .stButton > button:hover {{
        transform: scale(1.04);
        box-shadow: 0 12px 24px rgba(0, 0, 0, 0.2);
    }}
    .stSelectbox, .stCheckbox, .stSlider, .stNumberInput {{
        background: rgba(255, 255, 255, 0.2) !important;
        border-radius: 12px;
        padding: 0.5rem;
        color: {text_color};
        backdrop-filter: blur(10px);
    }}
    .los-result {{
        padding: 1rem;
        margin-top: 1.5rem;
        border-radius: 18px;
        background: {success_bg};
        backdrop-filter: blur(10px);
        animation: slideInUp 0.8s ease-out, glowPulse 2s ease-in-out infinite;
    }}
    </style>
""", unsafe_allow_html=True)

# ───────────────────────────────────────────────────────────────
# App header
# ───────────────────────────────────────────────────────────────
st.title("🏥 Hospital Length of Stay Predictor")
st.markdown("Use patient clinical data to predict whether their stay will be **Short**, **Medium**, or **Long**.")

# ───────────────────────────────────────────────────────────────
# Prediction form
# ───────────────────────────────────────────────────────────────
with st.form("predict_form"):
    st.markdown("<h4>🧾 Patient Information</h4>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        rcount = st.slider("Recent Admissions", 0, 10, 1)
        gender = st.selectbox("Gender", ["F", "M"])
        diagnosis = st.selectbox("Secondary Diagnosis", ["None", "DX1", "DX2", "DX3"])
    with col2:
        hemo = st.slider("Hemoglobin", 5.0, 20.0, 13.5)
        hematocrit = st.slider("Hematocrit", 20.0, 60.0, 40.0)
        neutrophils = st.slider("Neutrophils", 20.0, 90.0, 50.0)

    st.markdown("<h4>🩺 Clinical Conditions</h4>", unsafe_allow_html=True)
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

    st.markdown("<h4>📊 Vitals & Labs</h4>", unsafe_allow_html=True)
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
# Prediction logic
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
            <h3 style='color: {success_text};'>
                ✅ Predicted Length of Stay: <strong>{result}</strong>
            </h3>
        </div>
    """, unsafe_allow_html=True)
