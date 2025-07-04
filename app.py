import streamlit as st
import pandas as pd
import joblib
from xgboost import XGBClassifier

# Load model and label encoder
model = joblib.load("xgb_los_model.pkl")
label_encoder = joblib.load("los_label_encoder.pkl")

# Page config
st.set_page_config(page_title="Hospital LOS Predictor", layout="centered", page_icon="🏥")

# Render Switch Toggle UI
st.markdown("""
<style>
.toggle-switch {
  position: relative;
  width: 60px;
  display: inline-block;
}
.toggle-switch input {
  opacity: 0;
  width: 0;
  height: 0;
}
.slider-switch {
  position: absolute;
  cursor: pointer;
  background-color: #ccc;
  border-radius: 34px;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  transition: .4s;
}
.slider-switch:before {
  position: absolute;
  content: "";
  height: 26px;
  width: 26px;
  left: 4px;
  bottom: 4px;
  background-color: white;
  border-radius: 50%;
  transition: .4s;
}
input:checked + .slider-switch {
  background-color: #2196F3;
}
input:checked + .slider-switch:before {
  transform: translateX(26px);
}
</style>

<div style='display:flex; align-items:center; gap:10px; margin-top: 10px; margin-bottom: 20px;'>
    <label class="toggle-switch">
        <input type="checkbox" id="dark-toggle">
        <span class="slider-switch"></span>
    </label>
    <label for="dark-toggle" style='font-weight: bold; font-size: 1rem;'>🌙 Dark Mode</label>
</div>
""", unsafe_allow_html=True)

# Streamlit input
dark_mode = st.checkbox("Dark Mode Toggle (Hidden)", value=False, label_visibility="collapsed")

# Sync custom toggle with Streamlit
st.markdown(f"""
<script>
const checkbox = window.parent.document.querySelector('input#dark-toggle');
checkbox.checked = {str(dark_mode).lower()};
checkbox.onchange = () => {{
    const streamlitInput = window.parent.document.querySelector('input[data-testid="stCheckbox-input"]');
    streamlitInput.checked = checkbox.checked;
    streamlitInput.dispatchEvent(new Event('change'));
}};
</script>
""", unsafe_allow_html=True)

# Theme styling
if dark_mode:
    bg_gradient = "linear-gradient(to bottom right, #121212, #1e1e1e)"
    card_color = "rgba(30, 30, 30, 0.85)"
    text_color = "#ffffff"
    box_shadow = "0 4px 12px rgba(255, 255, 255, 0.05)"
    success_bg = "rgba(56, 142, 60, 0.2)"
    success_text = "#81c784"
else:
    bg_gradient = "linear-gradient(to bottom right, #e3f2fd, #fce4ec)"
    card_color = "rgba(255, 255, 255, 0.6)"
    text_color = "#212121"
    box_shadow = "0 10px 30px rgba(0, 0, 0, 0.1)"
    success_bg = "rgba(232, 245, 233, 0.6)"
    success_text = "#2e7d32"

# Dynamic theme CSS
st.markdown(f"""
    <style>
    html, body {{
        background: {bg_gradient};
        font-family: 'Segoe UI', sans-serif;
        color: {text_color};
    }}
    .stApp {{
        padding: 1rem;
        animation: fadeIn 1.2s ease-in-out;
    }}
    h1, h2 {{
        color: {text_color};
    }}
    .stButton > button {{
        background: rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        color: {text_color};
        font-weight: 600;
        padding: 0.6rem 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.2);
        backdrop-filter: blur(12px);
        box-shadow: {box_shadow};
        transition: all 0.3s ease;
    }}
    .stSelectbox, .stCheckbox, .stSlider, .stNumberInput {{
        background: rgba(255, 255, 255, 0.2) !important;
        border-radius: 12px;
        padding: 0.5rem;
        color: {text_color};
        backdrop-filter: blur(10px);
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
    </style>
""", unsafe_allow_html=True)

# Title and description
st.title("🏥 Hospital Length of Stay Predictor")
st.markdown("Use patient clinical data to predict whether their stay will be **Short**, **Medium**, or **Long**.")

# === FORM START ===
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

# === PREDICTION RESULT ===
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

    st.markdown(f"""
        <div style='padding: 1rem; margin-top: 1rem; border-radius: 18px;
            background: {success_bg};
            backdrop-filter: blur(10px);
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.2);'>
            <h3 style='color: {success_text};'>✅ Predicted Length of Stay: <strong>{result}</strong></h3>
        </div>
    """, unsafe_allow_html=True)
