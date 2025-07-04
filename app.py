import streamlit as st import pandas as pd import joblib from xgboost import XGBClassifier

Load model and label encoder

model = joblib.load("xgb_los_model.pkl") label_encoder = joblib.load("los_label_encoder.pkl")

Page config

st.set_page_config(page_title="Hospital LOS Predictor", layout="centered", page_icon="🏥")

--- TOP‑RIGHT DARK MODE TOGGLE (FUNCTIONAL) ---

Inject CSS that both styles and positions the native st.toggle as a floating switch

st.markdown(""" <style> /* Position and container styling */ div[data-testid="stToggle"] { position: fixed !important; top: 15px; right: 25px; z-index: 9999; display: flex; align-items: center; padding: 6px 12px; gap: 8px; background: rgba(255, 255, 255, 0.12); border-radius: 25px; backdrop-filter: blur(10px); box-shadow: 0 4px 12px rgba(0,0,0,0.12); }

/* Switch track */
div[data-testid="stToggle"] > div:first-child {
    width: 50px;
    height: 26px;
    background: #c6c6c6;
    border-radius: 50px;
    position: relative;
    transition: background 0.3s ease;
}

/* Knob */
div[data-testid="stToggle"] > div:first-child::after {
    content: "";
    position: absolute;
    width: 22px;
    height: 22px;
    border-radius: 50%;
    top: 2px;
    left: 2px;
    background: #ffffff;
    transition: transform 0.3s ease;
}

/* When checked */
div[data-testid="stToggle"] input:checked + div {
    background: #2196F3;
}
div[data-testid="stToggle"] input:checked + div::after {
    transform: translateX(24px);
}

/* Label next to switch */
div[data-testid="stToggle"] label {
    font-size: 15px;
    font-weight: 600;
    user-select: none;
}
</style>

""", unsafe_allow_html=True)

Functional st.toggle (uses emoji as label)

dark_mode = st.toggle("🌙")

---------------- THEME DEFINITIONS ----------------

if dark_mode: bg_gradient = "linear-gradient(to bottom right, #121212, #1e1e1e)" card_color  = "rgba(30, 30, 30, 0.85)" text_color  = "#ffffff" box_shadow  = "0 4px 12px rgba(255, 255, 255, 0.05)" success_bg  = "rgba(56, 142, 60, 0.25)" success_text = "#81c784" else: bg_gradient = "linear-gradient(to bottom right, #e3f2fd, #fce4ec)" card_color  = "rgba(255, 255, 255, 0.6)" text_color  = "#212121" box_shadow  = "0 10px 30px rgba(0, 0, 0, 0.1)" success_bg  = "rgba(232, 245, 233, 0.6)" success_text = "#2e7d32"

Apply theme CSS

st.markdown(f""" <style> html, body {{ background: {bg_gradient}; color: {text_color}; }} .stApp {{ padding: 1rem; }} h1, h2 {{ color: {text_color}; }} .stButton > button {{ background: rgba(255, 255, 255, 0.1); border-radius: 12px; color: {text_color}; font-weight: 600; padding: 0.6rem 1.5rem; border: 1px solid rgba(255, 255, 255, 0.25); backdrop-filter: blur(12px); box-shadow: {box_shadow}; transition: all 0.3s ease; }} .stSelectbox, .stCheckbox, .stSlider, .stNumberInput {{ background: rgba(255, 255, 255, 0.25) !important; border-radius: 12px; padding: 0.5rem; color: {text_color}; backdrop-filter: blur(10px); }} .block-container {{ max-width: 800px; margin: auto; padding: 2rem; border-radius: 25px; background: {card_color}; backdrop-filter: blur(15px); box-shadow: {box_shadow}; }} </style> """, unsafe_allow_html=True)

----------------- APP UI -----------------

st.title("🏥 Hospital Length of Stay Predictor") st.markdown("Use patient clinical data to predict whether their stay will be Short, Medium, or Long.")

with st.form("predict_form"): st.subheader("🧾 Patient Information") col1, col2 = st.columns(2) with col1: rcount = st.slider("Recent Admissions", 0, 10, 1) gender = st.selectbox("Gender", ["F", "M"]) diagnosis = st.selectbox("Secondary Diagnosis", ['None', 'DX1', 'DX2', 'DX3']) with col2: hemo = st.slider("Hemoglobin", 5.0, 20.0, 13.5) hematocrit = st.slider("Hematocrit", 20.0, 60.0, 40.0) neutrophils = st.slider("Neutrophils", 20.0, 90.0, 50.0)

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

if submitted: data = { 'rcount': rcount, 'gender': 0 if gender == 'F' else 1, 'dialysisrenalendstage': int(dialysis), 'asthma': int(asthma), 'irondef': int(irondef), 'pneum': int(pneum), 'substancedependence': int(substance), 'psychologicaldisordermajor': int(psychdisorder), 'depress': int(depress), 'psychother': int(psychother), 'fibrosisandother': int(fibrosis), 'malnutrition': 0, 'hemo': hemo, 'hematocrit': hematocrit, 'neutrophils': neutrophils, 'sodium': sodium, 'glucose': glucose, 'bloodureanitro': bun, 'creatinine': creatinine, 'bmi': bmi, 'pulse': pulse, 'respiration': respiration, } for dx in ['DX

