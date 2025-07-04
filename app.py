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
# Styling & Animations (no spinner)
# ───────────────────────────────────────────────────────────────
bg_gradient  = "linear-gradient(to bottom right, #e3f2fd, #fce4ec)"
card_color   = "rgba(255, 255, 255, 0.60)"
text_color   = "#212121"
box_shadow   = "0 10px 30px rgba(0, 0, 0, 0.10)"
success_bg   = "rgba(232, 245, 233, 0.60)"
success_text = "#2e7d32"

st.markdown(f"""
    <style>
    /* ── Keyframe animations ───────────────────── */
    @keyframes fadeInDown {{ 0% {{opacity:0;transform:translateY(-20px)}} 100% {{opacity:1;transform:translateY(0)}} }}
    @keyframes fadeInUp   {{ 0% {{opacity:0;transform:translateY(20px)}}  100% {{opacity:1;transform:translateY(0)}} }}
    @keyframes glowPulse  {{ 0% {{box-shadow:0 0 0px #81c784}} 50% {{box-shadow:0 0 20px #81c784}} 100% {{box-shadow:0 0 0px #81c784}} }}
    @keyframes slideInUp  {{ 0% {{opacity:0;transform:translateY(30px)}} 100% {{opacity:1;transform:translateY(0)}} }}

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
# Header
# ───────────────────────────────────────────────────────────────
st.title("🏥 Hospital Length of Stay Predictor")
st.markdown(
    "Use patient clinical data to predict whether their stay will be "
    "**Short**, **Medium**, or **Long**."
)

# ───────────────────────────────────────────────────────────────
# Prediction form
# ───────────────────────────────────────────────────────────────
with st.form("predict_form"):
    st.markdown("<h4>🧾 Patient Information</h4>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        rcount     = st.slider("Recent Admissions", 0, 10, 1)
        gender     = st.selectbox("Gender", ["F", "M"])
        diagnosis  = st.selectbox("Secondary Diagnosis", ["None", "DX1", "DX2", "DX3"])
    with col2:
        hemo        = st.slider("Hemoglobin", 5.0, 20.0, 13.5)
        hematocrit  = st.slider("Hematocrit", 20.0, 60.0, 40.0)
        neutrophils = st.slider("Neutrophils", 20.0, 90.0, 50.0)

    st.markdown("<h4>🩺 Clinical Conditions</h4>", unsafe_allow_html=True)
    col3,
