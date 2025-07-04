import streamlit as st
import pandas as pd
import joblib
from xgboost import XGBClassifier

# ───────────────────────────────────────────────────────────────
# App configuration
# ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Hospital LOS Predictor",
                   layout="centered",
                   page_icon="🏥")

# Load model & label-encoder
model = joblib.load("xgb_los_model.pkl")
label_encoder = joblib.load("los_label_encoder.pkl")

# ───────────────────────────────────────────────────────────────
# Styling – subtle left-to-right gradient flow
# ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ---------- animated gradient (left → right) ---------- */
@keyframes gradientShift {
  0%   {background-position: 0% 50%;}
  100% {background-position: 100% 50%;}
}

html, body, .stApp {
  height: 100%;
  margin: 0;
  background: linear-gradient(90deg,
              #b3e5fc,   /* pastel sky-blue  */
              #e1bee7,   /* soft lavender   */
              #bbdefb,   /* light blue-violet*/
              #d1c4e9);  /* pale purple     */
  background-size: 300% 300%;
  animation: gradientShift 25s linear infinite;
  font-family: 'Segoe UI', sans-serif;
}

/* ---------- glass-card container ---------- */
.block-container{
  max-width:860px;
  margin:2rem auto;
  padding:2.5rem;
  border-radius:20px;
  background:rgba(255,255,255,0.85);
  box-shadow:0 12px 32px rgba(0,0,0,0.10);
  backdrop-filter:blur(12px);
}

/* ---------- headings ---------- */
h1{font-size:2.2rem;text-align:center;color:#4527a0;margin-bottom:.5rem;}
.stSubheader{color:#512da8;}

/* ---------- buttons ---------- */
.stButton>button{
  background:#7e57c2;color:#fff;font-weight:600;
  padding:.6rem 1.8rem;border:none;border-radius:12px;
  box-shadow:0 4px 12px rgba(0,0,0,.1);transition:.3s;
}
.stButton>button:hover{
  background:#9575cd;transform:scale(1.05);
  box-shadow:0 6px 16px rgba(0,0,0,.15);
}

/* ---------- inputs ---------- */
.stSelectbox,.stCheckbox,.stSlider,.stNumberInput{
  background:rgba(255,255,255,.85)!important;
  border-radius:10px;padding:.4rem;color:#333;font-size:.95rem;
}

/* ---------- result card ---------- */
.los-result{
  padding:1.2rem;margin-top:1.8rem;border-radius:18px;
  background:rgba(232,234,246,.90);backdrop-filter:blur(10px);
  color:#4a148c;font-weight:bold;font-size:1.2rem;text-align:center;
  box-shadow:0 6px 20px rgba(0,0,0,.08);
}
</style>
""", unsafe_allow_html=True)

# ───────────────────────────────────────────────────────────────
# UI
# ───────────────────────────────────────────────────────────────
st.title("🏥 Hospital Length of Stay Predictor")
st.markdown("Use clinical inputs to predict if a patient’s stay will be **Short**, **Medium**, or **Long**.")

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
        dialysis  = st.checkbox("Dialysis End Stage")
        asthma    = st.checkbox("Asthma")
        irondef   = st.checkbox("Iron Deficiency")
    with col4:
        pneum     = st.checkbox("Pneumonia")
        substance = st.checkbox("Substance Dependence")
        psychdis  = st.checkbox("Psych Disorder")
    with col5:
        depress   = st.checkbox("Depression")
        psychother= st.checkbox("Psychotherapy")
        fibrosis  = st.checkbox("Fibrosis")

    st.subheader("📊 Vitals & Labs")
    col6, col7, col8 = st.columns(3)
    with col6:
        sodium  = st.slider("Sodium", 120.0, 160.0, 140.0)
        glucose = st.slider("Glucose", 50.0, 300.0, 100.0)
    with col7:
        bun        = st.slider("BUN", 5.0, 50.0, 15.0)
        creatinine = st.slider("Creatinine", 0.5, 5.0, 1.2)
    with col8:
        bmi         = st.slider("BMI", 10.0, 50.0, 22.0)
        pulse       = st.slider("Pulse", 40, 150, 70)
        respiration = st.slider("Respiration", 10, 40, 18)

    submitted = st.form_submit_button("Predict LOS")

# ───────────────────────────────────────────────────────────────
# Prediction
# ───────────────────────────────────────────────────────────────
if submitted:
    features = {
        "rcount": rcount, "gender": 0 if gender=="F" else 1,
        "dialysisrenalendstage": int(dialysis), "asthma": int(asthma),
        "irondef": int(irondef), "pneum": int(pneum),
        "substancedependence": int(substance), "psychologicaldisordermajor": int(psychdis),
        "depress": int(depress), "psychother": int(psychother),
        "fibrosisandother": int(fibrosis), "malnutrition": 0,
        "hemo": hemo, "hematocrit": hematocrit, "neutrophils": neutrophils,
        "sodium": sodium, "glucose": glucose, "bloodureanitro": bun,
        "creatinine": creatinine, "bmi": bmi, "pulse": pulse,
        "respiration": respiration
    }
    for dx in ["DX1", "DX2", "DX3"]:
        features[f"secondarydiagnosisnonicd9_{dx}"] = 1 if diagnosis == dx else 0

    df = pd.DataFrame([features])
    for col in model.get_booster().feature_names:
        if col not in df.columns:
            df[col] = 0
    df = df[model.get_booster().feature_names]

    pred   = model.predict(df)[0]
    result = label_encoder.inverse_transform([pred])[0]

    st.markdown(f"""
        <div class="los-result">
            ✅ Predicted Length of Stay: <strong>{result}</strong>
        </div>
    """, unsafe_allow_html=True)
