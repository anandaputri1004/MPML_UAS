import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("best_model_light.pkl")

st.set_page_config(page_title="OnlineFoods Prediction App", page_icon="🍽", layout="centered")
st.title("🍽 OnlineFoods Purchase Prediction")
st.write("Masukkan data di bawah untuk memprediksi kemungkinan seseorang membeli makanan online.")

# Form input
with st.form("prediction_form"):
    age = st.number_input("Umur", min_value=18, max_value=100, value=25)
    family_size = st.number_input("Jumlah anggota keluarga", min_value=1, max_value=15, value=3)
    monthly_income = st.number_input("Pendapatan bulanan (angka)", min_value=0, value=25000)

    gender = st.selectbox("Jenis Kelamin", ["Male", "Female"])
    marital_status = st.selectbox("Status Pernikahan", ["Single", "Married", "Prefer not to say"])
    education = st.selectbox("Pendidikan", ["Graduate", "Post Graduate", "Ph.D", "School", "Uneducated"])
    feedback = st.selectbox("Feedback", ["Positive", "Negative"])

    submitted = st.form_submit_button("Prediksi")

if submitted:
    input_df = pd.DataFrame([{
        "Age": age,
        "Family size": family_size,
        "Monthly Income (num)": monthly_income,
        "Gender": gender,
        "Marital Status": marital_status,
        "Educational Qualifications": education,
        "Feedback": feedback
    }])

    pred = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0][1]  # probabilitas Yes

    if pred == 1:
        st.success(f"✅ Model memprediksi **Yes** (membeli makanan online) dengan probabilitas {proba:.2%}")
    else:
        st.error(f"❌ Model memprediksi **No** (tidak membeli makanan online) dengan probabilitas {proba:.2%}")
