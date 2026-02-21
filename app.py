import streamlit as st
import requests

# FastAPI backend URL
API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(page_title="Loan Eligibility Prediction", layout="centered")

st.title("🏦 Loan Eligibility Prediction")

st.write("Fill in the applicant details below to check loan eligibility:")

# Input fields
Gender = st.selectbox("Gender", ["Male", "Female"])
Married = st.selectbox("Married", ["Yes", "No"])
Dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
Education = st.selectbox("Education", ["Graduate", "Not Graduate"])
Self_Employed = st.selectbox("Self Employed", ["Yes", "No"])
ApplicantIncome = st.number_input("Applicant Income", min_value=0)
CoapplicantIncome = st.number_input("Coapplicant Income", min_value=0)
LoanAmount = st.number_input("Loan Amount (in thousands)", min_value=0)
Loan_Amount_Term = st.number_input("Loan Amount Term (in months)", min_value=0, value=360)
Credit_History = st.selectbox("Credit History", [1.0, 0.0])
Property_Area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])
CIBIL_Score = st.number_input("CIBIL Score", min_value=300, max_value=900, value=650)


# Prediction button
if st.button("Predict Loan Eligibility"):
    input_data = {
        "Gender": Gender,
        "Married": Married,
        "Dependents": Dependents,
        "Education": Education,
        "Self_Employed": Self_Employed,
        "ApplicantIncome": ApplicantIncome,
        "CoapplicantIncome": CoapplicantIncome,
        "LoanAmount": LoanAmount,
        "Loan_Amount_Term": Loan_Amount_Term,
        "Credit_History": Credit_History,
        "Property_Area": Property_Area,
        "CIBIL_Score": CIBIL_Score
    }

    try:
        response = requests.post(API_URL, json=input_data)
        if response.status_code == 200:
            result = response.json()
            st.success(f"Prediction: **{result['prediction']}**")
            st.write("📊 Probabilities:")
            st.json(result["probability"])
        if "max_eligible_loan_amount" in result:
            st.info(f"💡 You are eligible for loan up to: ₹ {result['max_eligible_loan_amount']} (in thousands)")

        else:
            st.error("❌ Error: Could not get prediction. Check FastAPI server.")
    except Exception as e:
        st.error(f"⚠️ Could not connect to API: {e}")
