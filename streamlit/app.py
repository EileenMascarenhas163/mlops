import streamlit as st
import requests

st.set_page_config(page_title="Loan Approval Predictor", layout="centered")

st.title("🏦 Loan Approval Predictor")
st.write("Enter applicant details to predict loan approval status.")

# -----------------------------
# Input mappings (human → model)
# -----------------------------
gender = st.selectbox("Gender", ["Male", "Female"])
married = st.selectbox("Married", ["No", "Yes"])
dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed", ["No", "Yes"])
property_area = st.selectbox("Property Area", ["Rural", "Semiurban", "Urban"])

applicant_income = st.number_input("Applicant Income", min_value=0, value=5000)
coapplicant_income = st.number_input("Coapplicant Income", min_value=0, value=0)
loan_amount = st.number_input("Loan Amount", min_value=0, value=120)
loan_term = st.number_input("Loan Amount Term (months)", min_value=0, value=360)
credit_history = st.selectbox("Credit History", ["Bad (0)", "Good (1)"])

# -----------------------------
# Encode inputs (MATCH MODEL)
# -----------------------------
inputs = {
    "Gender": 1 if gender == "Male" else 0,
    "Married": 1 if married == "Yes" else 0,
    "Dependents": 3 if dependents == "3+" else int(dependents),
    "Education": 1 if education == "Graduate" else 0,
    "Self_Employed": 1 if self_employed == "Yes" else 0,
    "ApplicantIncome": applicant_income,
    "CoapplicantIncome": coapplicant_income,
    "LoanAmount": loan_amount,
    "Loan_Amount_Term": loan_term,
    "Credit_History": 1 if credit_history == "Good (1)" else 0,
    "Property_Area": 2 if property_area == "Urban" else 1 if property_area == "Semiurban" else 0
}

# -----------------------------
# Prediction
# -----------------------------
if st.button("🔮 Predict Loan Approval"):
    try:
        response = requests.post(
            "http://localhost:8000/predict",
            json=inputs,
            timeout=5
        )

        if response.status_code == 200:
            result = response.json()

            approved = result["loan_approved"]
            confidence = result["confidence"]

            if approved:
                st.success(f"✅ Loan Approved (Confidence: {confidence:.2f})")
            else:
                st.error(f"❌ Loan Rejected (Confidence: {confidence:.2f})")
        else:
            st.error("API error. Please check FastAPI logs.")

    except requests.exceptions.RequestException as e:
        st.error(f"Could not connect to API: {e}")
