import streamlit as st
import requests

st.title("Loan Approval Predictor")

inputs = {
    "Gender": st.selectbox("Gender", [0, 1]),
    "Married": st.selectbox("Married", [0, 1]),
    "Dependents": st.selectbox("Dependents", [0, 1, 2]),
    "Education": st.selectbox("Education", [0, 1]),
    "Self_Employed": st.selectbox("Self Employed", [0, 1]),
    "ApplicantIncome": st.number_input("Applicant Income", 1000),
    "CoapplicantIncome": st.number_input("Coapplicant Income", 0),
    "LoanAmount": st.number_input("Loan Amount", 50),
    "Loan_Amount_Term": st.number_input("Loan Term", 360),
    "Credit_History": st.selectbox("Credit History", [0, 1]),
    "Property_Area": st.selectbox("Property Area", [0, 1, 2])
}

if st.button("Predict"):
    res = requests.post(
        "http://localhost:8000/predict",
        json=inputs
    ).json()

    st.success(res)
