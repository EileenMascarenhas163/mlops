from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_prediction():
    payload = {
        "Gender": 1,
        "Married": 1,
        "Dependents": 0,
        "Education": 1,
        "Self_Employed": 0,
        "ApplicantIncome": 5000,
        "CoapplicantIncome": 0,
        "LoanAmount": 120,
        "Loan_Amount_Term": 360,
        "Credit_History": 1,
        "Property_Area": 2
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
