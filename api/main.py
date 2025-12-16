import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI
from prometheus_client import Counter, Histogram, generate_latest

app = FastAPI()

model = mlflow.sklearn.load_model(
    "models:/LoanApprovalModel@prod"
)

REQUESTS = Counter("requests_total", "Total requests")
LATENCY = Histogram("latency_seconds", "Prediction latency")

@app.post("/predict")
def predict(features: dict):
    REQUESTS.inc()
    with LATENCY.time():
        df = pd.DataFrame([features])
        prob = model.predict_proba(df)[0][1]

        return {
            "loan_approved": bool(prob >= 0.5),
            "confidence": float(round(prob, 3))
        }

@app.get("/metrics")
def metrics():
    return generate_latest()
