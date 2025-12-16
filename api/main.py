from fastapi import FastAPI
import mlflow.sklearn
import pandas as pd
from prometheus_client import Counter, Histogram, generate_latest

app = FastAPI()

model = mlflow.sklearn.load_model(
    "models:/LoanApprovalModel/Production"
)

REQUESTS = Counter("requests_total", "Total requests")
LATENCY = Histogram("latency_seconds", "Prediction latency")

@app.post("/predict")
def predict(features: dict):
    REQUESTS.inc()
    with LATENCY.time():
        df = pd.DataFrame([features])
        prob = model.predict_proba(df)[0][1]
        prediction = prob >= 0.5
        return {
            "loan_approved": bool(prediction),
            "confidence": round(prob, 3)
        }

@app.get("/metrics")
def metrics():
    return generate_latest()
