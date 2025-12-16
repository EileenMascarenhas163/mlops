import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import yaml

params = yaml.safe_load(open("params.yaml"))

mlflow.set_experiment("LoanApproval")

df = pd.read_csv("data/processed/loan.csv")
X = df.drop("Loan_Status", axis=1)
y = df["Loan_Status"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

with mlflow.start_run():
    model = LogisticRegression(C=params["model"]["C"], max_iter=1000)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    mlflow.log_param("C", params["model"]["C"])
    mlflow.log_metric("accuracy", acc)
    mlflow.sklearn.log_model(
        model, "model",
        registered_model_name="LoanApprovalModel"
    )
