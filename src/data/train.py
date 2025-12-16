import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.impute import SimpleImputer

# ---------------------------
# Load params
# ---------------------------
params = yaml.safe_load(open("params.yaml"))

# ---------------------------
# MLflow setup
# ---------------------------
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("LoanApproval")

# ---------------------------
# Load processed data
# ---------------------------
DATA_PATH = Path("data/processed/train_data.csv")

if not DATA_PATH.exists():
    raise FileNotFoundError("Processed data missing. Run preprocessing first.")

df = pd.read_csv(DATA_PATH)

# ---------------------------
# Split features / target
# ---------------------------
X = df.drop("Loan_Status", axis=1)
y = df["Loan_Status"]

# ---------------------------
# Handle missing values (CRITICAL FIX)
# ---------------------------
imputer = SimpleImputer(strategy="median")
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# ---------------------------
# Train-test split
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=params["data"]["test_size"],
    random_state=params["data"]["random_state"]
)

# ---------------------------
# MLflow run
# ---------------------------
with mlflow.start_run():

    model = LogisticRegression(
        C=params["model"]["C"],
        max_iter=params["model"]["max_iter"],
        solver="liblinear"
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    # Log params & metrics
    mlflow.log_param("C", params["model"]["C"])
    mlflow.log_param("max_iter", params["model"]["max_iter"])
    mlflow.log_metric("accuracy", acc)

    # Log model
    mlflow.sklearn.log_model(
        model,
        artifact_path="model",
        registered_model_name="LoanApprovalModel"
    )

    print("✅ Training completed successfully")
    print(f"Accuracy: {acc:.4f}")
    print(classification_report(y_test, preds))
