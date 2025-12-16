import pandas as pd
from sklearn.preprocessing import LabelEncoder
from pathlib import Path

def preprocess():
    data_path = Path("data/raw/train_data.csv")
    output_path = Path("data/processed/train_data.csv")

    if not data_path.exists():
        raise FileNotFoundError(f"{data_path} not found")

    df = pd.read_csv(data_path)

    # Drop ID column
    if "Loan_ID" in df.columns:
        df.drop(columns=["Loan_ID"], inplace=True)

    # Forward fill missing values
    df.ffill(inplace=True)

    # Encode categorical columns
    for col in df.select_dtypes(include="object").columns:
        df[col] = LabelEncoder().fit_transform(df[col])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print("✅ Preprocessing completed successfully")

if __name__ == "__main__":
    preprocess()
