import pandas as pd
from sklearn.preprocessing import LabelEncoder

def preprocess():
    df = pd.read_csv("data/raw/loan.csv")

    df.drop(columns=["Loan_ID"], inplace=True)
    df.fillna(method="ffill", inplace=True)

    for col in df.select_dtypes(include="object"):
        df[col] = LabelEncoder().fit_transform(df[col])

    df.to_csv("data/processed/loan.csv", index=False)

if __name__ == "__main__":
    preprocess()
