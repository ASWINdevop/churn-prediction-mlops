import pandas as pd
import os


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', "Telco-Customer-Churn.csv")

def load_data():
    """
    Loads the Telco Customer Churn dataset from a CSV file.
    """
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Data directory {DATA_PATH} does not exist.")

    df = pd.read_csv(DATA_PATH)
    print("Data loaded successfully.")
    print(f"Data shape: {df.shape}")
    return df

if __name__ == "__main__":
    data = load_data()
    print(data.head())
