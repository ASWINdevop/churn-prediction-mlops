import pandas as pd
import os 

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_PATH = os.path.join(BASE_DIR, 'data', "Telco-Customer-Churn.csv")
CLEANED_DATA_PATH = os.path.join(BASE_DIR, 'data', "cleaned.csv")


def clean_data():
    """
    Cleans the Telco Customer Churn dataset.
    - Converts 'TotalCharges' to numeric, forcing errors to NaN.
    """

    if not os.path.exists(RAW_DATA_PATH):
        raise FileNotFoundError(f"Data file {RAW_DATA_PATH} does not exist.")   
    
    df = pd.read_csv(RAW_DATA_PATH)
    print(f"Raw data shape: {df.shape}")

    # Conversion of 'TotalCharges' to numeric, forcing errors to NaN

    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    
    missing_values = df['TotalCharges'].isnull().sum()
    print(f"Missing values in 'TotalCharges': {missing_values}")    
    
    # Fill Nan values with 0
    df['TotalCharges'] = df["TotalCharges"].fillna(0)

    # Normalizing Column Names
    df.columns = df.columns.str.lower().str.replace(' ', '_')

    # Mapping YEs to 1, No to 0
    df.churn = df.churn.map({'Yes': 1, 'No': 0 })
    df.replace('No internet service', 'No', inplace=True)    
    df.replace('No phone service', 'No', inplace=True)

    df.to_csv(CLEANED_DATA_PATH, index=False)
    print(f"Cleaned data saved to {CLEANED_DATA_PATH}")
    print(f"Cleaned data shape: {df.shape}")
    return df 
if __name__ == "__main__":
    clean_data()