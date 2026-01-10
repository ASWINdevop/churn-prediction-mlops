import pandas as pd
import os 
from sklearn.model_selection import train_test_split


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLEAN_DATA_PATH = os.path.join(BASE_DIR, 'data', "cleaned.csv")
TRAIN_PATH = os.path.join(BASE_DIR, 'data', "train.csv")
TEST_PATH = os.path.join(BASE_DIR, 'data', "test.csv")


def split_data():
    """
    Splits the cleaned dataset into training and testing sets.
    - Uses Stratified sampling to maintain Churn Ratio
    """

    if not os.path.exists(CLEAN_DATA_PATH):
        raise FileNotFoundError(f"Cleaned data file {CLEAN_DATA_PATH} does not exist.")   
    
    df = pd.read_csv(CLEAN_DATA_PATH)
    print(f"Cleaned data shape: {df.shape}")

    # stratify=df['churn'] ensures both sets have the same % of churners.
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["churn"])

    train_df.to_csv(TRAIN_PATH, index=False)
    test_df.to_csv(TEST_PATH, index=False)

    print(f"Training data saved to {TRAIN_PATH} with shape: {train_df.shape}")
    print(f"Testing data saved to {TEST_PATH} with shape: {test_df.shape}")

if __name__ == "__main__":
    split_data()