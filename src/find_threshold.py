import pandas as pd
import joblib
import os
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# 1. Setup Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "cleaned.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "xgb_model.joblib")

def test_thresholds():
    print("⏳ Loading data and model...")
    df = pd.read_csv(DATA_PATH)
    
    # Load same test set split as training
    # (We re-split here to ensure we test on unseen data)
    from sklearn.model_selection import train_test_split
    X = df.drop(['churn', 'customerid'], axis=1)
    y = df['churn']
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = joblib.load(MODEL_PATH)
    
    # 2. Get Probabilities (The raw % score, not just Yes/No)
    # predict_proba gives [[prob_stay, prob_churn]]
    # We want column 1 (prob_churn)
    y_probs = model.predict_proba(X_test)[:, 1]
    
    print("\n📊 Threshold Analysis:")
    print(f"{'Threshold':<10} | {'Precision':<10} | {'Recall':<10} | {'F1 Score':<10}")
    print("-" * 46)
    
    # 3. Test thresholds from 0.30 to 0.80
    best_f1 = 0
    best_threshold = 0.5
    
    for thresh in np.arange(0.3, 0.85, 0.01):
        # Apply the new rule
        y_pred_custom = (y_probs >= thresh).astype(int)
        
        prec = precision_score(y_test, y_pred_custom)
        rec = recall_score(y_test, y_pred_custom)
        f1 = f1_score(y_test, y_pred_custom)
        
        print(f"{thresh:.2f}       | {prec:.4f}     | {rec:.4f}     | {f1:.4f}")
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = thresh

    print("-" * 46)
    print(f"🏆 Best Balanced Threshold (Max F1): {best_threshold:.2f}")

if __name__ == "__main__":
    test_thresholds()