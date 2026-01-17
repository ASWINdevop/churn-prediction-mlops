from fastapi import FastAPI, HTTPException
import joblib
import os
import pandas as pd
from app.schemas import CustomerData 

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'xgb_model.joblib')

app =FastAPI(title = "Customer Churn Prediction API")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Please train the model before starting the API.")

print("Loading model...")
pipeline = joblib.load(MODEL_PATH)
print("Model loaded successfully.")

CHURN_THRESHOLD = 0.56  # Default threshold for classification

@app.get("/")
def home():
    return {"message": "Welcome to the Customer Churn Prediction API. Go to /docs for Swagger UI documentation."}

@app.post("/predict")
def predict_churn(data: CustomerData):
    """
    Predicts whether a customer will churn based on input features.
    """

    # Convert input data(Pydantic object) to DataFrame
    input_data = data.dict()
    df = pd.DataFrame([input_data])
    # 2. Rename columns to match what the model expects
    df.columns = df.columns.str.lower().str.replace(" ","_")
    df.replace('No internet service', 'No', inplace=True)
    df.replace('No phone service', 'No', inplace=True)
    try:
        # 1. Get Probability instead of hard Prediction
        # predict_proba returns [[prob_stay, prob_churn]]
        probs = pipeline.predict_proba(df)
        churn_probability = float(probs[0][1])  # Probability of positive class (churn)

        # 2. Apply custom threshold
        prediction = 1 if churn_probability >= CHURN_THRESHOLD else 0

        return{
            "prediction": int(prediction),
            "churn_probability": float(churn_probability),
            "risk_level": "Critical" if churn_probability >= 0.7 else ("High" if prediction == 1 else "Low")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")