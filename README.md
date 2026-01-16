# End-to-End Customer Churn Prediction API

## Problem Statement

Customer churn directly impacts revenue and customer lifetime value.  
This system predicts the probability of customer churn in advance, enabling
business teams to take proactive retention actions such as targeted offers
or personalized engagement.


A production-grade MLOps pipeline to predict customer churn risk using **XGBoost**. This project features a complete lifecycle from data processing to deployment, including hyperparameter tuning, model explainability, and a **Dockerized REST API**.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.95-green)
![Docker](https://img.shields.io/badge/Docker-Enabled-blue)
![XGBoost](https://img.shields.io/badge/XGBoost-Optimized-orange)

##  Key Features

* **High-Performance Model:** XGBoost Classifier optimized with **Optuna** (F1 Score: 0.68).
* **Dockerized Deployment:** Fully containerized FastAPI application for consistent deployment.
* **Explainable AI:** Integrated **SHAP (SHapley Additive exPlanations)** to visualize why the model makes specific predictions (Waterfall & Beeswarm plots).
* **REST API:** Real-time inference endpoint with Pydantic data validation.
* **Imbalance Handling:** Tuned decision threshold (0.50) to balance Precision (62%) and Recall (75%).

## Tech Stack

* **Core:** Python 3.11, Pandas, Scikit-Learn
* **ML:** XGBoost, Joblib, Optuna (Hyperparameter Tuning)
* **Explainability:** SHAP
* **API:** FastAPI, Uvicorn, Pydantic
* **DevOps:** Docker, Git

##  Project Structure

```bash
churn-project/
├── app/
│   ├── main.py          # FastAPI application entry point
│   ├── schemas.py       # Pydantic data models for validation
├── data/                # Raw and processed datasets
├── models/              # Saved .joblib model binaries
├── notebooks/           # Jupyter notebooks for EDA
├── reports/             # SHAP plots and performance metrics
├── src/
│   ├── clean.py         # Data preprocessing pipeline
│   ├── train_xgb.py     # Model training script
│   ├── tune_xgb.py      # Optuna hyperparameter optimization
│   ├── explain.py       # SHAP visualization generator
│   ├── split.py         #Split Train and test dataset
│   ├── ingest.py        # Load data
│   ├── train_baseline   # Train baseline Logistic Regression
│   ├── find_threshold.py #Find optimal threshold value
│   └── run_pipeline     # For easy running
├── Dockerfile           # Docker image configuration
├── mlflow.db            #Database
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation
```

 How to Run
0. Download .csv file from 
https://www.kaggle.com/datasets/blastchar/telco-customer-churn 
and place in the ``` /data ``` folder

Option A: Using Docker (Recommended)
Build the Image:

```bash

docker build -t churn-prediction-app .
```

Run the Container:

```bash

docker run -p 80:80 churn-prediction-app
```

Access the API: Open your browser to: http://localhost/docs

Option B: Local Python Environment
Install Dependencies:


```bash
python run_pipeline.py
```
Run
```bash

Start the Server:


```bash
uvicorn app.main:app --reload
```

Model Performance
Precision: 62.2% (Minimizing false alarms)

Recall: 75.3% (Catching the majority of at-risk customers)

Threshold: 0.50 (Optimized for F1-Score)

 **Imbalance Handling:** Tuned decision threshold (0.50) to prioritize **Recall**
  over Precision, ensuring most at-risk customers are identified for retention actions.

##  Modeling Decisions

- XGBoost was chosen for its strong performance on tabular data and ability
  to model non-linear feature interactions.
- F1-score was used during optimization to balance Precision and Recall.
- Threshold tuning was applied instead of relying on the default 0.5 cutoff.
- SHAP was integrated to ensure model decisions remain interpretable
  for business stakeholders.


 API Usage Example
POST /predict

```JSON

{
  "gender": "Female",
  "SeniorCitizen": 0,
  "Partner": "Yes",
  "Dependents": "No",
  "tenure": 1,
  "PhoneService": "No",
  "MultipleLines": "No phone service",
  "InternetService": "DSL",
  "OnlineSecurity": "No",
  "OnlineBackup": "Yes",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "No",
  "StreamingMovies": "No",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 29.85,
  "TotalCharges": 29.85
}
```
Response:

```JSON

{
  "prediction": 1,
  "churn_probability": 0.67,
  "risk_level": "High"
}

```



