import pandas as pd
import xgboost as xgb
import optuna 
import mlflow
import numpy as np
import os
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_PATH = os.path.join(BASE_DIR, "data", "train.csv")
TEST_PATH = os.path.join(BASE_DIR, "data", "test.csv")

def load_data():
    if not os.path.exists(TRAIN_PATH):
        raise FileNotFoundError(f"The file at {TRAIN_PATH} was not found.")
    
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)    

    y_train = train_df['churn']
    X_train = train_df.drop(columns=['churn', 'customerid'], axis=1)
    y_test = test_df['churn']
    X_test = test_df.drop(columns=['churn', 'customerid'], axis=1)
    return X_train, y_train, X_test, y_test

def objective(trial):
    """
    Docstring for objective
    
    :param trial: Description
    """

    X_train, y_train, X_test, y_test = load_data()

    param = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'gamma' : trial.suggest_float('gamma', 0, 5),
        'reg_alpha' : trial.suggest_float('reg_alpha', 0, 10),
        'reg_lambda' : trial.suggest_float('reg_lambda', 0, 10),
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1.0, 10.0),
        'random_state': 42,
        'n_jobs': -1,
        'eval_metric': 'logloss',
    }

    categorical_features = X_train.select_dtypes(include=['object']).columns.tolist()
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ],
        remainder='passthrough'
    )

    model = xgb.XGBClassifier(**param)
    pipeline = Pipeline(steps =[
        ('preprocessor', preprocessor),
        ('model', model)        
    ]
    )

    try:
        pipeline.fit(X_train, y_train)
    except Exception as e:
        print(f"Trail failed: {e}")
        return 0.0
    
    preds = pipeline.predict(X_test)
    score = f1_score(y_test, preds) 

    return score

def run_tuning():
    print("Starting hyperparameter tuning with Optuna...")
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=20)
    print("Best trial:")
    print(f" Best F1 Score: {study.best_value:.4f}")
    print(" Best hyperparameters: ")
    for key, value in study.best_params.items():
        print(f"    '{key}': {value},")

    mlflow.set_experiment("Churn Prediction XGBoost Tuning ")
    with mlflow.start_run():
        mlflow.log_params(study.best_params)
        mlflow.log_metric("best_f1_score", study.best_value)
        print("Hyperparameter tuning completed and results logged to MLflow.")

if __name__ == "__main__":
    run_tuning()