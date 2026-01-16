import pandas as pd
import numpy as np
import os
import mlflow
import mlflow.xgboost
from xgboost import XGBClassifier 
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_PATH = os.path.join(BASE_DIR, 'data', "train.csv")    
TEST_PATH = os.path.join(BASE_DIR, 'data', "test.csv")


def train_xgb():
    """
    Trains an XGBoost model on the training data and evaluates it on the test data.
    Logs the model and metrics using MLflow.
    """

    mlflow.set_experiment("Churn Prediction XGBoost Experiment ")

    print("Loading training and testing data...")
    train_df = pd.read_csv(TRAIN_PATH)      
    test_df = pd.read_csv(TEST_PATH)

    # Separating features and target
    y_train = train_df["churn"]
    X_train = train_df.drop(["churn", "customerid"], axis=1)

    y_test = test_df["churn"]
    X_test = test_df.drop(["churn", "customerid"], axis=1)  

    # Claculate Class imbalance Weights
    # Formula: count(negative)/count(positive)
    # This tells XGBoost: "Hey, positive class is rare, pay more attention to it"
    neg, pos = np.bincount(y_train)
    scale_pos_weight = neg / pos
    print(f"Scale Positive Weight: {scale_pos_weight}") 

    # Preprocessing:
    #  OneHotEncoding for categorical features
    categorical_features = X_train.select_dtypes(include = ['object']).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers = [
            ('cat', OneHotEncoder(handle_unknown = 'ignore'), categorical_features)
        ], 
        remainder='passthrough' # Keep other columns unchanged
    )

    # Define Pipeline with Preprocessing and XGBoost Model
    model = XGBClassifier(
        n_estimators = 659,
        max_depth = 9,
        learning_rate = 0.18275073207500148,
        subsample = 0.997589664489237,
        colsample_bytree = 0.7303591546811505,
        gamma = 4.673954625794586,
        reg_alpha = 2.2891756878952307,
        reg_lambda = 4.647669681644919,
        scale_pos_weight = 2.2870414252761915,
        random_state=42,
        n_jobs=-1
    )

    pipeline = Pipeline(steps =[
        ('preprocessor', preprocessor),
        ('classifier', model)       
    ])

    # Train & Log with MLflow

    with mlflow.start_run():
        print("Training XGBoost Model...")
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)

        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)     
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        print(f"Metrics")
        print(f"Accuracy: {accuracy}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1-Score: {f1}")

        # Log parameters and metrics to MLflow
        mlflow.log_param("model_type", "XGBoost")
        mlflow.log_param("scale_pos_weight", scale_pos_weight)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        # Log the model to MLflow
        mlflow.sklearn.log_model(pipeline, "xgboost_model")
        print("Model and metrics logged to MLflow")

        MODEL_PATH = os.path.join(BASE_DIR, 'models', 'xgb_model.joblib')
        joblib.dump(pipeline, MODEL_PATH)
        print(f"Trained model saved to {MODEL_PATH}")

if __name__ == "__main__":
    train_xgb()
