import pandas as pd
import os
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_PATH = os.path.join(BASE_DIR, 'data', "train.csv")
TEST_PATH = os.path.join(BASE_DIR, 'data', "test.csv")  

def train_baseline():
    """
    Trains a baseline Logistic Regression model on the training data and evaluates it on the test data.
    Logs the model and metrics using MLflow.
    """

    print("Loading training and testing data...")
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    # Separating features and target
    y_train = train_df["churn"]
    X_train = train_df.drop("churn", axis=1)

    y_test = test_df["churn"]
    X_test = test_df.drop("churn", axis=1)

    # Dropping id (not a feature)
    X_train = X_train.drop("customerid", axis=1)
    X_test = X_test.drop("customerid", axis=1)

    # Identifying categorical and numerical columns
    numeric_features = ["tenure", "monthlycharges", "totalcharges"]
    categorical_features = ["gender", "partner", "dependents", 
                            "phoneservice", "multiplelines", 
                            "internetservice", "onlinesecurity", 
                            "onlinebackup", "deviceprotection", 
                            "techsupport", "streamingtv", 
                            "streamingmovies", "contract", 
                            "paperlessbilling", "paymentmethod"]
    

    # StandardScaler for numeric features, 
    # OneHotEncoder for categorical features

    preprocessor = ColumnTransformer(
        transformers = [
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(drop='first'), categorical_features)      
        ]
    )

    # Creating the pipeline with preprocessing and Logistic Regression model
    # Step 1 : Preprocessing
    # Step 2 : Logistic Regression Model

    pipeline = Pipeline(steps = [
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(random_state = 42, max_iter=1000))
    ])

    # MLflow experiment tracking
    # Start an MLflow run to log parameters, metrics, and the model
    mlflow.set_experiment("Customer_Churn_Prediction_Baseline")

    with mlflow.start_run():
        print("Training the Baseline Logistic Regression Model...")

        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)

        # Calculating evaluation metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)     
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        print(f"Metrics:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")

        # Logging metrics to MLflow
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        # Logging the model to MLflow
        mlflow.sklearn.log_model(pipeline, "logistic_regression_model")
        print("Model and metrics logged to MLflow.")

if __name__ == "__main__":
    train_baseline()