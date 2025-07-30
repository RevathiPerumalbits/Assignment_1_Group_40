# src/train.py
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import os
from mlflow import MlflowClient
from datetime import datetime
def train_model(model, model_name, X_train, y_train, X_test, y_test):
    with mlflow.start_run(run_name=model_name) as run:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        mlflow.log_param("model_name", model_name)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_score", f1)
        mlflow.sklearn.log_model(model, model_name)
        print(f"{model_name} - Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}")
        
        return model, accuracy,run.info.run_id

def main():
    # Load processed data
    X_train = pd.read_csv('data/processed/X_train.csv')
    X_test = pd.read_csv('data/processed/X_test.csv')
    y_train = pd.read_csv('data/processed/y_train.csv').values.ravel()
    y_test = pd.read_csv('data/processed/y_test.csv').values.ravel()
    
    # Set MLflow tracking
    mlflow.set_experiment("iris_classification")
    
    # Train models
    models = [
        (LogisticRegression(max_iter=200), "LogisticRegression"),
        (RandomForestClassifier(n_estimators=100, random_state=42), "RandomForest")
    ]
    
    best_accuracy = 0
    best_model = None
    best_model_name = ""
    best_run_id = None
    for model, name in models:
        trained_model, accuracy,run_id = train_model(model, name, X_train, y_train, X_test, y_test)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = trained_model
            best_model_name = name
            best_run_id = run_id
    
    # Register best model
    
    model_uri = f"runs:/{best_run_id}/{best_model_name}"
    registered_model=mlflow.register_model(model_uri, "IrisBestModel")
    client = MlflowClient()
    client.set_model_version_tag(
        name="IrisBestModel",
        version=registered_model.version,
        key="stage",
        value="Production"
    )
    # Define dynamic save path with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = f"saved_models/{best_model_name}_{timestamp}"
    os.makedirs('saved_models', exist_ok=True)
        
    # Save model with input example for signature
    input_example = X_train.iloc[:1]  # Use first row as input example
    mlflow.sklearn.save_model(model, save_path, input_example=input_example)
    mlflow.sklearn.log_model(model, best_model_name, input_example=input_example)
    # mlflow.sklearn.save_model(model, "saved_model")

if __name__ == "__main__":
    main()