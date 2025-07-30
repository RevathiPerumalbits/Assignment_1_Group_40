import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

def train_model(model, model_name, X_train, y_train, X_test, y_test):
    """Train a single model and log to MLflow."""
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
        
        return model, accuracy, run.info.run_id

def main():
    """Train multiple models and return their details."""
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
    
    trained_models = []
    for model, name in models:
        trained_model, accuracy, run_id = train_model(model, name, X_train, y_train, X_test, y_test)
        trained_models.append((trained_model, name, accuracy, run_id))
    
    return trained_models, X_train

if __name__ == "__main__":
    trained_models, _ = main()