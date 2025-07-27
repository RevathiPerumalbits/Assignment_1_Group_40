import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow
import mlflow.sklearn
import os


mlflow.set_experiment("Iris_Classification_Experiment")

def train_and_log_model(model, model_name, X_train, X_test, y_train, y_test):
    with mlflow.start_run(run_name=model_name):
        # Log parameters
        mlflow.log_param("model_name", model_name)
        if hasattr(model, 'get_params'):
            mlflow.log_params(model.get_params())

        # Train model
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Evaluate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        print(f"--- {model_name} Metrics ---")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")

        # Log model
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name="IrisClassifier" # This will create/update the registered model
        )
        print(f"{model_name} model logged to MLflow.")
    return accuracy

if __name__ == "__main__":
    # Load processed data
    df = pd.read_csv('data/processed/iris_processed.csv')
    X = df.drop('target', axis=1)
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Train Logistic Regression
    lr_model = LogisticRegression(max_iter=200, solver='liblinear')
    lr_accuracy = train_and_log_model(lr_model, "LogisticRegression", X_train, X_test, y_train, y_test)

    # Train Random Forest
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_accuracy = train_and_log_model(rf_model, "RandomForestClassifier", X_train, X_test, y_train, y_test)

    print("\nMLflow UI can be accessed by running: mlflow ui")