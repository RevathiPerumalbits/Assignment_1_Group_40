import mlflow
import mlflow.sklearn
from mlflow import MlflowClient
from datetime import datetime
import os

def evaluate_and_register_models(trained_models, X_train):
    """Evaluate models, select the best, register it, and save locally."""
    best_accuracy = 0
    best_model = None
    best_model_name = ""
    best_run_id = None
    
    # Find the best model based on accuracy
    for model, name, accuracy, run_id in trained_models:
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model
            best_model_name = name
            best_run_id = run_id
    
    # Register best model in MLflow
    model_uri = f"runs:/{best_run_id}/{best_model_name}"
    registered_model = mlflow.register_model(model_uri, "IrisBestModel")
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
    mlflow.sklearn.save_model(best_model, save_path, input_example=input_example)
    mlflow.sklearn.log_model(best_model, best_model_name, input_example=input_example)
    
    print(f"Best model ({best_model_name}) registered with URI: {model_uri}, saved to {save_path}")

if __name__ == "__main__":
    # For standalone testing, call train_models first
    from model_train import main
    trained_models, X_train = main()
    evaluate_and_register_models(trained_models, X_train)