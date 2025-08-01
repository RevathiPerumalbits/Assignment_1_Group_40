# src/api.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import mlflow.sklearn
from datetime import datetime
import pandas as pd
from typing import List
import os
import glob
from prometheus_client import Counter, Histogram,make_asgi_app
import sqlite3
app = FastAPI()

# Load the registered model
#model = mlflow.sklearn.load_model("models:/IrisBestModel/1")
#model = mlflow.sklearn.load_model("saved_model")
# Prometheus metrics
REQUEST_COUNT = Counter('iris_api_requests_total', 'Total API requests', ['endpoint'])
REQUEST_LATENCY = Histogram('iris_api_request_latency_seconds', 'Request latency', ['endpoint'])

# Load the latest version of the registered model
def load_latest_model():
    try:
        client = mlflow.tracking.MlflowClient()
        latest_versions = client.get_latest_versions("IrisBestModel", stages=["Production"])
        if not latest_versions:
            raise ValueError("No versions found for model 'IrisBestModel'")
        latest_version = latest_versions[0]
        model_uri = f"models:/IrisBestModel/{latest_version.version}"
        return mlflow.sklearn.load_model(model_uri)
    except Exception as e:
        print(f"Failed to load from MLflow registry: {e}")
        # Fallback to latest model in saved_models/
        model_dir = "/app/saved_models"
        if not os.path.exists(model_dir):
            raise ValueError(f"No models found in {model_dir}")
        model_paths = glob.glob(f"{model_dir}/*/model.pkl")
        if not model_paths:
            raise ValueError(f"No model files found in {model_dir}")
        latest_model_path = max(model_paths, key=os.path.getmtime)
        return mlflow.sklearn.load_model(os.path.dirname(latest_model_path))
    

model = load_latest_model()
class IrisInput(BaseModel):
    features: List[float] = Field(..., min_items=4, max_items=4, description="Four Iris features: sepal length, sepal width, petal length, petal width")

    # Custom validation for positive floats
    @classmethod
    def __get_validators__(cls):
        yield cls.validate_features

    @classmethod
    def validate_features(cls, value):
        for v in value.features:
            if not isinstance(v, (int, float)) or v <= 0:
                raise ValueError("All features must be positive numbers")
        return value

def init_db():
    conn = sqlite3.connect('logs/predictions.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS predictions
                 (timestamp TEXT, features TEXT, prediction INTEGER)''')
    conn.commit()
    conn.close()

init_db()
@app.post("/predict")
async def predict(data: IrisInput):
    try:
        # Convert input to DataFrame
        input_df = pd.DataFrame([data.features], columns=[
            'sepal length (cm)', 'sepal width (cm)', 
            'petal length (cm)', 'petal width (cm)'
        ])
        prediction = model.predict(input_df)[0]
        conn = sqlite3.connect('logs/predictions.db')
        c= conn.cursor()
        c.execute("INSERT INTO predictions (timestamp, features, prediction) VALUES (?, ?, ?)",
                      (datetime.now().isoformat(), str(data.features), prediction))
        conn.commit()
        conn.close()
        # Log prediction
        log_prediction(data.features, prediction)
        return {"prediction": int(prediction)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/")
async def root():
    return {"message": "Iris Classification API"}
@app.get("/metrics")
async def get_metrics():
    with REQUEST_LATENCY.labels(endpoint='/metrics').time():
        REQUEST_COUNT.labels(endpoint='/metrics').inc()
        conn = sqlite3.connect('logs/predictions.db')
        logs = pd.read_sql_query("SELECT * FROM predictions", conn)
        conn.close()
        return {
            "total_predictions": len(logs),
            "avg_prediction": logs['prediction'].mean() if not logs.empty else 0.0
        }
def log_prediction(features, prediction):
    import datetime
    import os
    log_entry = {
        'timestamp': datetime.datetime.now().isoformat(),
        'features': features,
        'prediction': prediction
    }
    log_df = pd.DataFrame([log_entry])
    
    # Append to CSV log
    log_file = 'logs/predictions.csv'
    os.makedirs('logs', exist_ok=True)
    if os.path.exists(log_file):
        log_df.to_csv(log_file, mode='a', header=False, index=False)
    else:
        log_df.to_csv(log_file, index=False)
# Mount Prometheus metrics endpoint
app.mount("/prometheus", make_asgi_app())