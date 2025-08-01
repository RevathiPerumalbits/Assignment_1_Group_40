# Iris MLOps Pipeline
## Setup
1. Clone: `git clone https://github.com/<your-username>/iris_mlops`
2. Install: `pip install -r requirements.txt`
3. Initialize DVC: `dvc init && dvc add data/raw/iris.csv`
4. Run pipeline: `dvc repro`
5. Trigger re-training: `python src/retrain_trigger.py`
6. Run MLflow: `mlflow ui --port 5000`
7. Build Docker: `docker build -t iris-mlops-api .`
8. Run Docker: `docker run -p 8000:8000 iris-mlops-api`
9. Test API: `curl -X POST "http://localhost:8000/predict" -H "Content-Type: application/json" -d '{"features": [5.1, 3.5, 1.4, 0.2]}'`
10. Test metrics: `curl http://localhost:8000/metrics`
11. Test Prometheus: `curl http://localhost:8000/prometheus`
12. Run Prometheus: `docker run -d -p 9090:9090 -v prometheus.yml:/etc/prometheus/prometheus.yml prom/prometheus`
13. Run Grafana: `docker run -d -p 3000:3000 grafana/grafana`, import `grafana_dashboard.json`
## Directory Structure
- `.dvc/`, `dvc.yaml`, `data/raw/iris.csv.dvc`: DVC configuration
- `data/`: Raw and processed Iris data, `last_hash.txt`
- `src/`: Pipeline scripts including `retrain_trigger.py`
- `logs/`: Prediction logs (CSV and SQLite)
- `saved_models/`: Trained models
- `mlruns/`: MLflow artifacts
- `grafana_dashboard.json`: Grafana dashboard configuration
## CI/CD
- Workflow: `.github/workflows/iris_mlops_pipeline.yml`
- Lints, tests, trains, builds, and pushes Docker image to Docker Hub