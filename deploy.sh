#!/bin/bash
docker pull 2023ad05044/iris-mlops-api:latest
docker run -d -p 8000:8000 2023ad05044/iris-mlops-api:latest
echo "Iris MLOps API is running on port 8000"
echo "You can access the API at http://localhost:8000"
echo "To stop the container, use 'docker stop <container_id>'"
echo "To remove the container, use 'docker rm <container_id>'"
echo "To view logs, use 'docker logs <container_id>'"
echo "To access the MLflow UI, run 'mlflow ui' in a separate terminal"
echo "MLflow UI will be available at http://localhost:5000"
echo "Ensure you have the necessary environment variables set for MLflow tracking URI and artifact root"
echo "For example, you can set them as follows:"
echo "export MLFLOW_TRACKING_URI=sqlite:///mlruns.db"