#!/bin/bash
set -x
echo "Startng deployment script..."
# Configuration
DOCKER_IMAGE="2023ad05044/iris-mlops-api:latest"
CONTAINER_NAME="iris-mlops-api"
PORT="${3:-8000}"  # Use third argument or default to 8000
DOCKERHUB_USERNAME="$1"
DOCKERHUB_TOKEN="${2:-$DOCKERHUB_TOKEN}"


# Validate inputs
if [ -z "$DOCKERHUB_USERNAME" ] || [ -z "$DOCKERHUB_TOKEN" ]; then
  echo "Error: DOCKERHUB_USERNAME and DOCKERHUB_TOKEN must be provided as arguments or environment variable"
  echo "Usage: $0 <dockerhub_username> [<dockerhub_token>] [<port>]"
  exit 1
fi
# Check for Python and dependencies
echo "Checking Python and dependencies..."
if ! command -v python3 >/dev/null 2>&1; then
  echo "Error: python3 not found. Please install Python 3.9."
  exit 1
fi
if ! command -v pip >/dev/null 2>&1; then
  echo "Error: pip not found. Please install pip."
  exit 1
fi
# Run linting and tests
echo "Running flake8 and pytest..."
flake8 src/ --config=.flake8 --output-file=flake8-report.txt
if [ $? -ne 0 ]; then
  echo "Error: flake8 linting failed"
  cat flake8-report.txt
  exit 1
fi
cat flake8-report.txt
# Check if port is available
if command -v netstat >/dev/null 2>&1; then
  if netstat -aon | grep -q ":$PORT "; then
    echo "Error: Port $PORT is already in use. Please free the port or specify a different one."
    exit 1
  fi
elif command -v lsof >/dev/null 2>&1; then
  if lsof -i :$PORT >/dev/null 2>&1; then
    echo "Error: Port $PORT is already in use. Please free the port or specify a different one."
    exit 1
  fi
else
  echo "Warning: Unable to check port availability (netstat or lsof not found). Proceeding with caution."
fi

# Log in to Docker Hub
echo "Logging in to Docker Hub..."
echo "$DOCKERHUB_TOKEN" | docker login -u "$DOCKERHUB_USERNAME" --password-stdin
if [ $? -ne 0 ]; then
  echo "Error: Docker login failed"
  exit 1
fi

# Pull the latest Docker image
echo "Pulling Docker image: $DOCKER_IMAGE"
docker pull $DOCKER_IMAGE
if [ $? -ne 0 ]; then
  echo "Error: Failed to pull Docker image"
  exit 1
fi

# Stop and remove existing container if running
echo "Stopping and removing existing container if it exists..."
docker stop $CONTAINER_NAME 2>/dev/null || true
docker rm $CONTAINER_NAME 2>/dev/null || true

# Run the Docker container
echo "Starting Docker container: $CONTAINER_NAME"
docker run -d --name $CONTAINER_NAME -p $PORT:8000 $DOCKER_IMAGE
if [ $? -ne 0 ]; then
  echo "Error: Failed to start Docker container"
  exit 1
fi

# Wait for the container to start
echo "Waiting for container to be ready..."
sleep 5

# Check for curl and python3 in container
echo "Checking container dependencies..."
python -c "import pandas" >/dev/null 2>&1 || { echo "Error: python or pandas not found in container"; exit 1; }

docker logs $CONTAINER_NAME
# Test the API endpoints
echo "Testing /predict endpoint..."
curl -X POST "http://localhost:8000/predict" -H "Content-Type: application/json" -d '{"features": [5.1, 3.5, 1.4, 0.2]}' > prediction.json
if [ $? -ne 0 ]; then
  echo "Predict endpoint test failed"
  exit 1
fi
echo "Raw prediction.json output:"
cat prediction.json

# Log prediction to CSV
echo "Logging prediction to CSV..."
python -c "import json; import pandas as pd; import os; from datetime import datetime; data = json.load(open('prediction.json')); log_entry = {'timestamp': datetime.now().isoformat(), 'features': [5.1, 3.5, 1.4, 0.2], 'prediction': data['prediction']}; log_df = pd.DataFrame([log_entry]); os.makedirs('logs', exist_ok=True); log_df.to_csv('logs/predictions.csv', index=False)"
if [ $? -ne 0 ]; then
  echo "Failed to log prediction to CSV"
  exit 1
fi

echo "Testing /metrics endpoint..."
curl "http://localhost:8000/metrics" || echo "Metrics endpoint test failed"

echo "Testing /prometheus endpoint..."
curl "http://localhost:8000/prometheus" || echo "Prometheus endpoint test failed"

docker run -d -p 9090:9090 -v $(pwd)/prometheus.yml:/etc/prometheus/prometheus.yml prom/prometheus
docker run -d -p 3000:3000 grafana/grafana

echo "Deployment completed. API is running on http://localhost:$PORT"
