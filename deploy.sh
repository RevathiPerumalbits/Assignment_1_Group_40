
#!/bin/bash

# Configuration
DOCKER_IMAGE="2023ad05044/iris-mlops-api:latest"
CONTAINER_NAME="iris-mlops-api"
PORT="8000"
DOCKERHUB_USERNAME="$1"
DOCKERHUB_TOKEN="$2"

# Validate inputs
if [ -z "$DOCKERHUB_USERNAME" ] || [ -z "$DOCKERHUB_TOKEN" ]; then
  echo "Error: DOCKERHUB_USERNAME and DOCKERHUB_TOKEN must be provided as arguments"
  echo "Usage: $0 <dockerhub_username> <dockerhub_token>"
  exit 1
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

# Test the API endpoints
echo "Testing /predict endpoint..."
docker exec iris-mlops-api curl -v -X POST "http://localhost:8000/predict" -H "Content-Type: application/json" -d '{"features": [5.1, 3.5, 1.4, 0.2]}'|| echo "Predict endpoint test failed"

echo "Testing /metrics endpoint..."
docker exec iris-mlops-api curl -v "http://localhost:8000/metrics" || echo "Metrics endpoint test failed"

echo "Testing /prometheus endpoint..."
docker exec iris-mlops-api curl -v "http://localhost:8000/prometheus" || echo "Prometheus endpoint test failed"

echo "Deployment completed. API is running on http://localhost:8000"


