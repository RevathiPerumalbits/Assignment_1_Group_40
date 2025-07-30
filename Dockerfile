FROM python:3.9-slim

# Set working directory
WORKDIR /app

COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt
COPY src/ ./src/
COPY saved_models/ ./saved_models/
COPY mlruns.db .
RUN mkdir -p /app/logs && chmod -R 777 /app/logs

#Set PYTHONPATH to ensure src module is importable

ENV PYTHONPATH=/app

# Set MLflow tracking URI
ENV MLFLOW_TRACKING_URI=sqlite:///mlruns.db

# Debug: List files to verify copying
RUN ls -R /app


EXPOSE 8000

CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]