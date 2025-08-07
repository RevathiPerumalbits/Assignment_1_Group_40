FROM python:3.9-slim

# Set working directory
WORKDIR /app

COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt
# Install flake8 for linting
RUN pip install --no-cache-dir flake8
# Copy the .flake8 configuration file
COPY .flake8 .
COPY src/ ./src/
COPY saved_models/ ./saved_models/
# Run flake8 linting during build
RUN flake8 src/ --config=.flake8 --output-file=flake8-report.txt
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