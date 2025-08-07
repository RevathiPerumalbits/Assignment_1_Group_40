import hashlib
import logging
import os

from _04_model_train import main as train_main
from _05_model_evaluate import evaluate_and_register_models

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_file_hash(file_path):
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def check_for_new_data(
    data_path="data/raw/iris.csv", last_hash_file="data/last_hash.txt"
):
    if not os.path.exists(data_path):
        logger.info("Data file not found, triggering data load")
        os.system("python src/data_load.py")
        return True
    current_hash = get_file_hash(data_path)
    if os.path.exists(last_hash_file):
        with open(last_hash_file, "r") as f:
            last_hash = f.read().strip()
        if current_hash != last_hash:
            logger.info("New data detected, triggering re-training")
            with open(last_hash_file, "w") as f:
                f.write(current_hash)
            return True
    else:
        logger.info("No previous hash, triggering re-training")
        with open(last_hash_file, "w") as f:
            f.write(current_hash)
        return True
    return False


def retrain_if_needed():
    os.makedirs("data", exist_ok=True)
    if check_for_new_data():
        logger.info("Starting re-training pipeline")
        trained_models, X_train = train_main()
        evaluate_and_register_models(trained_models, X_train)
        logger.info("Re-training completed")
    else:
        logger.info("No new data, skipping re-training")


if __name__ == "__main__":
    os.environ["MLFLOW_TRACKING_URI"] = "sqlite:///mlruns.db"
    retrain_if_needed()
