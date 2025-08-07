import os
import logging
import pandas as pd
from sklearn.datasets import load_iris

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_data():
    logger.info("Data Loading ....")
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df["species"] = iris.target.astype("float64")  # Convert to float64
    os.makedirs("data/raw", exist_ok=True)
    df.to_csv("data/raw/iris.csv", index=False)
    return df


if __name__ == "__main__":
    load_data()
    logger.info("Data Loaded Successfully")
