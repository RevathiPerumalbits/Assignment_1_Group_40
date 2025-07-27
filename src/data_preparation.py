import pandas as pd
from sklearn.datasets import load_iris
import os

def load_and_preprocess_iris():
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['target'] = iris.target
    # Simple preprocessing: rename columns to be more Python-friendly
    df.columns = [col.replace(' (cm)', '').replace(' ', '_') for col in df.columns]
    return df

if __name__ == "__main__":
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    print("Directories for raw and processed data created. ")
    # Simulate raw data saving (Iris is usually loaded directly)
    # For larger datasets, you'd download to data/raw
    iris_df = load_and_preprocess_iris()
    iris_df.to_csv('data/raw/iris.csv', index=False)
    print("Raw Iris data saved to data/raw/iris.csv")

    # Preprocess and save processed data
    # For Iris, preprocessing might be minimal, but in real scenarios,
    # this would involve scaling, encoding, etc.
    processed_df = iris_df.copy() # No complex preprocessing for Iris example
    processed_df.to_csv('data/processed/iris_processed.csv', index=False)
    print("Processed Iris data saved to data/processed/iris_processed.csv")