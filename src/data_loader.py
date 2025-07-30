import pandas as pd
from sklearn.datasets import load_iris
import os

def load_data():
    """Load the Iris dataset and save it as a raw CSV file."""
    # Create directory for raw data
    os.makedirs('data/raw', exist_ok=True)
    
    # Load Iris dataset
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['target'] = iris.target
    
    # Save raw data
    raw_data_path = 'data/raw/iris.csv'
    df.to_csv(raw_data_path, index=False)
    
    return df

if __name__ == "__main__":
    load_data()