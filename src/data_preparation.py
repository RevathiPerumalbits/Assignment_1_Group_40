import pandas as pd
from sklearn.model_selection import train_test_split
import os

def preprocess_data(raw_data_path='data/raw/iris.csv'):
    """Preprocess the raw Iris dataset by splitting into train and test sets."""
    # Create directory for processed data
    os.makedirs('data/processed', exist_ok=True)
    
    # Load raw data
    df = pd.read_csv(raw_data_path)
    df = pd.read_csv('data/raw/iris.csv')
    X = df.drop('species', axis=1)
    y = df['species']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Save processed data
    X_train.to_csv('data/processed/X_train.csv', index=False)
    X_test.to_csv('data/processed/X_test.csv', index=False)
    y_train.to_csv('data/processed/y_train.csv', index=False)
    y_test.to_csv('data/processed/y_test.csv', index=False)
    
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    preprocess_data()