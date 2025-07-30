import pandas as pd
import pytest

def test_data_load():
    df = pd.read_csv('data/raw/iris.csv')
    assert len(df) > 0
    assert 'species' in df.columns