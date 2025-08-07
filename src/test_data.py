import pandas as pd


def test_data_load():
    df = pd.read_csv("data/raw/iris.csv")
    assert len(df) > 0
