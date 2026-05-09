import pandas as pd
from pathlib import Path

DATA_PATH = Path(__file__).parents[2] / "data" / "raw" / "ENERGY_Hackathon_DataSet.csv"


def load_raw(path=DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";", decimal=",", parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def load_train(path=DATA_PATH) -> pd.DataFrame:
    df = load_raw(path)
    return df[df["timestamp"].dt.year == 2024].copy().reset_index(drop=True)


def load_test(path=DATA_PATH) -> pd.DataFrame:
    df = load_raw(path)
    return df[df["timestamp"].dt.year == 2025].copy().reset_index(drop=True)


def load_split(path=DATA_PATH):
    df = load_raw(path)
    train = df[df["timestamp"].dt.year == 2024].copy().reset_index(drop=True)
    test  = df[df["timestamp"].dt.year == 2025].copy().reset_index(drop=True)
    return train, test
