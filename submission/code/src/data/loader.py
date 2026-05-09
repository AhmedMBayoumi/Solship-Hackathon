import pandas as pd
from pathlib import Path

RAW_PATH       = Path(__file__).parents[2] / "data" / "raw"       / "ENERGY_Hackathon_DataSet.csv"
PROCESSED_PATH = Path(__file__).parents[2] / "data" / "processed" / "dataset_processed.csv"


def load_processed(path=PROCESSED_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp"])
    return df.sort_values("timestamp").reset_index(drop=True)


def load_train(path=PROCESSED_PATH) -> pd.DataFrame:
    df = load_processed(path)
    return df[df["timestamp"].dt.year == 2024].copy().reset_index(drop=True)


def load_test(path=PROCESSED_PATH) -> pd.DataFrame:
    df = load_processed(path)
    return df[df["timestamp"].dt.year == 2025].copy().reset_index(drop=True)


def load_split(path=PROCESSED_PATH):
    df = load_processed(path)
    train = df[df["timestamp"].dt.year == 2024].copy().reset_index(drop=True)
    test  = df[df["timestamp"].dt.year == 2025].copy().reset_index(drop=True)
    return train, test
