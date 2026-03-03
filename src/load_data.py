from pathlib import Path

import pandas as pd

from config import DATA_PATH


def load_data() -> pd.DataFrame:
    data_path = Path(__file__).resolve().parents[1] / DATA_PATH
    return pd.read_csv(data_path)

if __name__ == "__main__":
    data = load_data()
    print(data.head())