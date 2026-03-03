import pandas as pd


def add_features(data: pd.DataFrame) -> pd.DataFrame:
    df = data.copy()

    df["rooms_per_household"] = df["AveRooms"]
    df["bedrooms_ratio"] = df["AveBedrms"] / df["AveRooms"].replace(0, pd.NA)
    df["population_per_household"] = df["AveOccup"]

    return df
