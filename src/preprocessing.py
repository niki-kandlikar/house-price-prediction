"""
preprocessing.py
-----------------
Missing value handling, feature engineering, and feature scaling
for the house price prediction pipeline.
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler

RAW_FEATURES = [
    "SquareFootage", "Bedrooms", "Bathrooms", "LotSize",
    "YearBuilt", "GarageSpaces", "DistanceToCity_km", "SchoolRating",
]
TARGET = "Price"


def load_data(path="data/housing_data.csv"):
    return pd.read_csv(path)


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Impute numeric missing values with the column median."""
    df = df.copy()
    for col in ["Bathrooms", "LotSize", "SchoolRating", "GarageSpaces"]:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())
    return df


def remove_outliers(df: pd.DataFrame, col="SquareFootage", z_thresh=4) -> pd.DataFrame:
    """Drop rows where a feature is an extreme statistical outlier."""
    z = (df[col] - df[col].mean()) / df[col].std()
    return df[z.abs() <= z_thresh].reset_index(drop=True)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create derived features that improve linear model fit."""
    df = df.copy()
    df["HouseAge"] = 2026 - df["YearBuilt"]
    df["LotToLivingRatio"] = df["LotSize"] / df["SquareFootage"]
    df["TotalRooms"] = df["Bedrooms"] + df["Bathrooms"]
    return df

ENGINEERED_FEATURES = RAW_FEATURES + ["HouseAge", "LotToLivingRatio", "TotalRooms"]
ENGINEERED_FEATURES.remove("YearBuilt")  # superseded by HouseAge


def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """Fit the scaler on the training split only, then apply it to both,
    so test-set statistics don't leak into training."""
    scaler = StandardScaler()
    train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)
    return train, test, scaler


if __name__ == "__main__":
    raw = load_data()
    df = handle_missing_values(raw)
    df = remove_outliers(df)
    df = engineer_features(df)
    print(f"Raw rows: {len(raw)} -> Clean rows: {len(df)}")
    print(df[ENGINEERED_FEATURES].describe())
