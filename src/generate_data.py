"""
generate_data.py
-----------------
Generates a realistic synthetic housing dataset with structured
relationships between features and price (plus intentional missing
values and mild outliers), so the pipeline has real cleaning and
feature engineering work to do.
"""

import numpy as np
import pandas as pd

RNG = np.random.default_rng(7)
N_HOUSES = 800


def generate_houses(n=N_HOUSES):
    sqft = RNG.normal(1800, 650, n).clip(500, 5000)
    bedrooms = RNG.integers(1, 6, n)
    bathrooms = (bedrooms * 0.75 + RNG.normal(0, 0.6, n)).clip(1, 5).round(1)
    lot_size = (sqft * RNG.normal(1.8, 0.5, n)).clip(1000, 20000)
    year_built = RNG.integers(1950, 2024, n)
    garage_spaces = RNG.integers(0, 4, n)
    distance_to_city_km = RNG.exponential(8, n).clip(0.5, 60)
    school_rating = RNG.integers(1, 11, n)

    age = 2026 - year_built

    # Price driven by a realistic combination of features + noise
    price = (
        50000
        + sqft * 145
        + bedrooms * 8000
        + bathrooms * 12000
        + lot_size * 3.2
        - age * 550
        + garage_spaces * 6000
        - distance_to_city_km * 900
        + school_rating * 4200
        + RNG.normal(0, 25000, n)
    ).clip(60000, None)

    df = pd.DataFrame({
        "SquareFootage": sqft.round(0),
        "Bedrooms": bedrooms,
        "Bathrooms": bathrooms,
        "LotSize": lot_size.round(0),
        "YearBuilt": year_built,
        "GarageSpaces": garage_spaces,
        "DistanceToCity_km": distance_to_city_km.round(1),
        "SchoolRating": school_rating,
        "Price": price.round(0),
    })

    # Inject realistic missingness
    for col, frac in [("Bathrooms", 0.03), ("LotSize", 0.04),
                       ("SchoolRating", 0.05), ("GarageSpaces", 0.02)]:
        idx = RNG.choice(df.index, size=int(len(df) * frac), replace=False)
        df.loc[idx, col] = np.nan

    # A few mild outliers (data entry errors)
    outlier_idx = RNG.choice(df.index, size=5, replace=False)
    df.loc[outlier_idx, "SquareFootage"] = df.loc[outlier_idx, "SquareFootage"] * 4

    df.insert(0, "HouseID", [f"H{2000+i}" for i in range(len(df))])
    return df


if __name__ == "__main__":
    df = generate_houses()
    df.to_csv("data/housing_data.csv", index=False)
    print(f"Generated {len(df)} house records -> data/housing_data.csv")
    print(df.head())
    print(f"\nMissing values:\n{df.isnull().sum()}")
