"""Derive modeling features from merged EIA + ACS data."""

import pandas as pd
import numpy as np


FEATURE_COLS = [
    "electricity_per_customer",
    "electricity_per_capita",
    "renter_occupancy_rate",
    "housing_age",
    "income_log",
]


def engineer_features(df):
    """
    Compute 5 features for unsupervised learning:
      - electricity_per_customer: annual MWh per residential account
      - electricity_per_capita: annual MWh per person
      - renter_occupancy_rate: fraction of occupied units that are rented
      - housing_age: years since median structure was built (as of 2022)
      - income_log: log-transformed median household income
    """
    df = df.copy()

    df["electricity_per_customer"] = df["residential_mwh_sales"] / df["num_customers"]
    df["electricity_per_capita"] = df["residential_mwh_sales"] / df["population"]

    if "renter_occupied_units" in df.columns and "total_occupied_units" in df.columns:
        df["renter_occupancy_rate"] = df["renter_occupied_units"] / df["total_occupied_units"]
    else:
        df["renter_occupancy_rate"] = np.nan

    if "median_year_structure_built" in df.columns:
        df["housing_age"] = 2022 - df["median_year_structure_built"]
    else:
        df["housing_age"] = np.nan

    df["income_log"] = np.log(df["median_income"] + 1)

    before = len(df)
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURE_COLS)
    dropped = before - len(df)
    if dropped > 0:
        print(f"Dropped {dropped} rows with invalid feature values")

    print(f"Features ready for {len(df):,} ZIPs")
    print(df[FEATURE_COLS].describe())
    return df


def get_feature_matrix(df):
    """Return the 5 modeling features as a DataFrame."""
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}. Run engineer_features() first.")
    return df[FEATURE_COLS].copy()
