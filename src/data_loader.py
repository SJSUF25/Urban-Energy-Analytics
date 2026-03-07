"""Load EIA and ACS CSV files from data/raw/."""

import os
import pandas as pd


def _resolve_path(relative_path):
    # Resolve relative to repo root so this works from any working directory
    src_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(src_dir, ".."))
    return os.path.join(repo_root, relative_path)


def load_eia_data(filepath=None):
    """Load EIA Form 861 2022 ZIP-level residential electricity sales."""
    if filepath is None:
        filepath = _resolve_path("data/raw/eia861_sales_2022.csv")

    df = pd.read_csv(filepath, dtype={"ZIP": str})
    df["ZIP"] = df["ZIP"].str.zfill(5)

    print(f"Loaded {len(df):,} EIA rows from {os.path.basename(filepath)}")
    return df


def load_acs_data(filepath=None):
    """Load ACS 2022 5-year ZCTA-level socio-economic estimates."""
    if filepath is None:
        filepath = _resolve_path("data/raw/acs_zcta_2022.csv")

    df = pd.read_csv(filepath, dtype={"ZIP": str})
    df["ZIP"] = df["ZIP"].str.zfill(5)

    print(f"Loaded {len(df):,} ACS rows from {os.path.basename(filepath)}")
    return df


def load_all_data():
    """Load both EIA and ACS datasets. Returns (eia_df, acs_df)."""
    return load_eia_data(), load_acs_data()
