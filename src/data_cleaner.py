"""Clean and integrate EIA and ACS datasets."""

import pandas as pd
import numpy as np


def clean_eia_data(df):
    """Zero-pad ZIPs, drop invalid rows, and aggregate by ZIP (multiple utilities may serve one ZIP)."""
    df = df.copy()
    df["ZIP"] = df["ZIP"].astype(str).str.zfill(5)

    df = df.dropna(subset=["ZIP", "residential_mwh_sales", "num_customers"])
    df = df[df["residential_mwh_sales"] > 0]
    df = df[df["num_customers"] > 0]

    df_agg = (
        df.groupby("ZIP")
        .agg(
            residential_mwh_sales=("residential_mwh_sales", "sum"),
            num_customers=("num_customers", "sum"),
            state=("state", "first"),
        )
        .reset_index()
    )

    print(f"EIA after cleaning: {len(df_agg):,} unique ZIPs")
    return df_agg


def clean_acs_data(df):
    """Replace Census null sentinel, drop ZCTAs with missing or implausible values."""
    df = df.copy()

    # Census uses -666666666 for suppressed/missing values
    df = df.replace(-666666666, np.nan)

    df = df.dropna(subset=["ZIP", "population", "median_income"])
    df = df[df["population"] >= 100]   # exclude PO Box-only ZCTAs
    df = df[df["median_income"] > 0]

    df["ZIP"] = df["ZIP"].astype(str).str.zfill(5)

    print(f"ACS after cleaning: {len(df):,} ZCTAs retained")
    return df


def merge_eia_acs(eia_df, acs_df):
    """Inner join EIA and ACS on ZIP. ~25% loss expected due to ZIP vs ZCTA mismatch."""
    n_eia = len(eia_df)

    merged = pd.merge(eia_df, acs_df, on="ZIP", how="inner")
    loss_pct = 100 * (1 - len(merged) / n_eia)

    print(f"EIA: {n_eia:,} ZIPs | ACS: {len(acs_df):,} ZCTAs")
    print(f"After inner join: {len(merged):,} rows ({loss_pct:.1f}% loss — expected from ZIP/ZCTA mismatch)")
    return merged


def filter_nyc_la(df):
    """Keep only NYC and LA ZIP codes and add a 'city' column."""
    df = df.copy()
    df["ZIP_int"] = df["ZIP"].astype(int)

    nyc_ranges = [
        (10001, 10282),  # Manhattan
        (10301, 10314),  # Staten Island
        (10451, 10475),  # Bronx
        (11201, 11256),  # Brooklyn
        (11004, 11436),  # Queens
    ]
    la_range = (90001, 91899)

    nyc_mask = pd.Series(False, index=df.index)
    for lo, hi in nyc_ranges:
        nyc_mask |= (df["ZIP_int"] >= lo) & (df["ZIP_int"] <= hi)

    la_mask = (df["ZIP_int"] >= la_range[0]) & (df["ZIP_int"] <= la_range[1])

    df_filtered = df[nyc_mask | la_mask].copy()
    df_filtered["city"] = "LA"
    for lo, hi in nyc_ranges:
        df_filtered.loc[(df_filtered["ZIP_int"] >= lo) & (df_filtered["ZIP_int"] <= hi), "city"] = "NYC"

    print(f"Filtered to {len(df_filtered):,} ZIPs — NYC: {(df_filtered['city']=='NYC').sum()}, LA: {(df_filtered['city']=='LA').sum()}")
    return df_filtered.drop(columns="ZIP_int")


def clean_and_integrate(eia_df, acs_df):
    """Run the full cleaning and integration pipeline."""
    print("Step 1: Clean EIA")
    eia_clean = clean_eia_data(eia_df)

    print("\nStep 2: Clean ACS")
    acs_clean = clean_acs_data(acs_df)

    print("\nStep 3: Merge")
    merged = merge_eia_acs(eia_clean, acs_clean)

    print("\nStep 4: Filter to NYC + LA")
    result = filter_nyc_la(merged)

    print("\nDone.")
    return result