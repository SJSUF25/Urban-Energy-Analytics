#!/usr/bin/env python3
"""
prepare_data.py
================
Downloads and processes real data from:
  - EIA Form 861 (2022) sales data + ZIP-level utility mapping from data.gov
  - ACS 2022 5-year ZCTA-level estimates via Census Bureau API

Produces:
  - data/raw/eia861_sales_2022.csv  (ZIP, state, residential_mwh_sales, num_customers)
  - data/raw/acs_zcta_2022.csv      (ZIP, population, median_income,
                                     median_year_structure_built,
                                     renter_occupied_units, total_occupied_units)

Usage:
    cd /Users/spartan/Documents/cmpe255/Project/tmpDAW
    python3 scripts/prepare_data.py
"""

import os
import sys
import zipfile
import io
import requests
import pandas as pd
import numpy as np

# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_RAW = os.path.join(REPO_ROOT, "data", "raw")
EIA_ZIP_PATH = os.path.join(REPO_ROOT, "f8612022.zip")

os.makedirs(DATA_RAW, exist_ok=True)

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
IOU_CSV_URL = "https://data.openei.org/files/5993/iou_zipcodes_2022.csv"
NON_IOU_CSV_URL = "https://data.openei.org/files/5993/non_iou_zipcodes_2022.csv"

# ACS Census API base: 2022 5-year ACS summaries at ZCTA level
ACS_BASE = "https://api.census.gov/data/2022/acs/acs5"
# ZCTA geographic level
ACS_FOR = "zip+code+tabulation+area:*"

# ACS tables we need:
#   B01003_001E = total population
#   B19013_001E = median household income
#   B25035_001E = median year structure built
#   B25003_001E = total occupied housing units
#   B25003_003E = renter-occupied housing units
ACS_VARIABLES = {
    "B01003_001E": "population",
    "B19013_001E": "median_income",
    "B25035_001E": "median_year_structure_built",
    "B25003_001E": "total_occupied_units",
    "B25003_003E": "renter_occupied_units",
}

CENSUS_NULL = -666666666


# ═══════════════════════════════════════════════════════════════════════════
# Part 1: EIA 861 → ZIP-level residential sales
# ═══════════════════════════════════════════════════════════════════════════

def load_eia_sales(eia_zip_path: str) -> pd.DataFrame:
    """Extract utility-level residential MWh sales and customer counts from
    EIA 861 Sales_Ult_Cust_2022.xlsx (States sheet)."""

    print("\n[EIA] Extracting Sales_Ult_Cust_2022.xlsx from EIA 861 ZIP...")
    with zipfile.ZipFile(eia_zip_path, "r") as zf:
        with zf.open("Sales_Ult_Cust_2022.xlsx") as f:
            raw_bytes = f.read()

    # Skip 2 header rows (row 0 = section labels, row 1 = sub-labels, row 2 = actual headers)
    xl = pd.ExcelFile(io.BytesIO(raw_bytes), engine="openpyxl")
    df = xl.parse("States", header=2)  # 0-indexed → rows 0,1,2 read; row 2 is header

    # Columns of interest (per EIA 861 layout):
    #   Col 2  = "Utility Name"
    #   Col 1  = "Utility Number"   (eiaid)
    #   Col 6  = "State"
    #   Col 10 = Residential Megawatthours
    #   Col 11 = Residential customer Count

    df.columns = [str(c).strip() for c in df.columns]
    # Raw columns by position (col indices 0-based from parse result)
    cols = df.columns.tolist()
    # Utility Number is always col 1, State col 6; but Megawatthours/Count
    # appear multiple times. We need the RESIDENTIAL ones (first occurrence = col 10/11)

    # Rename by position
    df_sel = df.iloc[:, [1, 6, 10, 11]].copy()
    df_sel.columns = ["eiaid", "state", "residential_mwh", "num_customers"]

    # Drop non-numeric or total rows
    df_sel["eiaid"] = pd.to_numeric(df_sel["eiaid"], errors="coerce")
    df_sel["residential_mwh"] = pd.to_numeric(df_sel["residential_mwh"], errors="coerce")
    df_sel["num_customers"] = pd.to_numeric(df_sel["num_customers"], errors="coerce")
    df_sel = df_sel.dropna(subset=["eiaid", "residential_mwh", "num_customers"])
    df_sel = df_sel[df_sel["residential_mwh"] > 0]
    df_sel = df_sel[df_sel["num_customers"] > 0]
    df_sel["eiaid"] = df_sel["eiaid"].astype(int)

    print(f"[EIA] Loaded {len(df_sel)} utilities with residential sales data")
    return df_sel


def load_utility_zip_mapping() -> pd.DataFrame:
    """Download IOU and Non-IOU ZIP↔utility mapping CSVs from data.gov."""

    print("\n[EIA] Downloading IOU ZIP mapping from data.gov...")
    iou = pd.read_csv(IOU_CSV_URL, dtype={"zip": str})
    print(f"  IOU rows: {len(iou)}")

    print("[EIA] Downloading Non-IOU ZIP mapping from data.gov...")
    non_iou = pd.read_csv(NON_IOU_CSV_URL, dtype={"zip": str})
    print(f"  Non-IOU rows: {len(non_iou)}")

    mapping = pd.concat([iou, non_iou], ignore_index=True)
    mapping = mapping[["zip", "eiaid", "state"]].drop_duplicates()
    mapping.columns = ["ZIP", "eiaid", "state_zip"]

    # Zero-pad ZIP to 5 digits
    mapping["ZIP"] = mapping["ZIP"].astype(str).str.zfill(5)

    print(f"[EIA] Total utility-ZIP mapping rows: {len(mapping)}")
    return mapping


def build_zip_level_eia(sales: pd.DataFrame, mapping: pd.DataFrame) -> pd.DataFrame:
    """Join utility-level sales with ZIP mapping and distribute sales across ZIPs."""

    print("\n[EIA] Joining sales with ZIP mapping...")
    merged = mapping.merge(sales, on="eiaid", how="inner")
    print(f"[EIA] Joined rows: {len(merged)}")

    # Count how many ZIPs each utility serves
    zip_count = merged.groupby("eiaid")["ZIP"].transform("count")

    # Distribute MWh and customers proportionally (evenly across ZIPs for now)
    merged["residential_mwh_sales"] = merged["residential_mwh"] / zip_count
    merged["num_customers"] = merged["num_customers"] / zip_count

    # Use state from the ZIP mapping (more accurate than sales table in some cases)
    merged["state"] = merged["state_zip"].combine_first(merged["state"])

    # Aggregate by ZIP: sum MWh/customers from all utilities serving that ZIP
    df_zip = (
        merged.groupby("ZIP")
        .agg(
            residential_mwh_sales=("residential_mwh_sales", "sum"),
            num_customers=("num_customers", "sum"),
            state=("state", "first"),
        )
        .reset_index()
    )

    # Drop rows that still have bad values
    df_zip = df_zip[df_zip["residential_mwh_sales"] > 0]
    df_zip = df_zip[df_zip["num_customers"] > 0]

    # Round customers to int
    df_zip["num_customers"] = df_zip["num_customers"].round().astype(int)
    df_zip["residential_mwh_sales"] = df_zip["residential_mwh_sales"].round(2)

    print(f"[EIA] Final ZIP-level rows: {len(df_zip)}")
    print(f"  NY ZIPs: {len(df_zip[df_zip['state'] == 'NY'])}")
    print(f"  CA ZIPs: {len(df_zip[df_zip['state'] == 'CA'])}")

    return df_zip[["ZIP", "state", "residential_mwh_sales", "num_customers"]]


# ═══════════════════════════════════════════════════════════════════════════
# Part 2: ACS 2022 → ZCTA-level socio-economic data
# ═══════════════════════════════════════════════════════════════════════════

def fetch_acs_data() -> pd.DataFrame:
    """Download 2022 ACS 5-year ZCTA data for required variables via Census API.

    Census API returns data for all ZCTAs in one request (no geographic subset needed).
    Endpoint: https://api.census.gov/data/2022/acs/acs5?get=VAR1,VAR2,...&for=zip+code+tabulation+area:*
    """

    vars_param = ",".join(ACS_VARIABLES.keys())
    url = f"{ACS_BASE}?get={vars_param}&for={ACS_FOR}"

    print(f"\n[ACS] Downloading ZCTA-level ACS 2022 data...")
    print(f"  URL snippet: ...?get={vars_param[:80]}...")

    try:
        resp = requests.get(url, timeout=120)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"[ACS] ERROR fetching data: {e}")
        sys.exit(1)

    data = resp.json()
    headers = data[0]
    rows = data[1:]

    df = pd.DataFrame(rows, columns=headers)
    print(f"[ACS] Received {len(df)} ZCTAs from Census API")

    # Rename ZCTA column
    df = df.rename(columns={"zip code tabulation area": "ZIP"})

    # Rename ACS variable columns
    df = df.rename(columns=ACS_VARIABLES)

    # Keep only what we need
    keep_cols = ["ZIP"] + list(ACS_VARIABLES.values())
    df = df[keep_cols].copy()

    # Cast numeric columns
    for col in list(ACS_VARIABLES.values()):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Replace Census null sentinel
    df = df.replace(CENSUS_NULL, np.nan)

    # Drop rows missing critical values
    df = df.dropna(subset=["population", "median_income"])
    df = df[df["population"] > 100]  # Remove tiny / PO Box ZCTAs
    df = df[df["median_income"] > 0]

    # Zero-pad ZIP
    df["ZIP"] = df["ZIP"].astype(str).str.zfill(5)

    print(f"[ACS] Retained {len(df)} ZCTAs after cleaning")
    return df


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("Urban Energy Analysis — Real Data Preparation")
    print("=" * 60)

    # ── EIA ───────────────────────────────────────────────────────────────
    if not os.path.exists(EIA_ZIP_PATH):
        print(f"\n[ERROR] EIA ZIP not found at: {EIA_ZIP_PATH}")
        print("Please download f8612022.zip from https://www.eia.gov/electricity/data/eia861/")
        sys.exit(1)

    eia_sales = load_eia_sales(EIA_ZIP_PATH)
    zip_mapping = load_utility_zip_mapping()
    eia_zip = build_zip_level_eia(eia_sales, zip_mapping)

    eia_out = os.path.join(DATA_RAW, "eia861_sales_2022.csv")
    eia_zip.to_csv(eia_out, index=False)
    print(f"\n[EIA] Saved → {eia_out}  ({len(eia_zip)} rows)")

    # ── ACS ───────────────────────────────────────────────────────────────
    acs_df = fetch_acs_data()

    acs_out = os.path.join(DATA_RAW, "acs_zcta_2022.csv")
    acs_df.to_csv(acs_out, index=False)
    print(f"[ACS] Saved → {acs_out}  ({len(acs_df)} rows)")

    # ── Quick validation ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Validation")
    print("=" * 60)
    eia_check = pd.read_csv(eia_out, dtype={"ZIP": str})
    acs_check = pd.read_csv(acs_out, dtype={"ZIP": str})

    required_eia = {"ZIP", "state", "residential_mwh_sales", "num_customers"}
    required_acs = {
        "ZIP", "population", "median_income",
        "median_year_structure_built", "renter_occupied_units", "total_occupied_units",
    }

    assert required_eia.issubset(eia_check.columns), f"Missing EIA cols: {required_eia - set(eia_check.columns)}"
    assert required_acs.issubset(acs_check.columns), f"Missing ACS cols: {required_acs - set(acs_check.columns)}"

    ny = len(eia_check[eia_check["state"] == "NY"])
    ca = len(eia_check[eia_check["state"] == "CA"])
    print(f"EIA rows: {len(eia_check)} | NY: {ny} | CA: {ca}")
    print(f"ACS rows: {len(acs_check)}")

    assert ny > 100, f"Too few NY ZIPs ({ny})"
    assert ca > 100, f"Too few CA ZIPs ({ca})"

    print("\n✓ All checks passed — data is ready for the notebook!")


if __name__ == "__main__":
    main()