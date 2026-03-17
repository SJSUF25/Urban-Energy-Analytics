# Urban Energy Analytics: NYC vs LA 2022

This project analyzes residential electricity consumption patterns across ZIP codes in New York City and Los Angeles using 2022 data. We use PCA for dimensionality reduction and compare three clustering algorithms (K-Means, Hierarchical, DBSCAN) to identify distinct neighborhood energy profiles and understand how socio-economic factors relate to electricity usage.

**Data sources:** EIA Form 861 2022 (residential electricity sales) + ACS 2022 5-year estimates (demographics)

---

## Setup

```bash
git clone <repo>
cd Urban-Energy-Analytics
pip install -r requirements.txt
jupyter notebook notebooks/urban_energy_analysis.ipynb
```

### Google Colab

1. Upload the repo or clone it in Colab
2. Mount Drive if needed and install dependencies:
   ```python
   !pip install -r requirements.txt
   ```
3. Run cells in order

---

## Repository Structure

```
Urban-Energy-Analytics/
├── data/
│   ├── raw/
│   │   ├── eia861_sales_2022.csv       # EIA residential electricity data
│   │   └── acs_zcta_2022.csv           # ACS demographic data
│   └── processed/
│       └── nyc_la_merged.csv           # Pipeline output
├── notebooks/
│   └── urban_energy_analysis.ipynb     # Main analysis
├── scripts/
│   └── prepare_data.py                 # Data download script (run once)
├── src/
│   ├── data_loader.py
│   ├── data_cleaner.py
│   ├── feature_engineering.py
│   └── modeling.py
└── reports/
    └── report_draft.md
```

The datasets are committed to the repo — no API keys or downloads needed to run the notebook.

---

## Data Pipeline

1. Load EIA + ACS CSVs from `data/raw/`
2. Clean: standardize ZIP codes, remove invalid rows
3. Merge: inner join on ZIP/ZCTA (~26% loss expected — normal for ZIP vs ZCTA mismatch)
4. Filter: keep only NYC and LA ZIP codes
5. Feature engineering: 5 derived features per ZIP
6. PCA: reduce to components explaining 85%+ variance
7. Clustering: compare K-Means, Hierarchical, and DBSCAN
8. Evaluate: Silhouette, Davies-Bouldin, Calinski-Harabasz metrics

---

## Team

| Name | SJSU ID |
|------|---------|
| Atharva Prasanna Mokashi | 019117046 |
| Maitreya Patankar | 019146166 |
| Vineet Malewar | 018399589 |
| Shefali Saini | 018281848 |

---

## References

- EIA Form 861: https://www.eia.gov/electricity/data/eia861/
- ACS 2022 5-Year Estimates: https://data.census.gov

