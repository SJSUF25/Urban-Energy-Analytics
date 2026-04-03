# Urban Energy Analytics: NYC vs LA 2022

**Urban Sustainability and Energy Behavior: A 2022 Comparative Study of NYC and LA**

This project analyzes the relationship between socio-economic characteristics and residential electricity consumption across ZIP codes in New York City and Los Angeles using 2022 data. We apply PCA for dimensionality reduction and compare three clustering algorithms (Agglomerative Hierarchical, K-Means, DBSCAN) to identify distinct neighborhood energy profiles.

**GitHub Repository:** https://github.com/SJSUF25/Urban-Energy-Analytics

---

## Data Sources

| Dataset | Source | Records |
|---------|--------|---------|
| EIA Form 861 ZIP-level residential electricity sales (2022) | [U.S. Energy Information Administration](https://www.eia.gov/electricity/data/eia861/) | 39,075 ZIPs |
| ACS 2022 5-Year Estimates (ZCTA-level demographics) | [U.S. Census Bureau](https://data.census.gov) | 29,996 ZCTAs |

Both datasets are committed to `data/raw/` — no API keys or downloads needed to run the notebook.

---

## Setup

### Local (recommended)

```bash
git clone https://github.com/SJSUF25/Urban-Energy-Analytics.git
cd Urban-Energy-Analytics
pip install -r requirements.txt
jupyter notebook notebooks/urban_energy_analysis.ipynb
```

### Google Colab

The notebook auto-detects Colab and mounts Google Drive. Update the `_repo_root` path in the first code cell to match your Drive location, then run all cells.

---

## Repository Structure

```
Urban-Energy-Analytics/
├── data/
│   ├── raw/
│   │   ├── eia861_sales_2022.csv          # EIA residential electricity data
│   │   └── acs_zcta_2022.csv              # ACS demographic data
│   └── processed/
│       └── nyc_la_merged.csv              # Final clustered dataset (475 ZIPs)
├── notebooks/
│   └── urban_energy_analysis.ipynb        # Main analysis notebook
├── scripts/
│   └── prepare_data.py                    # Data download/processing (reference)
├── src/
│   ├── __init__.py
│   ├── data_loader.py                     # CSV loading utilities
│   ├── data_cleaner.py                    # Cleaning, merging, NYC/LA filtering
│   ├── feature_engineering.py             # 5 derived features
│   └── modeling.py                        # PCA, 3 clustering algos, bootstrap stability
├── reports/
│   ├── algorithm_comparison.csv           # 3-algorithm × 3-metric comparison
│   ├── bootstrap_stability.png            # ARI histogram (100 iterations)
│   ├── pca_loadings_heatmap.png           # Feature loadings on 4 PCs
│   ├── pca_scatter_by_city.png            # NYC vs LA in PCA space
│   └── cluster_radar_profiles.png         # Radar charts for 6 clusters
└── requirements.txt
```

---

## Pipeline

1. **Load** — EIA Form 861 + ACS 2022 CSVs
2. **Clean** — Standardize ZIPs, drop invalid rows, aggregate by ZIP
3. **Merge** — Inner join on ZIP/ZCTA (28,692 matched rows)
4. **Filter** — NYC (185 ZIPs) + LA (291 ZIPs) = 476 ZIPs
5. **Feature Engineering** — 5 features: electricity per customer, electricity per capita, renter occupancy rate, housing age, log income (475 ZIPs after dropping 1 invalid row)
6. **PCA** — 4 components explaining 93.03% of variance
7. **Clustering** — Compare Agglomerative Hierarchical (Ward), K-Means, DBSCAN
8. **Evaluation** — Silhouette Score, Davies-Bouldin Index, Calinski-Harabasz Index
9. **Bootstrap Stability** — 100 iterations, 80% subsamples, mean ARI = 0.666 (stable)

---

## Key Results

| Algorithm | Silhouette ↑ | Davies-Bouldin ↓ | Calinski-Harabasz ↑ | Clusters | Noise |
|-----------|-------------|-----------------|--------------------:|----------|-------|
| **Hierarchical (k=6)** | 0.2697 | 1.0045 | 181.33 | 6 | 0 |
| K-Means (k=3) | 0.3181 | 0.9764 | 190.19 | 3 | 0 |
| DBSCAN (eps=0.7) | 0.4679 | 0.4821 | 28.87 | 2 | 48 |

**Selected: Agglomerative Hierarchical (Ward, k=6)** — produces 6 interpretable, balanced neighborhood profiles. K-Means k=3 creates a degenerate 4-ZIP outlier cluster; DBSCAN discards 10% of data as noise.

**Bootstrap stability:** Mean ARI = 0.666 ± 0.101 → cluster assignments are **stable**.

---

## Cluster Summary

| Cluster | ZIPs | NYC | LA | Character |
|---------|------|-----|-----|-----------|
| 0 | 148 | 16 | 132 | LA-dominant, moderate-high income, low renter rate |
| 1 | 84 | 69 | 15 | NYC-dominant, low income, high renter rate, old housing |
| 2 | 4 | 0 | 4 | Outlier — very low population, extreme elec/capita |
| 3 | 38 | 35 | 3 | NYC-dominant, high income, newer housing |
| 4 | 136 | 0 | 136 | 100% LA, moderate income, moderate energy use |
| 5 | 65 | 64 | 1 | ~100% NYC, lowest energy use, oldest housing |

Clusters 4 and 5 are city-exclusive, confirming that NYC and LA occupy structurally different positions in the socio-economic energy landscape.

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

1. Jolliffe, I. T. (2002). *Principal Component Analysis*. Springer.
2. Jain, A. K., Murty, M. N., & Flynn, P. J. (1999). Data Clustering: A Review. *ACM Computing Surveys*, 31(3), 264–323.
3. Rousseeuw, P. J. (1987). Silhouettes: A graphical aid to the interpretation and validation of cluster analysis. *J. Computational and Applied Mathematics*, 20, 53–65.
4. Davies, D. L. & Bouldin, D. W. (1979). A cluster separation measure. *IEEE TPAMI*, 1(2), 224–227.
5. U.S. Energy Information Administration. EIA Form 861. https://www.eia.gov/electricity/data/eia861/
6. U.S. Census Bureau. American Community Survey 2022. https://data.census.gov
