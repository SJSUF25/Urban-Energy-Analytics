# Urban Energy Analytics: NYC vs LA 2022

**Urban Sustainability and Energy Behavior: A 2022 Comparative Study of NYC and LA**

This project examines the relationship between socio-economic characteristics and residential electricity consumption across ZIP codes in New York City and Los Angeles using 2022 data. It applies PCA for dimensionality reduction and compares three clustering algorithms; Agglomerative Hierarchical, K-Means, and DBSCAN—to identify distinct neighborhood energy-use profiles.

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
│   │   ├── acs_zcta_2022.csv              # ACS demographic data
│   │   └── 2022_Gaz_zcta_national.txt     # Census ZCTA centroids (for maps)
│   └── processed/
│       └── nyc_la_merged.csv              # Final clustered dataset (470 ZIPs)
├── notebooks/
│   └── urban_energy_analysis.ipynb        # Main analysis notebook
├── scripts/
│   └── prepare_data.py                    # Data download/processing (reference)
├── src/
│   ├── __init__.py
│   ├── data_loader.py                     # CSV loading utilities
│   ├── data_cleaner.py                    # Cleaning, merging, NYC/LA filtering
│   ├── feature_engineering.py             # 6 derived features
│   └── modeling.py                        # PCA, 3 clustering algos, bootstrap stability
├── reports/
│   ├── algorithm_comparison.csv           # 3-algorithm × 3-metric comparison
│   ├── bootstrap_stability.png            # ARI histogram (100 iterations)
│   ├── pca_loadings_heatmap.png           # Feature loadings on 4 PCs
│   ├── pca_scatter_by_city.png            # NYC vs LA in PCA space
│   ├── cluster_radar_profiles.png         # Radar charts for 6 clusters
│   ├── cluster_map_nyc_la.png             # Side-by-side NYC + LA cluster maps
│   └── cluster_map_combined.png           # Combined geographic view
└── requirements.txt
```

---

## Pipeline

1. **Load** — EIA Form 861 + ACS 2022 CSVs
2. **Clean** — Standardize ZIPs, drop invalid rows, aggregate by ZIP
3. **Merge** — Inner join on ZIP/ZCTA (28,692 matched rows)
4. **Filter** — NYC (185 ZIPs) + LA (291 ZIPs) = 476 ZIPs
5. **Feature Engineering** — 6 features: electricity per customer, electricity per capita, renter occupancy rate, housing age, log income, household size (470 ZIPs after dropping 1 invalid row + 5 EIA attribution outliers with electricity_per_capita > 15 MWh/person)
6. **PCA** — 4 components explaining 88.77% of variance
7. **Clustering** — Compare Agglomerative Hierarchical (Ward), K-Means, DBSCAN
8. **Evaluation** — Silhouette Score, Davies-Bouldin Index, Calinski-Harabasz Index
9. **Geographic visualization** — Cluster maps using Census 2022 ZCTA centroids
10. **Bootstrap Stability** — 100 iterations, 80% subsamples, mean ARI = 0.605 (stable)

---

## Key Results

| Algorithm | Silhouette ↑ | Davies-Bouldin ↓ | Calinski-Harabasz ↑ | Clusters | Noise |
|-----------|-------------|-----------------|--------------------:|----------|-------|
| Hierarchical (silhouette-optimal k=3) | 0.2456 | 1.5115 | 138.28 | 3 | 0 |
| K-Means (k=3) | 0.2709 | 1.3847 | 154.58 | 3 | 0 |
| DBSCAN (eps=0.3) | 0.8957 | 0.135 | 571.07 | 2 | 455 |
| **Hierarchical (Ward, k=6) — selected** | 0.2373 | — | — | 6 | 0 |

**Selected: Agglomerative Hierarchical (Ward, k=6).** Although the silhouette sweep favors k=3, the resulting clusters collapse into three coarse buckets (NYC / LA-affluent / LA-other) that do not support neighborhood-level policy interpretation. Hierarchical k=6 yields six balanced, interpretable clusters with a silhouette only 0.008 below the k=3 maximum. DBSCAN's high scores are an artifact of discarding 455 of 470 ZIPs (97%) as noise.

**Bootstrap stability:** Mean ARI = 0.605 ± 0.091 → cluster assignments are **stable**.

---

## Cluster Summary

| Cluster | ZIPs | NYC | LA | Character |
|---------|------|-----|-----|-----------|
| 0 | 93 | 85 | 8 | NYC-dominant, high income, low usage, dense urban |
| 1 | 25 | 7 | 18 | Mixed, very high income, large households |
| 2 | 114 | 21 | 93 | LA-leaning, highest income, suburban |
| 3 | 71 | 0 | 71 | 100% LA, moderate income, moderate energy use |
| 4 | 92 | 10 | 82 | LA-dominant, moderate income |
| 5 | 75 | 61 | 14 | NYC-dominant, lowest income, oldest housing, high renter rate |

Cluster 3 is city-exclusive (100% LA), and clusters 0 and 5 are nearly so for NYC, confirming that NYC and LA occupy structurally different positions in the socio-economic energy landscape.

---

## Limitations and Future Work

- **No seasonal or peak-demand analysis.** EIA Form 861 reports only annual residential sales per ZIP. Sub-annual patterns (winter heating peaks in NYC, summer cooling peaks in LA) and hourly demand spikes are not resolvable from this dataset. Monthly utility-level data exists in EIA Form 826 but is not ZIP-resolved; hourly grid data exists in EIA-930 only at the balancing-authority level. Adding either would force us to drop the ZIP granularity that the clustering depends on.
- **Two-city scope.** Findings describe NYC and LA in 2022 only and should not be generalized to other metros without re-running the pipeline.
- **ZCTA ≠ ZIP.** ACS demographics are reported at the ZIP Code Tabulation Area level, which approximates but does not exactly match USPS ZIP delivery areas used by EIA. This introduces small boundary mismatches.
- **Cross-sectional snapshot.** A single year (2022) cannot capture year-over-year shifts in efficiency, electrification, or post-pandemic occupancy changes.

**Future work:** Integrate NOAA heating and cooling degree days (HDD/CDD) as climate covariates; incorporate NYC Local Law 84 building benchmarking data for monthly residential consumption (NYC only); and extend the analysis to additional MSAs (Chicago, Houston, and Phoenix) to evaluate the generalizability of cluster archetypes.

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
