"""PCA, clustering (Hierarchical / K-Means / DBSCAN), and evaluation."""

import warnings
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# sklearn's PCA emits cosmetic divide-by-zero RuntimeWarnings during _assess_dimension
# when one feature has a very wide tail (electricity_per_capita does, due to EIA's
# ZIP-attribution artifacts). Outputs are finite and correct; suppress the noise.
warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn")
from sklearn.cluster import AgglomerativeClustering, KMeans, DBSCAN
from sklearn.metrics import (
    silhouette_score,
    silhouette_samples,
    davies_bouldin_score,
    calinski_harabasz_score,
    adjusted_rand_score,
)


def standardize_features(feature_matrix):
    """StandardScaler fit+transform. Returns (scaler, scaled_array)."""
    scaler = StandardScaler()
    scaled = scaler.fit_transform(feature_matrix)
    print(f"Standardized {scaled.shape[1]} features for {scaled.shape[0]} ZIPs")
    return scaler, scaled


def apply_pca(std_features, variance_threshold=0.85):
    """
    Run PCA and keep enough components to explain >= variance_threshold of variance.
    Returns a dict with: pca, transformed, n_components, explained_variance_ratio, cumulative_variance
    """
    pca_full = PCA().fit(std_features)
    cumsum = np.cumsum(pca_full.explained_variance_ratio_)
    n_comp = np.argmax(cumsum >= variance_threshold) + 1

    pca = PCA(n_components=n_comp)
    transformed = pca.fit_transform(std_features)

    print(f"PCA: {n_comp} components explain {cumsum[n_comp-1]:.2%} variance")
    print(f"Per-component variance: {pca.explained_variance_ratio_}")

    return {
        "pca": pca,
        "transformed": transformed,
        "n_components": n_comp,
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "cumulative_variance": cumsum[:n_comp],
    }


def apply_clustering(pca_data, k_min=2, k_max=7):
    """
    Agglomerative (Ward) clustering. Sweeps k and picks the best silhouette score.
    Returns a dict with: clustering, labels, optimal_k, silhouette_scores, best_silhouette
    """
    X = pca_data["transformed"]
    scores = {}

    for k in range(k_min, k_max + 1):
        labels = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(X)
        scores[k] = silhouette_score(X, labels)
        print(f"  Hierarchical k={k}: silhouette={scores[k]:.4f}")

    best_k = max(scores, key=scores.get)
    best_labels = AgglomerativeClustering(n_clusters=best_k, linkage="ward").fit_predict(X)
    print(f"Best k={best_k} (silhouette={scores[best_k]:.4f})")

    return {
        "clustering": AgglomerativeClustering(n_clusters=best_k, linkage="ward"),
        "labels": best_labels,
        "optimal_k": best_k,
        "silhouette_scores": scores,
        "best_silhouette": scores[best_k],
    }


def apply_kmeans(pca_data, k_min=2, k_max=7, random_state=42):
    """
    K-Means clustering. Sweeps k and picks the best silhouette score.
    Returns a dict with: labels, optimal_k, silhouette_scores, inertia_scores, best_silhouette
    """
    X = pca_data["transformed"]
    sil_scores = {}
    inertia_scores = {}

    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = km.fit_predict(X)
        sil_scores[k] = silhouette_score(X, labels)
        inertia_scores[k] = km.inertia_
        print(f"  K-Means k={k}: silhouette={sil_scores[k]:.4f}, inertia={km.inertia_:.2f}")

    best_k = max(sil_scores, key=sil_scores.get)
    best_labels = KMeans(n_clusters=best_k, random_state=random_state, n_init=10).fit_predict(X)
    print(f"Best k={best_k} (silhouette={sil_scores[best_k]:.4f})")

    return {
        "labels": best_labels,
        "optimal_k": best_k,
        "silhouette_scores": sil_scores,
        "inertia_scores": inertia_scores,
        "best_silhouette": sil_scores[best_k],
    }


def apply_dbscan(pca_data, eps_values=None, min_samples=5):
    """
    DBSCAN clustering. Sweeps eps values and picks the best silhouette (on non-noise points).
    Returns a dict with: labels, best_eps, n_clusters, n_noise, silhouette_scores, best_silhouette
    Returns None if no valid configuration is found.
    """
    X = pca_data["transformed"]

    if eps_values is None:
        eps_values = [0.3, 0.5, 0.7, 1.0, 1.5, 2.0]

    results = {}
    for eps in eps_values:
        labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = (labels == -1).sum()

        if n_clusters < 2:
            print(f"  DBSCAN eps={eps}: {n_clusters} cluster(s), {n_noise} noise — skipping")
            continue

        mask = labels != -1
        if mask.sum() < 2:
            continue

        sil = silhouette_score(X[mask], labels[mask])
        results[eps] = {"labels": labels, "n_clusters": n_clusters, "n_noise": n_noise, "silhouette": sil}
        print(f"  DBSCAN eps={eps}: {n_clusters} clusters, {n_noise} noise, silhouette={sil:.4f}")

    if not results:
        print("No valid DBSCAN configuration found. Try adjusting eps_values or min_samples.")
        return None

    best_eps = max(results, key=lambda e: results[e]["silhouette"])
    best = results[best_eps]
    print(f"Best eps={best_eps}: {best['n_clusters']} clusters, silhouette={best['silhouette']:.4f}")

    return {
        "labels": best["labels"],
        "best_eps": best_eps,
        "n_clusters": best["n_clusters"],
        "n_noise": best["n_noise"],
        "silhouette_scores": {e: r["silhouette"] for e, r in results.items()},
        "best_silhouette": best["silhouette"],
        "all_results": results,
    }


def compare_algorithms(pca_data, hierarchical_result, kmeans_result, dbscan_result=None):
    """
    Compute Silhouette, Davies-Bouldin, and Calinski-Harabasz scores for each algorithm
    and return a summary DataFrame.
    """
    X = pca_data["transformed"]
    rows = []

    h_labels = hierarchical_result["labels"]
    rows.append({
        "Algorithm": f"Hierarchical (k={hierarchical_result['optimal_k']})",
        "Silhouette ↑": round(silhouette_score(X, h_labels), 4),
        "Davies-Bouldin ↓": round(davies_bouldin_score(X, h_labels), 4),
        "Calinski-Harabasz ↑": round(calinski_harabasz_score(X, h_labels), 2),
        "n_clusters": hierarchical_result["optimal_k"],
        "n_noise": 0,
    })

    km_labels = kmeans_result["labels"]
    rows.append({
        "Algorithm": f"K-Means (k={kmeans_result['optimal_k']})",
        "Silhouette ↑": round(silhouette_score(X, km_labels), 4),
        "Davies-Bouldin ↓": round(davies_bouldin_score(X, km_labels), 4),
        "Calinski-Harabasz ↑": round(calinski_harabasz_score(X, km_labels), 2),
        "n_clusters": kmeans_result["optimal_k"],
        "n_noise": 0,
    })

    if dbscan_result is not None:
        db_labels = dbscan_result["labels"]
        mask = db_labels != -1
        if mask.sum() >= 2 and len(set(db_labels[mask])) >= 2:
            rows.append({
                "Algorithm": f"DBSCAN (eps={dbscan_result['best_eps']})",
                "Silhouette ↑": round(silhouette_score(X[mask], db_labels[mask]), 4),
                "Davies-Bouldin ↓": round(davies_bouldin_score(X[mask], db_labels[mask]), 4),
                "Calinski-Harabasz ↑": round(calinski_harabasz_score(X[mask], db_labels[mask]), 2),
                "n_clusters": dbscan_result["n_clusters"],
                "n_noise": dbscan_result["n_noise"],
            })

    df = pd.DataFrame(rows).set_index("Algorithm")

    print("\nAlgorithm Comparison")
    print("=" * 60)
    print(df.to_string())
    print(f"\nBest Silhouette:         {df['Silhouette ↑'].idxmax()}")
    print(f"Best Davies-Bouldin:     {df['Davies-Bouldin ↓'].idxmin()}")
    print(f"Best Calinski-Harabasz:  {df['Calinski-Harabasz ↑'].idxmax()}")

    return df


def bootstrap_stability(pca_data, n_clusters, n_iterations=100, subsample_frac=0.8, random_state=42):
    """
    Assess cluster stability by running agglomerative clustering on random 80% subsamples
    and computing Adjusted Rand Index against the full-data labels for matching indices.
    Returns a dict with: ari_scores, mean_ari, std_ari
    """
    X = pca_data["transformed"]
    rng = np.random.default_rng(random_state)

    full_labels = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward").fit_predict(X)

    ari_scores = []
    n_subsample = int(len(X) * subsample_frac)

    for _ in range(n_iterations):
        idx = rng.choice(len(X), size=n_subsample, replace=False)
        sub_labels = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward").fit_predict(X[idx])
        ari = adjusted_rand_score(full_labels[idx], sub_labels)
        ari_scores.append(ari)

    mean_ari = np.mean(ari_scores)
    std_ari = np.std(ari_scores)
    print(f"Bootstrap stability (n={n_iterations}, subsample={subsample_frac:.0%}): "
          f"mean ARI={mean_ari:.4f}, std={std_ari:.4f}")

    if mean_ari > 0.6:
        print("Interpretation: STABLE (ARI > 0.6)")
    elif mean_ari >= 0.4:
        print("Interpretation: MODERATE (ARI 0.4–0.6)")
    else:
        print("Interpretation: UNSTABLE (ARI < 0.4)")

    return {"ari_scores": ari_scores, "mean_ari": mean_ari, "std_ari": std_ari}


def evaluate_clustering(df, pca_data, clustering_data):
    """
    Profile the clusters: mean feature values per cluster, silhouette samples, city distribution.
    Returns a dict with: df_clustered, silhouette_values, profiles, city_distribution
    """
    X = pca_data["transformed"]
    labels = clustering_data["labels"]

    df_clustered = df.copy()
    df_clustered["cluster"] = labels

    silhouette_vals = silhouette_samples(X, labels)

    profile_cols = [
        "electricity_per_customer",
        "electricity_per_capita",
        "renter_occupancy_rate",
        "housing_age",
        "income_log",
        "household_size",
        "median_income",
    ]

    profiles = df_clustered.groupby("cluster")[profile_cols].mean()
    city_dist = pd.crosstab(df_clustered["cluster"], df_clustered["city"], margins=True)

    print("\nCluster Profiles (means):")
    print(profiles)
    print("\nCluster distribution by city:")
    print(city_dist)

    return {
        "df_clustered": df_clustered,
        "silhouette_values": silhouette_vals,
        "profiles": profiles,
        "city_distribution": city_dist,
    }
