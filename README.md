# Customer Segmentation with K-Means Clustering

A complete, end-to-end unsupervised learning project that segments retail
customers into actionable groups based on demographics and purchasing
behavior.

## Project Structure
```
customer_segmentation/
├── data/
│   └── customers.csv              # 500 synthetic customer records
├── src/
│   ├── generate_data.py           # Creates the sample dataset
│   └── segmentation.py            # Full clustering pipeline
├── outputs/
│   ├── 01_optimal_k_selection.png # Elbow + silhouette score plots
│   ├── 02_pca_clusters.png        # 2D PCA cluster visualization
│   ├── 03_income_vs_spending.png  # Business-friendly cluster view
│   ├── 04_feature_distributions.png
│   ├── 05_cluster_sizes.png
│   ├── cluster_profiles.csv       # Mean feature values per cluster
│   └── segmented_customers.csv    # Original data + cluster labels
└── README.md
```

## Features Used
| Feature | Description |
|---|---|
| Age | Customer age |
| AnnualIncome_k | Annual income in $1,000s |
| SpendingScore | Store-assigned score (1-100) based on spending behavior |
| PurchaseFrequency | Average purchases per month |
| TenureMonths | Months since customer acquisition |

## Methodology
1. **EDA** — summary statistics and missing-value checks.
2. **Preprocessing** — `StandardScaler` to normalize all features (essential
   for K-Means since it uses Euclidean distance).
3. **Optimal K selection** — ran K=2..10, plotted the **Elbow method**
   (inertia/WCSS) and **Silhouette score** side by side.
4. **Model fit** — final `KMeans` with `n_init=10` for stable centroids.
5. **Visualization** — PCA projection to 2D, an Income-vs-Spending scatter
   (the classic, business-readable view), per-feature boxplots, and cluster
   size bars.
6. **Profiling & labeling** — computed mean feature values per cluster and
   applied a simple income/spending heuristic to name each segment.

## How to Run
```bash
python3 src/generate_data.py      # (optional) regenerate the dataset
python3 src/segmentation.py       # run the full pipeline
```
Swap in your own data by replacing `data/customers.csv` — just keep the same
column names, or edit the `FEATURES` list in `segmentation.py`.

## Results Summary
Silhouette analysis confirmed **K=5** as the best-separated solution
(score ≈ 0.36). The five discovered segments:

| Cluster | Segment | Age | Income (k$) | Spending Score | Size |
|---|---|---|---|---|---|
| 0 | Young Impulse Spenders | ~25 | ~32k | High | 21.8% |
| 1 | Budget-Conscious (Established) | ~47 | ~36k | Low | 26.0% |
| 2 | Premium / High-Value | ~37 | ~104k | High | 15.4% |
| 3 | Affluent but Cautious | ~48 | ~109k | Low | 15.8% |
| 4 | Mid-Market / Average | ~39 | ~55k | Mid | 21.0% |

*(exact values in `outputs/cluster_profiles.csv`)*

## Suggested Business Actions
- **Premium / High-Value** → loyalty perks, early access to new products, VIP service.
- **Affluent but Cautious** → trust-building campaigns, quality/value messaging rather than discounts.
- **Young Impulse Spenders** → flash sales, social-media promotions, buy-now-pay-later options.
- **Budget-Conscious (Established)** → discount bundles, loyalty points, value-tier products.
- **Mid-Market / Average** → broad seasonal campaigns; test upsell offers to shift them toward Premium.

## Notes
- The dataset here is **synthetic** (generated with realistic underlying
  segment structure) since no data file was provided. Drop in a real CSV
  (e.g., a Mall Customers or CRM export) with the same columns to run this
  on actual data — no other code changes needed.
