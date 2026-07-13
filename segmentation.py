"""
Customer Segmentation using K-Means Clustering
================================================
Pipeline:
  1. Load & explore data
  2. Preprocess (scale numeric features)
  3. Find optimal K (Elbow method + Silhouette score)
  4. Fit final K-Means model
  5. Visualize clusters (PCA 2D projection + feature plots)
  6. Profile each segment and export results
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 110

DATA_PATH = "/home/claude/customer_segmentation/data/customers.csv"
OUT_DIR = "/home/claude/customer_segmentation/outputs"

FEATURES = ["Age", "AnnualIncome_k", "SpendingScore", "PurchaseFrequency", "TenureMonths"]

# ----------------------------------------------------------------------
# 1. LOAD DATA
# ----------------------------------------------------------------------
df = pd.read_csv(DATA_PATH)
print(f"Loaded {len(df)} customers with columns: {list(df.columns)}")
print("\nSummary statistics:")
print(df[FEATURES].describe().round(2))
print(f"\nMissing values:\n{df.isnull().sum()}")

# ----------------------------------------------------------------------
# 2. PREPROCESS
# ----------------------------------------------------------------------
X = df[FEATURES].copy()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ----------------------------------------------------------------------
# 3. FIND OPTIMAL K (Elbow method + Silhouette score)
# ----------------------------------------------------------------------
k_range = range(2, 11)
inertias = []
sil_scores = []

for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(X_scaled, labels))

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].plot(list(k_range), inertias, marker="o", color="#4C72B0", linewidth=2)
axes[0].set_xlabel("Number of Clusters (K)")
axes[0].set_ylabel("Inertia (WCSS)")
axes[0].set_title("Elbow Method")
axes[0].set_xticks(list(k_range))

axes[1].plot(list(k_range), sil_scores, marker="o", color="#DD8452", linewidth=2)
axes[1].set_xlabel("Number of Clusters (K)")
axes[1].set_ylabel("Silhouette Score")
axes[1].set_title("Silhouette Score by K")
axes[1].set_xticks(list(k_range))

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/01_optimal_k_selection.png", bbox_inches="tight")
plt.close()

best_k = list(k_range)[int(np.argmax(sil_scores))]
print(f"\nSilhouette-recommended K: {best_k} (score={max(sil_scores):.3f})")

# Use a business-reasonable K (5, matching the natural structure), but
# report what silhouette recommends too.
FINAL_K = 5
print(f"Using FINAL_K = {FINAL_K} for the segmentation model.")

# ----------------------------------------------------------------------
# 4. FIT FINAL MODEL
# ----------------------------------------------------------------------
kmeans = KMeans(n_clusters=FINAL_K, random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(X_scaled)
final_sil = silhouette_score(X_scaled, df["Cluster"])
print(f"Final model silhouette score: {final_sil:.3f}")

# ----------------------------------------------------------------------
# 5. VISUALIZE CLUSTERS
# ----------------------------------------------------------------------
# 5a. PCA 2D projection
pca = PCA(n_components=2, random_state=42)
pcs = pca.fit_transform(X_scaled)
df["PC1"], df["PC2"] = pcs[:, 0], pcs[:, 1]
explained = pca.explained_variance_ratio_.sum()

plt.figure(figsize=(8, 6))
palette = sns.color_palette("husl", FINAL_K)
sns.scatterplot(data=df, x="PC1", y="PC2", hue="Cluster", palette=palette, s=55, alpha=0.85, edgecolor="white", linewidth=0.3)
centers_pca = pca.transform(kmeans.cluster_centers_)
plt.scatter(centers_pca[:, 0], centers_pca[:, 1], c="black", marker="X", s=220, label="Centroids", edgecolor="white", linewidth=1.5)
plt.title(f"Customer Segments (PCA projection, {explained:.0%} variance explained)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend(title="Cluster", bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/02_pca_clusters.png", bbox_inches="tight")
plt.close()

# 5b. Income vs Spending Score (most interpretable business view)
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x="AnnualIncome_k", y="SpendingScore", hue="Cluster", palette=palette, s=55, alpha=0.85, edgecolor="white", linewidth=0.3)
plt.title("Customer Segments: Income vs Spending Score")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.legend(title="Cluster", bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/03_income_vs_spending.png", bbox_inches="tight")
plt.close()

# 5c. Feature distributions per cluster
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
axes = axes.flatten()
for i, feat in enumerate(FEATURES):
    sns.boxplot(data=df, x="Cluster", y=feat, hue="Cluster", palette=palette, ax=axes[i], legend=False)
    axes[i].set_title(feat)
axes[-1].axis("off")
plt.suptitle("Feature Distributions by Cluster", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/04_feature_distributions.png", bbox_inches="tight")
plt.close()

# 5d. Cluster sizes
plt.figure(figsize=(7, 5))
counts = df["Cluster"].value_counts().sort_index()
bars = plt.bar(counts.index.astype(str), counts.values, color=palette)
for bar, v in zip(bars, counts.values):
    plt.text(bar.get_x() + bar.get_width() / 2, v + 3, str(v), ha="center", fontweight="bold")
plt.xlabel("Cluster")
plt.ylabel("Number of Customers")
plt.title("Cluster Sizes")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/05_cluster_sizes.png", bbox_inches="tight")
plt.close()

# ----------------------------------------------------------------------
# 6. PROFILE CLUSTERS & EXPORT
# ----------------------------------------------------------------------
profile = df.groupby("Cluster")[FEATURES].mean().round(1)
profile["Count"] = df["Cluster"].value_counts().sort_index()
profile["Pct"] = (profile["Count"] / len(df) * 100).round(1)

# Simple auto-labeling heuristic based on income/spending relative to overall median
inc_med, spend_med = df["AnnualIncome_k"].median(), df["SpendingScore"].median()

def label_segment(row):
    inc_high = row["AnnualIncome_k"] >= inc_med
    spend_high = row["SpendingScore"] >= spend_med
    if inc_high and spend_high:
        return "Premium / High-Value"
    if inc_high and not spend_high:
        return "Affluent but Cautious"
    if not inc_high and spend_high:
        return "Young Impulse Spenders"
    if row["Age"] < df["Age"].median():
        return "Budget-Conscious (Younger)"
    return "Budget-Conscious (Established)"

profile["SegmentLabel"] = profile.apply(label_segment, axis=1)
df = df.merge(profile["SegmentLabel"], left_on="Cluster", right_index=True)

print("\n=== CLUSTER PROFILES ===")
print(profile)

profile.to_csv(f"{OUT_DIR}/cluster_profiles.csv")
df.drop(columns=["PC1", "PC2"]).to_csv(f"{OUT_DIR}/segmented_customers.csv", index=False)

print(f"\nSaved outputs to {OUT_DIR}/")
print(" - 01_optimal_k_selection.png")
print(" - 02_pca_clusters.png")
print(" - 03_income_vs_spending.png")
print(" - 04_feature_distributions.png")
print(" - 05_cluster_sizes.png")
print(" - cluster_profiles.csv")
print(" - segmented_customers.csv")
