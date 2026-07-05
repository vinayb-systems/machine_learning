# ============================================================
# Task-02: K-Means Clustering — Customer Segmentation
# Dataset: Kaggle — Customer Segmentation Tutorial in Python
# https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python
# Features Used: Annual Income (k$), Spending Score (1-100)
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# ── 1. LOAD DATA ─────────────────────────────────────────────
# Download Mall_Customers.csv from the Kaggle link above.
# Place it in the same directory as this script.

df = pd.read_csv("Mall_Customers.csv")
print("Dataset shape:", df.shape)
print("\nFirst 6 rows:")
print(df.head(6).to_string(index=False))

# ── 2. EXPLORATORY DATA ANALYSIS ─────────────────────────────
print("\n── Dataset Info ──")
print(df.info())
print("\n── Basic Statistics ──")
print(df.describe().to_string())

print("\n── Gender Distribution ──")
print(df["Gender"].value_counts().to_string())

# ── 3. FEATURE SELECTION ─────────────────────────────────────
# Primary clustering: Annual Income vs Spending Score (classic 2D)
# Extended clustering: Age + Annual Income + Spending Score (3D)

X = df[["Annual Income (k$)", "Spending Score (1-100)"]].copy()
X_full = df[["Age", "Annual Income (k$)", "Spending Score (1-100)"]].copy()

# ── 4. FEATURE SCALING ───────────────────────────────────────
# Two separate scalers — scaler_2d is used for clustering & inverse_transform
scaler_2d = StandardScaler()
scaler_3d = StandardScaler()
X_scaled      = scaler_2d.fit_transform(X)
X_full_scaled = scaler_3d.fit_transform(X_full)

# ── 5. ELBOW METHOD — Find Optimal K ─────────────────────────
inertia     = []
sil_scores  = []
K_range     = range(2, 11)

for k in K_range:
    km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=42)
    km.fit(X_scaled)
    inertia.append(km.inertia_)
    sil_scores.append(silhouette_score(X_scaled, km.labels_))

print("\n── Elbow Method Results ──")
print(f"{'K':>4}  {'Inertia':>12}  {'Silhouette Score':>18}")
for k, ine, sil in zip(K_range, inertia, sil_scores):
    print(f"{k:>4}  {ine:>12.2f}  {sil:>18.4f}")

optimal_k = K_range[np.argmax(sil_scores)]
print(f"\n✔ Optimal K (max silhouette): {optimal_k}")

# ── 6. TRAIN FINAL K-MEANS MODEL ─────────────────────────────
kmeans = KMeans(n_clusters=optimal_k, init="k-means++", n_init=10, random_state=42)
df["Cluster"] = kmeans.fit_predict(X_scaled)

print(f"\n── Cluster Counts (K={optimal_k}) ──")
print(df["Cluster"].value_counts().sort_index().to_string())

# ── 7. CLUSTER PROFILING ─────────────────────────────────────
cluster_profile = df.groupby("Cluster").agg(
    Count            = ("CustomerID",            "count"),
    Avg_Age          = ("Age",                   "mean"),
    Avg_Income_k     = ("Annual Income (k$)",    "mean"),
    Avg_SpendScore   = ("Spending Score (1-100)", "mean"),
    Female_Pct       = ("Gender", lambda x: (x == "Female").mean() * 100)
).round(2)

print("\n── Cluster Profiles ──")
print(cluster_profile.to_string())

# ── 8. SEGMENT LABELS (heuristic based on income × spend) ────
SEGMENT_NAMES = {
    0: "Cluster 0",
    1: "Cluster 1",
    2: "Cluster 2",
    3: "Cluster 3",
    4: "Cluster 4",
}

# Auto-label based on Income & Spending Score ranking
profile = cluster_profile[["Avg_Income_k", "Avg_SpendScore"]].copy()
labels_map = {}
high_inc_high_spend = profile[(profile["Avg_Income_k"] >= profile["Avg_Income_k"].median()) &
                               (profile["Avg_SpendScore"] >= profile["Avg_SpendScore"].median())].index
high_inc_low_spend  = profile[(profile["Avg_Income_k"] >= profile["Avg_Income_k"].median()) &
                               (profile["Avg_SpendScore"] <  profile["Avg_SpendScore"].median())].index
low_inc_high_spend  = profile[(profile["Avg_Income_k"] <  profile["Avg_Income_k"].median()) &
                               (profile["Avg_SpendScore"] >= profile["Avg_SpendScore"].median())].index
low_inc_low_spend   = profile[(profile["Avg_Income_k"] <  profile["Avg_Income_k"].median()) &
                               (profile["Avg_SpendScore"] <  profile["Avg_SpendScore"].median())].index
remaining           = set(range(optimal_k)) - set(high_inc_high_spend) - set(high_inc_low_spend) - \
                      set(low_inc_high_spend) - set(low_inc_low_spend)

for idx in high_inc_high_spend: labels_map[idx] = "💎 High Income, High Spender"
for idx in high_inc_low_spend:  labels_map[idx] = "💰 High Income, Low Spender"
for idx in low_inc_high_spend:  labels_map[idx] = "🛒 Low Income, High Spender"
for idx in low_inc_low_spend:   labels_map[idx] = "🏠 Low Income, Low Spender"
for idx in remaining:            labels_map[idx] = "📊 Average Customer"

df["Segment"] = df["Cluster"].map(labels_map)

print("\n── Segment Labels ──")
print(df[["CustomerID", "Gender", "Age",
          "Annual Income (k$)", "Spending Score (1-100)",
          "Cluster", "Segment"]].head(15).to_string(index=False))

# ── 9. OUTPUT FORMAT ─────────────────────────────────────────
output_df = df[["CustomerID", "Gender", "Age",
                "Annual Income (k$)", "Spending Score (1-100)",
                "Cluster", "Segment"]].copy()
output_df.to_csv("customer_clusters.csv", index=False)
print("\nFull results saved → customer_clusters.csv")

# ── 10. VISUALISATIONS — 4 separate figures, no collision possible ───────────
PALETTE = ["#E63946", "#457B9D", "#2A9D8F", "#E9C46A", "#F4A261"]

# ── Figure 1: Elbow + Silhouette ─────────────────────────────
fig1, ax = plt.subplots(figsize=(6, 4))
ax2 = ax.twinx()
ax.plot(K_range, inertia,     "o-",  color="#E63946", lw=2, label="Inertia")
ax2.plot(K_range, sil_scores, "s--", color="#457B9D", lw=2, label="Silhouette Score")
ax.set_xlabel("Number of Clusters (K)")
ax.set_ylabel("Inertia", color="#E63946")
ax2.set_ylabel("Silhouette Score", color="#457B9D")
ax.set_title("Elbow Method + Silhouette Score", fontsize=13, fontweight="bold")
ax.axvline(optimal_k, color="gray", lw=1, linestyle=":")
ax.annotate(f"Optimal K = {optimal_k}",
            xy=(optimal_k, inertia[optimal_k - 2]),
            xytext=(optimal_k + 1, inertia[optimal_k - 2] + 15),
            fontsize=9, color="gray",
            arrowprops=dict(arrowstyle="->", color="gray"))
lines1, lbl1 = ax.get_legend_handles_labels()
lines2, lbl2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, lbl1 + lbl2, loc="upper right", fontsize=9)
fig1.tight_layout()
fig1.savefig("plot1_elbow.png", dpi=150, bbox_inches="tight")
plt.show()

# ── Figure 2: 2D Cluster Scatter ─────────────────────────────
fig2, ax = plt.subplots(figsize=(7, 5))
centroids_orig = scaler_2d.inverse_transform(kmeans.cluster_centers_)
# Use only "C0–C4" as label; put full names in a text box instead
for c in range(optimal_k):
    mask = df["Cluster"] == c
    ax.scatter(df.loc[mask, "Annual Income (k$)"],
               df.loc[mask, "Spending Score (1-100)"],
               color=PALETTE[c], alpha=0.75, edgecolors="white", lw=0.4,
               label=f"C{c}", s=70)
ax.scatter(centroids_orig[:, 0], centroids_orig[:, 1],
           c="black", s=220, marker="X", zorder=5, label="Centroids")
ax.set_xlabel("Annual Income (k$)")
ax.set_ylabel("Spending Score (1–100)")
ax.set_title(f"Customer Clusters (K={optimal_k})", fontsize=13, fontweight="bold")
ax.legend(fontsize=9, loc="upper left")
# Segment name table as text box (right side, no overlap)
seg_lines = "\n".join([f"C{k}: {v}" for k, v in sorted(labels_map.items())])
ax.text(1.02, 0.5, seg_lines, transform=ax.transAxes,
        fontsize=8, va="center", ha="left",
        bbox=dict(boxstyle="round,pad=0.4", fc="lightyellow", ec="gray", alpha=0.8))
fig2.tight_layout()
fig2.savefig("plot2_clusters.png", dpi=150, bbox_inches="tight")
plt.show()

# ── Figure 3: Cluster Profile Bar Chart ──────────────────────
fig3, ax = plt.subplots(figsize=(6, 4))
x = np.arange(optimal_k)
w = 0.35
bars1 = ax.bar(x - w/2, cluster_profile["Avg_Income_k"],
               width=w, color="#457B9D", label="Avg Income (k$)")
bars2 = ax.bar(x + w/2, cluster_profile["Avg_SpendScore"],
               width=w, color="#2A9D8F", label="Avg Spending Score")
ax.set_xticks(x)
ax.set_xticklabels([f"C{i}" for i in cluster_profile.index])
ax.set_ylabel("Value")
ax.set_title("Cluster Profile: Avg Income vs Avg Spending Score",
             fontsize=13, fontweight="bold")
ax.legend(fontsize=9, loc="upper right")
for bar in list(bars1) + list(bars2):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.8,
            f"{bar.get_height():.0f}",
            ha="center", va="bottom", fontsize=8)
fig3.tight_layout()
fig3.savefig("plot3_profile.png", dpi=150, bbox_inches="tight")
plt.show()

# ── Figure 4: Age Distribution ───────────────────────────────
fig4, ax = plt.subplots(figsize=(6, 4))
for c in range(optimal_k):
    subset = df[df["Cluster"] == c]["Age"]
    ax.hist(subset, bins=10, alpha=0.6,
            color=PALETTE[c], label=f"C{c}", edgecolor="white")
ax.set_xlabel("Age")
ax.set_ylabel("Count")
ax.set_title("Age Distribution by Cluster", fontsize=13, fontweight="bold")
ax.legend(fontsize=9, loc="upper right")
fig4.tight_layout()
fig4.savefig("plot4_age.png", dpi=150, bbox_inches="tight")
plt.show()

print("Plots saved → plot1_elbow.png | plot2_clusters.png | plot3_profile.png | plot4_age.png")

# ── 11. SUMMARY ──────────────────────────────────────────────
print("\n" + "="*55)
print("FINAL SUMMARY")
print("="*55)
print(f"Total Customers   : {len(df)}")
print(f"Optimal Clusters  : {optimal_k}")
print(f"Silhouette Score  : {sil_scores[optimal_k - 2]:.4f}")
print("\nCluster Breakdown:")
for c, row in cluster_profile.iterrows():
    print(f"  Cluster {c} ({labels_map.get(c,'?'):35s}): "
          f"{int(row['Count']):>3} customers | "
          f"Avg Income=${row['Avg_Income_k']:.0f}k | "
          f"Avg Score={row['Avg_SpendScore']:.0f}")
