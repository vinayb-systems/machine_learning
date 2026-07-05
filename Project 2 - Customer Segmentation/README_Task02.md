# 🛍️ Task-02: Customer Segmentation — K-Means Clustering

Group retail store customers into meaningful segments based on their **Annual Income** and **Spending Score** using the K-Means clustering algorithm.

---

## 📂 Dataset
[Kaggle — Customer Segmentation Tutorial in Python](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python)

Download `Mall_Customers.csv` and place it in the same directory as the script.

### CSV Format
```
CustomerID  Gender  Age  Annual Income (k$)  Spending Score (1-100)
1           Male    19   15                  39
2           Male    21   15                  81
3           Female  20   16                  6
...
```

---

## 🔧 Features Used

| Feature | Column | Description |
|---|---|---|
| Annual Income | `Annual Income (k$)` | Customer's annual income in thousands |
| Spending Score | `Spending Score (1-100)` | Store-assigned score based on purchase behaviour |
| Age | `Age` | Used in cluster profiling |
| Gender | `Gender` | Used in cluster profiling |

---

## ⚙️ Algorithm

- **K-Means++** initialization for better centroid seeding
- **Elbow Method** → plots inertia vs K
- **Silhouette Score** → selects optimal K automatically
- **StandardScaler** → features normalized before clustering

---

## 🗂️ Output Format

### Console Output
```
── Elbow Method Results ──
   K       Inertia  Silhouette Score
   2       ...              0.3512
   3       ...              0.4281
   4       ...              0.4891
   5       ...              0.5539   ← optimal
   ...

✔ Optimal K (max silhouette): 5

── Cluster Profiles ──
         Count  Avg_Age  Avg_Income_k  Avg_SpendScore  Female_Pct
Cluster
0           ...     ...           ...             ...         ...

── Segment Labels ──
CustomerID  Gender  Age  Annual Income (k$)  Spending Score (1-100)  Cluster  Segment
         1    Male   19                  15                      39        2  📊 Average Customer
         2    Male   21                  15                      81        4  🛒 Low Income, High Spender
       ...

FINAL SUMMARY
=======================================================
Total Customers   : 200
Optimal Clusters  : 5
Silhouette Score  : 0.5539

Cluster Breakdown:
  Cluster 0 (💎 High Income, High Spender    ):  40 customers | Avg Income=$86k | Avg Score=82
  Cluster 1 (💰 High Income, Low Spender     ):  35 customers | Avg Income=$88k | Avg Score=17
  Cluster 2 (📊 Average Customer             ):  49 customers | Avg Income=$55k | Avg Score=49
  Cluster 3 (🏠 Low Income, Low Spender      ):  38 customers | Avg Income=$26k | Avg Score=20
  Cluster 4 (🛒 Low Income, High Spender     ):  38 customers | Avg Income=$25k | Avg Score=79
```

### customer_clusters.csv
```
CustomerID, Gender, Age, Annual Income (k$), Spending Score (1-100), Cluster, Segment
1, Male, 19, 15, 39, 2, 📊 Average Customer
2, Male, 21, 15, 81, 4, 🛒 Low Income High Spender
...
```

### Plots saved — `customer_segmentation_plots.png` (2×2 grid)
| Panel | Description |
|---|---|
| Top-left | Elbow Curve + Silhouette Score vs K |
| Top-right | 2D Scatter: Income vs Spending Score, colored by cluster |
| Bottom-left | Bar chart: Avg Income & Spending Score per cluster |
| Bottom-right | Age distribution histogram per cluster |

---

## 🚀 How to Run

```bash
# 1. Install dependencies
pip install pandas numpy scikit-learn matplotlib seaborn

# 2. Place Mall_Customers.csv in the same folder, then:
python customer_segmentation_kmeans.py
```

---

## 📦 File Structure

```
├── customer_segmentation_kmeans.py   # Main script
├── Mall_Customers.csv                # Kaggle dataset (download separately)
├── customer_clusters.csv             # Generated: all customers with cluster labels
├── customer_segmentation_plots.png   # Generated: 4-panel visualization
└── README.md
```

---

## 📊 Customer Segments Explained

| Segment | Income | Spending | Strategy |
|---|---|---|---|
| 💎 High Income, High Spender | High | High | Priority VIP customers — loyalty rewards |
| 💰 High Income, Low Spender | High | Low | Target with premium campaigns |
| 🛒 Low Income, High Spender | Low | High | Watch credit risk; offer deals |
| 🏠 Low Income, Low Spender | Low | Low | Budget offers, entry-level products |
| 📊 Average Customer | Mid | Mid | Standard retention strategy |

---

## 🛠️ Dependencies

```
pandas
numpy
scikit-learn
matplotlib
seaborn
```
