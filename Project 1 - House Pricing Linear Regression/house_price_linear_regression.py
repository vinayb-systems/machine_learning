# ============================================================
# Task-01: Linear Regression — House Price Prediction
# Dataset: Kaggle House Prices - Advanced Regression Techniques
# Features Used: GrLivArea (sq ft), BedroomAbvGr, FullBath + HalfBath
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ── 1. LOAD DATA ─────────────────────────────────────────────
# Download train.csv from:
# https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data
# Place it in the same directory as this script.

df = pd.read_csv("train.csv")
print("Dataset shape:", df.shape)
print(df[["GrLivArea", "BedroomAbvGr", "FullBath", "HalfBath", "SalePrice"]].head())

# ── 2. FEATURE ENGINEERING ───────────────────────────────────
# Square Footage  → GrLivArea  (above-ground living area in sq ft)
# Bedrooms        → BedroomAbvGr
# Bathrooms       → FullBath + 0.5 * HalfBath  (weighted count)

df["TotalBathrooms"] = df["FullBath"] + 0.5 * df["HalfBath"]

features = ["GrLivArea", "BedroomAbvGr", "TotalBathrooms"]
target   = "SalePrice"

# Drop rows with missing values in selected columns
data = df[features + [target]].dropna()
print(f"\nRows after dropping NaN: {len(data)}")

X = data[features]
y = data[target]

# ── 3. TRAIN / TEST SPLIT ────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\nTrain size: {X_train.shape[0]}  |  Test size: {X_test.shape[0]}")

# ── 4. FEATURE SCALING ───────────────────────────────────────
scaler  = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# ── 5. TRAIN LINEAR REGRESSION MODEL ────────────────────────
model = LinearRegression()
model.fit(X_train_sc, y_train)

print("\n── Model Coefficients ──")
for feat, coef in zip(features, model.coef_):
    print(f"  {feat:20s}: {coef:,.2f}")
print(f"  {'Intercept':20s}: {model.intercept_:,.2f}")

# ── 6. PREDICTIONS ───────────────────────────────────────────
y_pred = model.predict(X_test_sc)

# ── 7. EVALUATION METRICS ────────────────────────────────────
mae  = mean_absolute_error(y_test, y_pred)
mse  = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2   = r2_score(y_test, y_pred)

print("\n── Evaluation Metrics ──")
print(f"  MAE  (Mean Absolute Error)       : ${mae:,.2f}")
print(f"  MSE  (Mean Squared Error)        : ${mse:,.2f}")
print(f"  RMSE (Root Mean Squared Error)   : ${rmse:,.2f}")
print(f"  R²   (Coefficient of Determination): {r2:.4f}")

# ── 8. OUTPUT FORMAT ─────────────────────────────────────────
# Build a results DataFrame that mirrors a Kaggle submission
results_df = pd.DataFrame({
    "Id"            : X_test.index,          # original row index
    "Actual_Price"  : y_test.values,
    "Predicted_Price": np.round(y_pred, 2),
    "Error"         : np.round(y_test.values - y_pred, 2)
})
print("\n── Sample Predictions (first 10 rows) ──")
print(results_df.head(10).to_string(index=False))

# Save predictions to CSV
results_df.to_csv("predictions.csv", index=False)
print("\nPredictions saved to → predictions.csv")

# ── 9. VISUALISATIONS ────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 8))
fig.suptitle("Linear Regression — House Price Prediction", fontsize=15, fontweight="bold")

# (a) Actual vs Predicted
axes[0].scatter(y_test, y_pred, alpha=0.4, color="#4C72B0", edgecolors="none")
axes[0].plot([y_test.min(), y_test.max()],
             [y_test.min(), y_test.max()], "r--", lw=1, label="Perfect fit")
axes[0].set_xlabel("Actual Price ($)")
axes[0].set_ylabel("Predicted Price ($)")
axes[0].set_title("Actual vs Predicted")
axes[0].legend()

# (b) Residuals
residuals = y_test.values - y_pred
axes[1].scatter(y_pred, residuals, alpha=0.4, color="#DD8452", edgecolors="none")
axes[1].axhline(0, color="red", lw=1.5, linestyle="--")
axes[1].set_xlabel("Predicted Price ($)")
axes[1].set_ylabel("Residual ($)")
axes[1].set_title("Residual Plot")

# (c) Feature correlation heatmap
axes[2].set_title("Feature Correlation Heatmap")
corr = data[features + [target]].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
            ax=axes[2], linewidths=0.5)

plt.tight_layout()
plt.savefig("house_price_lr_plots.png", dpi=150, bbox_inches="tight")
plt.show()
print("Plot saved → house_price_lr_plots.png")

# ── 10. PREDICT ON NEW DATA (example) ────────────────────────
print("\n── Predicting on New Sample Houses ──")
new_houses = pd.DataFrame({
    "GrLivArea"     : [1500, 2200, 800],
    "BedroomAbvGr"  : [3,    4,    2  ],
    "TotalBathrooms": [2.0,  3.0,  1.0]
})
new_scaled = scaler.transform(new_houses)
new_preds  = model.predict(new_scaled)

for i, row in new_houses.iterrows():
    print(f"  House {i+1}: {int(row.GrLivArea)} sqft, "
          f"{int(row.BedroomAbvGr)} bed, {row.TotalBathrooms} bath  "
          f"→  Predicted Price: ${new_preds[i]:,.0f}")
