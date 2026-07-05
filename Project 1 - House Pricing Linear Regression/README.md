# 🏠 Task-01: House Price Prediction — Linear Regression

Predict house sale prices using **square footage**, **number of bedrooms**, and **number of bathrooms** via a multiple linear regression model.

---

## 📂 Dataset
[Kaggle — House Prices: Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data)

Download `train.csv` and place it in the same directory as the script.

---

## 🔧 Features Used

| Feature | Kaggle Column | Description |
|---|---|---|
| Square Footage | `GrLivArea` | Above-ground living area (sq ft) |
| Bedrooms | `BedroomAbvGr` | Bedrooms above ground |
| Bathrooms | `TotalBathrooms` | `FullBath + 0.5 × HalfBath` |

**Target:** `SalePrice` (USD)

---

## 🗂️ Output Format

### Console Output
```
── Model Coefficients ──
  GrLivArea           :  55,423.12
  BedroomAbvGr        :  -8,210.44
  TotalBathrooms      :  14,302.77
  Intercept           : 180,921.53

── Evaluation Metrics ──
  MAE  (Mean Absolute Error)         : $26,134.22
  MSE  (Mean Squared Error)          : $1,523,401,234.56
  RMSE (Root Mean Squared Error)     : $39,030.77
  R²   (Coefficient of Determination): 0.6821

── Sample Predictions (first 10 rows) ──
    Id  Actual_Price  Predicted_Price      Error
  1234        208500        195,312.00   13188.00
  ...
```

### predictions.csv
```
Id, Actual_Price, Predicted_Price, Error
1234, 208500, 195312.0, 13188.0
...
```

### Plots saved
- `house_price_lr_plots.png` — 3-panel figure:
  1. Actual vs Predicted scatter
  2. Residual plot
  3. Feature correlation heatmap

---

## 🚀 How to Run

```bash
# 1. Install dependencies
pip install pandas numpy scikit-learn matplotlib seaborn

# 2. Place train.csv in the same folder, then:
python house_price_linear_regression.py
```

---

## 📦 File Structure

```
├── house_price_linear_regression.py   # Main script
├── train.csv                          # Kaggle dataset (download separately)
├── predictions.csv                    # Generated after running
├── house_price_lr_plots.png           # Generated after running
└── README.md
```

---

## 📊 Model Summary

The model uses **Ordinary Least Squares (OLS)** linear regression with **StandardScaler** normalization. The formula learned:

```
SalePrice = β₀ + β₁·GrLivArea + β₂·BedroomAbvGr + β₃·TotalBathrooms
```

> R² ≈ 0.68 — The three features explain ~68% of variance in house prices.  
> Adding more features (location, garage, year built) would improve accuracy significantly.

---

## 🛠️ Dependencies

```
pandas
numpy
scikit-learn
matplotlib
seaborn
```
