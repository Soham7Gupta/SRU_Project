# Sulphur Recovery Unit (SRU) Modeling using JIT + Random Forest

This project explores predictive modeling for a Sulphur Recovery Unit (SRU) system using a Just-In-Time (JIT) learning framework combined with Random Forest regressors.

## Data

- **Input File:** `Cleaned_IN_Table.csv` – contains 5 input features.
- **Output File:** `Cleaned_OUT_Table.csv` – contains 2 target variables representing SRU output metrics.

The dataset was preprocessed and horizontally stacked to form a complete input-output matrix for supervised regression.

##  Model Overview

We use a two-step approach:
1. **Just-In-Time (JIT) Local Modeling**:
   - For each test instance, find the `k=60` nearest training points using Euclidean distance.
2. **Random Forest Regression**:
   - Two separate Random Forest models are trained on-the-fly for each query to predict the two output variables.

## Current Results

- **Mean Squared Error (MSE):**
  - Output 1: `0.00056856`
  - Output 2: `0.00190613`

- **R-squared (R²):**
  - Output 1: `0.7855`
  - Output 2: `0.6476`

These results indicate that the JIT + RF model captures substantial variance in the data, especially for Output 1.

## Previous Work

Previously, a standalone JIT modeling approach was tried, yielding an average R² around `0.72`. The combination with Random Forest has now improved the performance on both outputs, especially Output 1.

## Visualizations

- Error deviation from mean across samples
- Predicted vs True values for each output variable

These plots help in understanding the local modeling performance and the overall prediction accuracy.

## Work in Progress

- Hyperparameter tuning for `k` (neighbors) and `n_estimators` (RF depth)
- Batch prediction optimizations
- Comparison with global models (e.g., XGBoost, MLP)
- Generalization testing with unseen SRU data


