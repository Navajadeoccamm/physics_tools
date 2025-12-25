# This file is part of the project's source code.
# Copyright (c) 2025 Daniel Monzon.
# Licensed under the MIT License. See the LICENSE file in the repository root.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# ========================================
# 1. Importing data from a CSV file
# ========================================
df = pd.read_csv("test_data.csv")  # df: pandas DataFrame
x = df.iloc[:, 0].values           # Values from the first column
y = df.iloc[:, 1].values           # Values from the second column
# ============================
# 2. Polynomial Fitting
# ============================
coef = np.polyfit(x, y, 2)          # Fit a 2nd-degree polynomial using least squares
a, b, c = coef                     # Resulting coefficients (a·x² + b·x + c)
p = np.poly1d(coef)                # Create a polynomial function from the coefficients
# ============================
# 3. Plotting the fit
# ============================
x_fit = np.linspace(min(x), max(x), 300)  # Generate points for a smooth fitted curve
y_fit = p(x_fit)                          # Values of the fitted curve (for plotting)

# Predictions at the original x points (same length as y)
y_pred = p(x)

# Pearson correlation coefficient (r) between observed and predicted values
corr_matrix = np.corrcoef(y, y_pred)
corr = corr_matrix[0, 1]

# Coefficient of determination R² (R-squared)
ss_res = np.sum((y - y_pred)**2)          # Sum of squared residuals
ss_tot = np.sum((y - np.mean(y))**2)      # Total sum of squares
r_squared = 1 - (ss_res / ss_tot)

plt.scatter(x, y, label="Experimental Data", color="blue")   # Scatter plot of original data
plt.plot(x_fit, y_fit, label="Quadratic Fit", color="red")  # Smooth curve of the fitted model
plt.xlabel("x")
plt.ylabel("y")
plt.title("Quadratic Polynomial Fit")
plt.grid(True)
plt.legend()
plt.show()
# ============================
# 4. Display Results
# ============================
print("Coefficients of the fitted polynomial:")
print(f"a = {a:.2f}")
print(f"b = {b:.2f}")
print(f"c = {c:.2f}")

print("\nFitted equation:")
print(f"y = {a:.2f}·x² + {b:.2f}·x + {c:.2f}")

print("\nGoodness of fit:")
print(f"Pearson correlation coefficient (r) = {corr:.3f}")
print(f"Coefficient of determination (R²) = {r_squared:.3f}")