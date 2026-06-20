
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

# --- Data from the notebook ---
# RandomForest Comparison
mae_A = 2.041078
mse_A = 7.912745
r2_A = 0.892100

mae_B = 2.338637
mse_B = 16.353140
r2_B = 0.777004

# --- Create Directory for Plots ---
desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
plots_dir = os.path.join(desktop_path, "presentation_plots")
os.makedirs(plots_dir, exist_ok=True)

# --- Plot 1: RandomForest with vs. without PCA (MAE) ---
plt.figure(figsize=(8, 6))
models_rf = ['RandomForest (No PCA)', 'RandomForest (PCA)']
mae_scores_rf = [mae_A, mae_B]
bars_mae = plt.bar(models_rf, mae_scores_rf, color=['#F5A623', '#D0021B'])
plt.ylabel("MAE (Lower is better)")
plt.title("RandomForest MAE: With vs. Without PCA")
plt.ylim(2.0, 2.4)
for bar in bars_mae:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height, f'{height:.4f}', ha='center', va='bottom')
plot_mae_path = os.path.join(plots_dir, "rf_mae_comparison.png")
plt.savefig(plot_mae_path)
plt.close()

# --- Plot 2: RandomForest with vs. without PCA (MSE) ---
plt.figure(figsize=(8, 6))
mse_scores_rf = [mse_A, mse_B]
bars_mse = plt.bar(models_rf, mse_scores_rf, color=['#BD10E0', '#4A90E2'])
plt.ylabel("MSE (Lower is better)")
plt.title("RandomForest MSE: With vs. Without PCA")
plt.ylim(7, 17)
for bar in bars_mse:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height, f'{height:.4f}', ha='center', va='bottom')
plot_mse_path = os.path.join(plots_dir, "rf_mse_comparison.png")
plt.savefig(plot_mse_path)
plt.close()

print(f"New plots saved in: {plots_dir}")
print(f"MAE comparison plot: {plot_mae_path}")
print(f"MSE comparison plot: {plot_mse_path}")
