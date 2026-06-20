
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

# --- Data from the notebook ---
# RandomForest Comparison
rf_r2_A = 0.892100
rf_r2_B = 0.777004

# Boosting Models Comparison (No PCA)
mae_A_xgb = 1.890887
mse_A_xgb = 6.909232
r2_A_xgb = 0.905784

mae_A_lgb = 1.942053
mse_A_lgb = 7.996003
r2_A_lgb = 0.890964

mae_A_cat = 1.770570
mse_A_cat = 7.722557
r2_A_cat = 0.894693

# --- Create Directory for Plots ---
desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
plots_dir = os.path.join(desktop_path, "presentation_plots")
os.makedirs(plots_dir, exist_ok=True)

# --- Plot 1: RandomForest with vs. without PCA (R²) ---
plt.figure(figsize=(8, 5))
models_rf = ['RandomForest (No PCA)', 'RandomForest (PCA)']
r2_scores_rf = [rf_r2_A, rf_r2_B]
bars_rf = plt.bar(models_rf, r2_scores_rf, color=['#4A90E2', '#B0A8B9'])
plt.ylabel("R² Score")
plt.title("RandomForest Performance: With vs. Without PCA")
plt.ylim(0.7, 0.95)
for bar in bars_rf:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height, f'{height:.4f}', ha='center', va='bottom')
plot1_path = os.path.join(plots_dir, "rf_pca_comparison.png")
plt.savefig(plot1_path)
plt.close()

# --- Plot 2: Boosting Models Performance (R²) ---
results_boosting = pd.DataFrame({
    'Model': ['XGBoost', 'LightGBM', 'CatBoost'],
    'R²':  [r2_A_xgb, r2_A_lgb, r2_A_cat]
})
results_boosting = results_boosting.sort_values('R²', ascending=False)
colors_r2 = ['#4A90E2', '#50E3C2', '#B8E986']

plt.figure(figsize=(10, 5))
bars_r2 = plt.bar(results_boosting['Model'], results_boosting['R²'], color=colors_r2)
plt.ylabel("R² Score")
plt.title("Boosting Models Performance Comparison (R²)")
plt.ylim(0.88, 0.91)
for bar in bars_r2:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height, f'{height:.4f}', ha='center', va='bottom')
plot2_path = os.path.join(plots_dir, "boosting_r2_comparison.png")
plt.savefig(plot2_path)
plt.close()

# --- Plot 3: Boosting Models Performance (MAE) ---
results_mae = pd.DataFrame({
    'Model': ['XGBoost', 'LightGBM', 'CatBoost'],
    'MAE': [mae_A_xgb, mae_A_lgb, mae_A_cat]
})
results_mae = results_mae.sort_values('MAE', ascending=True)
colors_mae = ['#F5A623', '#F8E71C', '#D0021B']

plt.figure(figsize=(10, 5))
bars_mae = plt.bar(results_mae['Model'], results_mae['MAE'], color=colors_mae)
plt.ylabel("MAE (Lower is better)")
plt.title("Boosting Models Performance Comparison (MAE)")
plt.ylim(1.7, 2.0)
for bar in bars_mae:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height, f'{height:.4f}', ha='center', va='bottom')
plot3_path = os.path.join(plots_dir, "boosting_mae_comparison.png")
plt.savefig(plot3_path)
plt.close()


print(f"Plots saved in: {plots_dir}")
print(f"RF PCA plot: {plot1_path}")
print(f"Boosting R2 plot: {plot2_path}")
print(f"Boosting MAE plot: {plot3_path}")
