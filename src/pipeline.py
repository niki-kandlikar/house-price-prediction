"""
pipeline.py
-----------
End-to-end run: load -> clean -> engineer features -> split -> scale ->
train -> evaluate (MSE / R^2) -> visualize -> save results.

Run with:  python pipeline.py   (from inside src/)
Outputs land in ../results/
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from preprocessing import (
    load_data, handle_missing_values, remove_outliers,
    engineer_features, scale_features, ENGINEERED_FEATURES, TARGET
)
from modeling import split_data, train_model, evaluate_model, coefficient_table

sns.set_style("whitegrid")
RESULTS_DIR = "../results"


def plot_actual_vs_predicted(y_test, preds):
    plt.figure(figsize=(6.5, 6))
    plt.scatter(y_test, preds, alpha=0.5, color="#2E86AB", s=25)
    lims = [min(y_test.min(), preds.min()), max(y_test.max(), preds.max())]
    plt.plot(lims, lims, "--", color="#A23B72", label="Perfect Prediction")
    plt.xlabel("Actual Price ($)")
    plt.ylabel("Predicted Price ($)")
    plt.title("Actual vs. Predicted House Prices")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/actual_vs_predicted.png", dpi=150)
    plt.close()


def plot_residuals(y_test, preds):
    residuals = y_test - preds
    plt.figure(figsize=(7, 4.5))
    sns.histplot(residuals, kde=True, color="#F18F01")
    plt.axvline(0, color="black", linestyle="--", linewidth=1)
    plt.title("Residual Distribution")
    plt.xlabel("Residual (Actual - Predicted)")
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/residuals.png", dpi=150)
    plt.close()


def plot_coefficients(coef_df):
    plt.figure(figsize=(8, 5))
    colors = ["#2E86AB" if c > 0 else "#A23B72" for c in coef_df["Coefficient"]]
    plt.barh(coef_df["Feature"], coef_df["Coefficient"], color=colors)
    plt.title("Standardized Feature Coefficients")
    plt.xlabel("Coefficient (effect on price, scaled features)")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/feature_coefficients.png", dpi=150)
    plt.close()


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # 1. Load & clean
    raw = load_data("../data/housing_data.csv")
    df = handle_missing_values(raw)
    df = remove_outliers(df)

    # 2. Feature engineering
    df = engineer_features(df)

    # 3. Train/test split
    X_train, X_test, y_train, y_test = split_data(df[ENGINEERED_FEATURES], df[TARGET])

    # 4. Feature scaling (fit on train only)
    X_train, X_test, scaler = scale_features(X_train, X_test)

    # 5. Train
    model = train_model(X_train, y_train)

    # 6. Evaluate
    metrics = evaluate_model(model, X_test, y_test)
    print(f"MSE:  {metrics['mse']:,.0f}")
    print(f"RMSE: {metrics['rmse']:,.0f}")
    print(f"R^2:  {metrics['r2']:.3f}")

    pd.DataFrame([{
        "MSE": metrics["mse"], "RMSE": metrics["rmse"], "R2": metrics["r2"]
    }]).to_csv(f"{RESULTS_DIR}/model_metrics.csv", index=False)

    # 7. Coefficients
    coef_df = coefficient_table(model, ENGINEERED_FEATURES)
    coef_df.to_csv(f"{RESULTS_DIR}/feature_coefficients.csv", index=False)

    # 8. Visualize
    plot_actual_vs_predicted(y_test, metrics["predictions"])
    plot_residuals(y_test, metrics["predictions"])
    plot_coefficients(coef_df)

    print(f"\nAll outputs saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
