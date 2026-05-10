"""
Risk Score Modeling
--------------------
Compares Linear Regression vs XGBoost for Medicare RAF score prediction.
Includes SHAP explainability, feature importance, and model evaluation.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
import os
import pickle

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import shap

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────
DATA_PATH    = "data/synthetic/medicare_members.csv"
MODEL_DIR    = "outputs/models"
SHAP_DIR     = "outputs/shap"
REPORT_DIR   = "outputs/reports"

for d in [MODEL_DIR, SHAP_DIR, REPORT_DIR]:
    os.makedirs(d, exist_ok=True)

# ── 1. Load & Feature Engineering ─────────────────────────────────────────
def load_and_engineer(path: str) -> tuple:
    df = pd.read_csv(path)

    # Encode categoricals
    le_gender = LabelEncoder()
    le_region = LabelEncoder()
    df["gender_enc"]  = le_gender.fit_transform(df["gender"])
    df["region_enc"]  = le_region.fit_transform(df["region"])

    # HCC-based features
    df["hcc_count"]         = df["current_hcc_list"].apply(
        lambda x: len(str(x).split("|")) if pd.notna(x) and x != "" else 0)
    df["has_diabetes"]      = df["current_icd_codes"].str.contains("E11", na=False).astype(int)
    df["has_chf"]           = df["current_icd_codes"].str.contains("I50", na=False).astype(int)
    df["has_ckd"]           = df["current_icd_codes"].str.contains("N18", na=False).astype(int)
    df["has_copd"]          = df["current_icd_codes"].str.contains("J44", na=False).astype(int)
    df["has_cancer"]        = df["current_icd_codes"].str.contains("C[0-9]", na=False, regex=True).astype(int)
    df["has_mental_health"] = df["current_icd_codes"].str.contains("F[0-9]", na=False, regex=True).astype(int)

    # Interaction features
    df["diabetes_ckd"]      = df["has_diabetes"] * df["has_ckd"]
    df["chf_ckd"]           = df["has_chf"] * df["has_ckd"]
    df["diabetes_chf"]      = df["has_diabetes"] * df["has_chf"]

    # Age bands
    df["age_band"] = pd.cut(df["age"],
                            bins=[64, 69, 74, 79, 84, 99],
                            labels=[0, 1, 2, 3, 4]).astype(int)

    FEATURES = [
        "age", "age_band", "gender_enc", "dual_eligible", "region_enc",
        "hcc_count",
        "has_diabetes", "has_chf", "has_ckd", "has_copd",
        "has_cancer", "has_mental_health",
        "diabetes_ckd", "chf_ckd", "diabetes_chf"
    ]

    FEATURE_LABELS = {
        "age":              "Age",
        "age_band":         "Age Band",
        "gender_enc":       "Gender",
        "dual_eligible":    "Dual Eligible",
        "region_enc":       "Region",
        "hcc_count":        "HCC Count",
        "has_diabetes":     "Diabetes",
        "has_chf":          "Congestive Heart Failure",
        "has_ckd":          "Chronic Kidney Disease",
        "has_copd":         "COPD",
        "has_cancer":       "Cancer",
        "has_mental_health":"Mental Health Condition",
        "diabetes_ckd":     "Diabetes × CKD",
        "chf_ckd":          "CHF × CKD",
        "diabetes_chf":     "Diabetes × CHF",
    }

    X = df[FEATURES]
    y = df["current_raf_score"]

    return df, X, y, FEATURES, FEATURE_LABELS


# ── 2. Train Models ────────────────────────────────────────────────────────
def train_models(X_train, y_train):
    models = {}

    # Linear Regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    models["Linear Regression"] = lr

    # Ridge Regression
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train, y_train)
    models["Ridge Regression"] = ridge

    # XGBoost
    xgb_model = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        verbosity=0
    )
    xgb_model.fit(X_train, y_train,
                  eval_set=[(X_train, y_train)],
                  verbose=False)
    models["XGBoost"] = xgb_model

    return models


# ── 3. Evaluate Models ─────────────────────────────────────────────────────
def evaluate_models(models, X_train, X_test, y_train, y_test) -> pd.DataFrame:
    results = []
    for name, model in models.items():
        y_pred_test  = model.predict(X_test)
        y_pred_train = model.predict(X_train)

        rmse  = np.sqrt(mean_squared_error(y_test, y_pred_test))
        mae   = mean_absolute_error(y_test, y_pred_test)
        r2    = r2_score(y_test, y_pred_test)
        r2_tr = r2_score(y_train, y_pred_train)

        results.append({
            "Model":        name,
            "R² (Test)":    round(r2, 4),
            "R² (Train)":   round(r2_tr, 4),
            "RMSE":         round(rmse, 4),
            "MAE":          round(mae, 4),
            "Overfit Gap":  round(r2_tr - r2, 4)
        })

    return pd.DataFrame(results)


# ── 4. SHAP Explainability ─────────────────────────────────────────────────
def run_shap(model, X_test, feature_labels: dict):
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    return explainer, shap_values


def plot_shap_summary(shap_values, X_test, feature_labels: dict, save_path: str):
    X_display = X_test.rename(columns=feature_labels)
    shap_display = shap_values.copy()

    plt.figure(figsize=(10, 7))
    shap.summary_plot(shap_display, X_display, show=False, plot_size=None)
    plt.title("SHAP Feature Importance — XGBoost RAF Score Model",
              fontsize=13, fontweight="bold", pad=15)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ SHAP summary plot saved → {save_path}")


def plot_shap_waterfall_sample(explainer, shap_values, X_test,
                                feature_labels: dict, sample_idx: int,
                                save_path: str):
    X_display = X_test.rename(columns=feature_labels)
    shap_exp = shap.Explanation(
        values=shap_values[sample_idx],
        base_values=explainer.expected_value,
        data=X_display.iloc[sample_idx],
        feature_names=list(X_display.columns)
    )
    plt.figure(figsize=(10, 6))
    shap.waterfall_plot(shap_exp, show=False, max_display=12)
    plt.title(f"SHAP Waterfall — Member #{sample_idx} RAF Score Breakdown",
              fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ SHAP waterfall saved → {save_path}")


# ── 5. Comparison Plot ─────────────────────────────────────────────────────
def plot_model_comparison(results_df: pd.DataFrame, save_path: str):
    colors = {"Linear Regression": "#4C72B0",
              "Ridge Regression":  "#55A868",
              "XGBoost":           "#C44E52"}

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle("Model Comparison: Linear Regression vs Ridge vs XGBoost",
                 fontsize=14, fontweight="bold", y=1.02)

    metrics = [("R² (Test)", "R² Score (higher = better)", True),
               ("RMSE",      "RMSE (lower = better)",      False),
               ("MAE",       "MAE (lower = better)",       False)]

    for ax, (metric, ylabel, higher_better) in zip(axes, metrics):
        bars = ax.bar(results_df["Model"],
                      results_df[metric],
                      color=[colors[m] for m in results_df["Model"]],
                      edgecolor="white", linewidth=1.5, width=0.5)
        ax.set_title(metric, fontsize=12, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_xticklabels(results_df["Model"], rotation=15, ha="right", fontsize=9)

        for bar, val in zip(bars, results_df[metric]):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.002,
                    f"{val:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

        best_idx = results_df[metric].idxmax() if higher_better else results_df[metric].idxmin()
        bars[best_idx].set_edgecolor("gold")
        bars[best_idx].set_linewidth(3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ Model comparison plot saved → {save_path}")


# ── 6. Feature Importance Plot ─────────────────────────────────────────────
def plot_feature_importance(model, feature_names: list,
                             feature_labels: dict, save_path: str):
    importance = model.feature_importances_
    feat_imp = pd.DataFrame({
        "Feature": [feature_labels.get(f, f) for f in feature_names],
        "Importance": importance
    }).sort_values("Importance", ascending=True)

    fig, ax = plt.subplots(figsize=(9, 7))
    colors = ["#C44E52" if imp > feat_imp["Importance"].quantile(0.75) else "#4C72B0"
              for imp in feat_imp["Importance"]]
    bars = ax.barh(feat_imp["Feature"], feat_imp["Importance"],
                   color=colors, edgecolor="white", linewidth=0.8)

    ax.set_xlabel("Feature Importance Score", fontsize=11)
    ax.set_title("XGBoost Feature Importance\nMedicare RAF Score Prediction",
                 fontsize=13, fontweight="bold")

    red_patch   = mpatches.Patch(color="#C44E52", label="Top Drivers (>75th pct)")
    blue_patch  = mpatches.Patch(color="#4C72B0", label="Supporting Features")
    ax.legend(handles=[red_patch, blue_patch], loc="lower right", fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ Feature importance plot saved → {save_path}")


# ── 7. Save Model & Report ─────────────────────────────────────────────────
def save_model(model, path: str):
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"  ✅ XGBoost model saved → {path}")


def save_report(results_df: pd.DataFrame, path: str):
    results_df.to_csv(path, index=False)
    print(f"  ✅ Model comparison report saved → {path}")


# ── MAIN ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*55)
    print("  MEDICARE RAF SCORE — MODEL TRAINING & EVALUATION")
    print("="*55)

    # Load data
    print("\n[1/6] Loading and engineering features...")
    df, X, y, FEATURES, FEATURE_LABELS = load_and_engineer(DATA_PATH)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    print(f"  Train: {len(X_train):,} | Test: {len(X_test):,}")

    # Train
    print("\n[2/6] Training models...")
    models = train_models(X_train, y_train)
    for name in models:
        print(f"  ✅ {name} trained")

    # Evaluate
    print("\n[3/6] Evaluating models...")
    results_df = evaluate_models(models, X_train, X_test, y_train, y_test)
    print("\n" + results_df.to_string(index=False))
    save_report(results_df, f"{REPORT_DIR}/model_comparison.csv")

    # Plots
    print("\n[4/6] Generating comparison plots...")
    plot_model_comparison(results_df, f"{SHAP_DIR}/model_comparison.png")

    # Feature importance
    print("\n[5/6] Generating feature importance...")
    plot_feature_importance(models["XGBoost"], FEATURES, FEATURE_LABELS,
                            f"{SHAP_DIR}/feature_importance.png")

    # SHAP
    print("\n[6/6] Running SHAP explainability...")
    explainer, shap_values = run_shap(models["XGBoost"], X_test, FEATURE_LABELS)
    plot_shap_summary(shap_values, X_test, FEATURE_LABELS,
                      f"{SHAP_DIR}/shap_summary.png")
    plot_shap_waterfall_sample(explainer, shap_values, X_test,
                                FEATURE_LABELS, sample_idx=5,
                                save_path=f"{SHAP_DIR}/shap_waterfall_sample.png")

    # Save model
    save_model(models["XGBoost"], f"{MODEL_DIR}/xgboost_raf_model.pkl")

    print("\n" + "="*55)
    print("  MODELING COMPLETE")
    print("="*55)
    best = results_df.loc[results_df["R² (Test)"].idxmax()]
    print(f"\n  🏆 Best Model : {best['Model']}")
    print(f"     R² (Test)  : {best['R² (Test)']:.4f}")
    print(f"     RMSE       : {best['RMSE']:.4f}")
    print(f"     MAE        : {best['MAE']:.4f}")
    print(f"     Overfit Gap: {best['Overfit Gap']:.4f}")
