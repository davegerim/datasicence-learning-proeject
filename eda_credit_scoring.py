"""
================================================================================
Ahadu Bank — Credit Scoring Exploratory Data Analysis (EDA)
================================================================================
Course: AI from Zero to Hero | Data Science Project
Context: Inclusive finance & data-driven lending in Ethiopia (MaaS roadmap)

This script loads the anonymized historical credit dataset (~150k borrowers),
performs structured EDA, saves publication-style figures, and trains baseline
models to quantify which features most strongly predict financial distress
within the next two years (target: SeriousDlqin2yrs).

How to run (from project folder):
    pip install -r requirements.txt
    python eda_credit_scoring.py

Outputs:
    outputs/figures/   — PNG charts for your report or slides
    outputs/eda_summary.txt — Plain-text summary you can paste into a write-up
================================================================================
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# -----------------------------------------------------------------------------
# Configuration — paths and plot style (change only if you move files)
# -----------------------------------------------------------------------------
# Resolve project root so the script works no matter where you invoke it from.
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "cs-training (2).csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"

# Random seed so train/test split and model results are reproducible for grading.
RANDOM_STATE = 42

# Suppress noisy sklearn warnings in console (does not affect results).
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Seaborn theme: readable defaults for reports and projector slides.
sns.set_theme(style="whitegrid", context="notebook", font_scale=1.05)


# -----------------------------------------------------------------------------
# Data dictionary (matches your assignment; kept here so code stays self-explanatory)
# -----------------------------------------------------------------------------
# Unnamed: 0 — Unique borrower row id (we will rename to borrower_id)
# SeriousDlqin2yrs — Target: 1 = serious delinquency / financial distress in 2y
# RevolvingUtilizationOfUnsecuredLines — balance / limit on unsecured lines (>1 = over-limit)
# age — borrower age in years
# NumberOfTime30-59DaysPastDueNotWorse — mild delinquency count (last 2 years)
# DebtRatio — monthly debt payments / monthly gross income
# MonthlyIncome — monthly gross income
# NumberOfOpenCreditLinesAndLoans — count of open trades
# NumberOfTimes90DaysLate — severe delinquency count
# NumberRealEstateLoansOrLines — mortgage / real-estate related lines
# NumberOfTime60-89DaysPastDueNotWorse — moderate delinquency count (last 2 years)
# NumberOfDependents — dependents excluding self (has missing values in raw CSV)


def ensure_output_dirs() -> None:
    """Create output folders if they do not exist (safe to run every time)."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def load_raw_dataframe(csv_path: Path) -> pd.DataFrame:
    """
    Load the training CSV.

    The file uses the literal string 'NA' for missing values (not empty cells).
    We tell pandas to parse those as NaN so downstream statistics are correct.
    """
    if not csv_path.is_file():
        print(f"ERROR: Dataset not found at:\n  {csv_path}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(csv_path, na_values=["NA"])
    # First column is unnamed in the file; rename for clarity in plots and reports.
    df = df.rename(columns={"Unnamed: 0": "borrower_id"})
    return df


def basic_quality_report(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a per-column quality table: dtype, non-null count, missing %, min/max.

    This answers 'what is wrong with the data before we model?' — essential for EDA.
    """
    rows = []
    n = len(df)
    for col in df.columns:
        s = df[col]
        miss = s.isna().sum()
        row = {
            "column": col,
            "dtype": str(s.dtype),
            "non_null": int(s.count()),
            "missing_count": int(miss),
            "missing_pct": round(100.0 * miss / n, 3),
            "n_unique": int(s.nunique(dropna=True)),
        }
        if pd.api.types.is_numeric_dtype(s):
            row["min"] = s.min()
            row["max"] = s.max()
        else:
            row["min"] = None
            row["max"] = None
        rows.append(row)
    return pd.DataFrame(rows)


def impute_for_eda(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create an analysis copy with simple imputations ONLY for visualization continuity.

    Why median for MonthlyIncome: robust to extreme high earners (reduces pull from outliers).
    Why median for NumberOfDependents: discrete counts; median is a stable whole number.

    NOTE: For a production credit model you would document imputation policy with the bank
    (e.g. separate 'missing income' indicator flags). Here we keep EDA plots readable.
    """
    out = df.copy()
    if out["MonthlyIncome"].isna().any():
        out["MonthlyIncome"] = out["MonthlyIncome"].fillna(out["MonthlyIncome"].median())
    if out["NumberOfDependents"].isna().any():
        out["NumberOfDependents"] = out["NumberOfDependents"].fillna(
            out["NumberOfDependents"].median()
        )
    return out


def plot_target_distribution(df: pd.DataFrame) -> None:
    """Bar chart of the binary target — shows class imbalance (common in credit data)."""
    counts = df["SeriousDlqin2yrs"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(x=counts.index.astype(str), y=counts.values, ax=ax, palette="Set2")
    ax.set_xlabel("SeriousDlqin2yrs (0 = no distress, 1 = distress in 2y)")
    ax.set_ylabel("Number of borrowers")
    ax.set_title("Target distribution — class imbalance check")
    for i, v in enumerate(counts.values):
        ax.text(i, v + 500, f"{v:,}", ha="center", fontsize=10)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "01_target_distribution.png", dpi=150)
    plt.close(fig)


def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    """
    Pearson correlation among numeric features + target.

    Interpreting heatmaps: dark red/blue = linear association. Credit features often
    correlate with each other (e.g. delinquency buckets), but multicollinearity is
    handled later via regularization or tree models.
    """
    num_cols = [
        c
        for c in df.columns
        if c not in ("borrower_id",) and pd.api.types.is_numeric_dtype(df[c])
    ]
    corr = df[num_cols].corr()
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(
        corr,
        annot=False,
        cmap="RdBu_r",
        center=0,
        square=True,
        ax=ax,
        cbar_kws={"shrink": 0.7},
    )
    ax.set_title("Correlation heatmap (numeric features + target)")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "02_correlation_heatmap.png", dpi=150)
    plt.close(fig)


def cap_series_for_plot(x: pd.Series, upper_quantile: float = 0.995) -> pd.Series:
    """
    Winsorize extreme upper tail for plotting only (so histograms are readable).

    Raw utilization and debtRatio can contain extreme outliers that squash the x-axis.
    """
    hi = x.quantile(upper_quantile)
    return x.clip(upper=hi)


def plot_key_distributions(df: pd.DataFrame) -> None:
    """Histograms for the main risk drivers mentioned in credit scoring literature."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    util = cap_series_for_plot(df["RevolvingUtilizationOfUnsecuredLines"])
    axes[0, 0].hist(util, bins=60, color="steelblue", edgecolor="white", alpha=0.85)
    axes[0, 0].set_title("Revolving utilization (capped at 99.5th pct for display)")
    axes[0, 0].set_xlabel("Utilization ratio")

    axes[0, 1].hist(df["age"], bins=40, color="teal", edgecolor="white", alpha=0.85)
    axes[0, 1].set_title("Age")
    axes[0, 1].set_xlabel("Years")

    dr = cap_series_for_plot(df["DebtRatio"], 0.995)
    axes[1, 0].hist(dr, bins=60, color="coral", edgecolor="white", alpha=0.85)
    axes[1, 0].set_title("Debt ratio (capped at 99.5th pct for display)")
    axes[1, 0].set_xlabel("Debt / income")

    mi = cap_series_for_plot(df["MonthlyIncome"], 0.995)
    axes[1, 1].hist(mi, bins=60, color="mediumpurple", edgecolor="white", alpha=0.85)
    axes[1, 1].set_title("Monthly income (imputed; capped for display)")
    axes[1, 1].set_xlabel("Income")

    plt.suptitle("Univariate distributions — core financial variables", y=1.02, fontsize=13)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "03_key_distributions.png", dpi=150)
    plt.close(fig)


def default_rate_by_bucket(
    df: pd.DataFrame, col: str, bins: int | list, labels: list[str] | None = None
) -> pd.DataFrame:
    """
    Compute default rate (mean of target) within each bin of `col`.

    This is the heart of EDA for credit: monotonic risk patterns across delinquency
    history and utilization support policy trust and model sanity checks.
    """
    s = df[col]
    if isinstance(bins, int):
        cats = pd.qcut(s, q=bins, duplicates="drop")
    else:
        cats = pd.cut(s, bins=bins, labels=labels, include_lowest=True)
    g = df.assign(_bin=cats).groupby("_bin", observed=True)["SeriousDlqin2yrs"]
    out = g.agg(n="count", default_rate="mean").reset_index()
    return out


def plot_delinquency_risk(df: pd.DataFrame) -> None:
    """
    Line/bar style view: default rate vs delinquency severity counts.

    Expected pattern: higher counts of 30–59, 60–89, and 90+ late events → higher default rate.
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    for ax, col, title in zip(
        axes,
        [
            "NumberOfTime30-59DaysPastDueNotWorse",
            "NumberOfTime60-89DaysPastDueNotWorse",
            "NumberOfTimes90DaysLate",
        ],
        ["30–59 days past due (count)", "60–89 days past due (count)", "90+ days late (count)"],
    ):
        # Aggregate default rate at each integer count (capped for readability).
        tmp = (
            df.groupby(col, observed=True)["SeriousDlqin2yrs"]
            .agg(n="count", dr="mean")
            .reset_index()
        )
        tmp = tmp[tmp[col] <= 10]  # cap x-axis; rare extreme counts still visible in tables
        sns.barplot(data=tmp, x=col, y="dr", ax=ax, color="darkred", alpha=0.85)
        ax.set_title(title)
        ax.set_ylabel("Default rate (mean target)")
        ax.set_xlabel("Count")

    plt.suptitle("Delinquency history vs observed default rate", y=1.05, fontsize=13)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "04_delinquency_vs_default_rate.png", dpi=150)
    plt.close(fig)


def plot_utilization_vs_default(df: pd.DataFrame) -> None:
    """Binned utilization: shows non-linear risk — very high utilization predicts stress."""
    x = df["RevolvingUtilizationOfUnsecuredLines"]
    # Quantile bins on capped utilization for stable buckets
    x_cap = cap_series_for_plot(x, 0.995)
    df_plot = df.assign(_u=x_cap)
    df_plot["_bin"] = pd.qcut(df_plot["_u"], q=10, duplicates="drop")
    agg = df_plot.groupby("_bin", observed=True)["SeriousDlqin2yrs"].mean().reset_index()

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(range(len(agg)), agg["SeriousDlqin2yrs"], marker="o", color="navy")
    ax.set_xticks(range(len(agg)))
    ax.set_xticklabels([str(i) for i in range(len(agg))])
    ax.set_xlabel("Utilization decile (low → high)")
    ax.set_ylabel("Default rate")
    ax.set_title("Default rate by revolving utilization decile (upper tail capped for binning)")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "05_utilization_deciles_default_rate.png", dpi=150)
    plt.close(fig)


def numeric_feature_columns(df: pd.DataFrame) -> list[str]:
    """Numeric feature list excluding id and target."""
    return [
        c
        for c in df.columns
        if c not in ("borrower_id", "SeriousDlqin2yrs") and pd.api.types.is_numeric_dtype(df[c])
    ]


def univariate_statistics_report(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build univariate statistics per numeric feature.

    Includes quartiles, skewness, and kurtosis requested in the assignment.
    """
    rows = []
    for col in numeric_feature_columns(df):
        s = df[col].dropna()
        q1 = s.quantile(0.25)
        q2 = s.quantile(0.50)
        q3 = s.quantile(0.75)
        rows.append(
            {
                "feature": col,
                "count": int(s.shape[0]),
                "mean": float(s.mean()),
                "std": float(s.std(ddof=1)),
                "min": float(s.min()),
                "q1": float(q1),
                "median_q2": float(q2),
                "q3": float(q3),
                "max": float(s.max()),
                "iqr": float(q3 - q1),
                "skewness": float(s.skew()),
                "kurtosis_excess": float(s.kurt()),
            }
        )
    return pd.DataFrame(rows).sort_values("feature").reset_index(drop=True)


def outlier_iqr_report(df: pd.DataFrame) -> pd.DataFrame:
    """Detect outliers using 1.5*IQR rule for each numeric feature."""
    rows = []
    n = len(df)
    for col in numeric_feature_columns(df):
        s = df[col].dropna()
        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3 - q1
        lo = q1 - 1.5 * iqr
        hi = q3 + 1.5 * iqr
        outlier_mask = (s < lo) | (s > hi)
        outlier_count = int(outlier_mask.sum())
        rows.append(
            {
                "feature": col,
                "lower_bound": float(lo),
                "upper_bound": float(hi),
                "outlier_count": outlier_count,
                "outlier_pct_of_rows": round(100.0 * outlier_count / n, 4),
            }
        )
    return pd.DataFrame(rows).sort_values("outlier_pct_of_rows", ascending=False).reset_index(
        drop=True
    )


def hypothesis_testing_report(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run hypothesis tests between target classes:
    - Welch's t-test for difference in means per numeric feature.
    """
    y0 = df["SeriousDlqin2yrs"] == 0
    y1 = df["SeriousDlqin2yrs"] == 1

    rows = []
    for col in numeric_feature_columns(df):
        s0 = df.loc[y0, col].dropna()
        s1 = df.loc[y1, col].dropna()
        t_stat, p_value = stats.ttest_ind(s1, s0, equal_var=False, nan_policy="omit")
        rows.append(
            {
                "feature": col,
                "mean_target_1": float(s1.mean()),
                "mean_target_0": float(s0.mean()),
                "mean_diff_1_minus_0": float(s1.mean() - s0.mean()),
                "t_statistic": float(t_stat),
                "p_value": float(p_value),
                "significant_at_0_05": bool(p_value < 0.05),
            }
        )
    return pd.DataFrame(rows).sort_values("p_value").reset_index(drop=True)


def plot_bivariate_relationships(df: pd.DataFrame) -> None:
    """Bivariate boxplots: key numeric features by target class."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    pairs = [
        ("RevolvingUtilizationOfUnsecuredLines", "Utilization by target"),
        ("DebtRatio", "Debt ratio by target"),
        ("MonthlyIncome", "Monthly income by target"),
        ("NumberOfTimes90DaysLate", "90+ days late count by target"),
    ]
    for ax, (col, title) in zip(axes.flat, pairs):
        plot_df = df[["SeriousDlqin2yrs", col]].copy()
        plot_df[col] = cap_series_for_plot(plot_df[col], 0.995)
        sns.boxplot(
            data=plot_df,
            x="SeriousDlqin2yrs",
            y=col,
            ax=ax,
            palette="Set2",
            showfliers=False,
        )
        ax.set_title(title)
        ax.set_xlabel("Target (0=no distress, 1=distress)")
    plt.suptitle("Bivariate analysis — distributions by target class", y=1.02, fontsize=13)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "07_bivariate_boxplots_by_target.png", dpi=150)
    plt.close(fig)


def build_feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Separate id/target from predictors used in sklearn.

    We drop borrower_id — it is not a legitimate predictive feature (random row order).
    """
    y = df["SeriousDlqin2yrs"]
    feature_cols = [c for c in df.columns if c not in ("borrower_id", "SeriousDlqin2yrs")]
    X = df[feature_cols].copy()
    return X, y


def train_baseline_models(X: pd.DataFrame, y: pd.Series) -> dict:
    """
    Train two industry-standard baselines:

    1) Logistic regression with imputation + scaling — interpretable coefficients.
    2) Random Forest — captures non-linearities & interactions; strong out-of-the-box AUC.

    We use stratified split to preserve default rate in train and test.
    For imbalanced targets, ROC-AUC is often more informative than raw accuracy.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
    )

    # All features are numeric; median impute then scale for logistic regression.
    numeric_features = X_train.columns.tolist()
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            )
        ]
    )

    log_reg = Pipeline(
        steps=[
            ("prep", preprocessor),
            (
                "model",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    solver="lbfgs",
                ),
            ),
        ]
    )

    log_reg.fit(X_train, y_train)
    y_proba_lr = log_reg.predict_proba(X_test)[:, 1]
    y_pred_lr = log_reg.predict(X_test)

    rf = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=200,
                    max_depth=12,
                    random_state=RANDOM_STATE,
                    class_weight="balanced_subsample",
                    n_jobs=-1,
                ),
            ),
        ]
    )
    rf.fit(X_train, y_train)
    y_proba_rf = rf.predict_proba(X_test)[:, 1]
    y_pred_rf = rf.predict(X_test)

    results = {
        "logistic_regression": {
            "roc_auc": roc_auc_score(y_test, y_proba_lr),
            "accuracy": accuracy_score(y_test, y_pred_lr),
            "confusion_matrix": confusion_matrix(y_test, y_pred_lr),
            "classification_report": classification_report(y_test, y_pred_lr, digits=4),
            "model": log_reg,
            "feature_names": numeric_features,
        },
        "random_forest": {
            "roc_auc": roc_auc_score(y_test, y_proba_rf),
            "accuracy": accuracy_score(y_test, y_pred_rf),
            "confusion_matrix": confusion_matrix(y_test, y_pred_rf),
            "classification_report": classification_report(y_test, y_pred_rf, digits=4),
            "model": rf,
            "feature_names": numeric_features,
        },
    }
    return results


def plot_feature_importance(rf_pipeline: Pipeline, feature_names: list[str]) -> None:
    """Mean decrease in impurity — relative ranking of risk drivers (tree model view)."""
    rf_model = rf_pipeline.named_steps["model"]
    importances = rf_model.feature_importances_
    order = np.argsort(importances)[::-1]
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.barplot(
        x=importances[order],
        y=[feature_names[i] for i in order],
        ax=ax,
        color="seagreen",
    )
    ax.set_title("Random Forest — feature importance (Gini importance)")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "06_feature_importance_random_forest.png", dpi=150)
    plt.close(fig)


def write_summary_text(
    df: pd.DataFrame,
    quality: pd.DataFrame,
    model_results: dict,
    univariate_stats: pd.DataFrame,
    outlier_stats: pd.DataFrame,
    hypothesis_stats: pd.DataFrame,
) -> None:
    """Plain-text summary file for your written report (copy/paste friendly)."""
    lines = []
    lines.append("Ahadu Bank — Credit Scoring EDA Summary")
    lines.append("=" * 60)
    lines.append(f"Rows: {len(df):,} | Columns: {df.shape[1]}")
    pos_rate = df["SeriousDlqin2yrs"].mean()
    lines.append(f"Overall default / distress rate (target mean): {pos_rate:.4%}")
    lines.append("")
    lines.append("Missing values (from raw data, before imputation for plots):")
    miss = quality[quality["missing_count"] > 0][["column", "missing_count", "missing_pct"]]
    if miss.empty:
        lines.append("  None detected.")
    else:
        for _, r in miss.iterrows():
            lines.append(f"  {r['column']}: {int(r['missing_count'])} ({r['missing_pct']}%)")
    lines.append("")
    lines.append("Baseline model metrics (hold-out test set, stratified split):")
    for name, m in model_results.items():
        lines.append(f"  {name}:")
        lines.append(f"    ROC-AUC: {m['roc_auc']:.4f}")
        lines.append(f"    Accuracy: {m['accuracy']:.4f}")
    lines.append("")
    lines.append("Interpretation tips for your report:")
    lines.append(
        "  - ROC-AUC summarizes ranking quality of predicted risk scores (0.5 = random)."
    )
    lines.append(
        "  - Accuracy can look 'high' when defaults are rare; always pair with AUC + confusion matrix."
    )
    lines.append("")
    lines.append("Univariate analysis (with quartiles, skewness, kurtosis):")
    for _, r in univariate_stats.head(5).iterrows():
        lines.append(
            f"  {r['feature']}: Q1={r['q1']:.3f}, Median={r['median_q2']:.3f}, "
            f"Q3={r['q3']:.3f}, Skew={r['skewness']:.3f}, Kurtosis={r['kurtosis_excess']:.3f}"
        )
    lines.append("")
    lines.append("Outlier detection (IQR method, top 5 by outlier %):")
    for _, r in outlier_stats.head(5).iterrows():
        lines.append(
            f"  {r['feature']}: {int(r['outlier_count'])} outliers "
            f"({r['outlier_pct_of_rows']:.2f}% of rows)"
        )
    lines.append("")
    lines.append("Hypothesis testing (Welch t-test, top 5 smallest p-values):")
    for _, r in hypothesis_stats.head(5).iterrows():
        lines.append(
            f"  {r['feature']}: p-value={r['p_value']:.3e}, "
            f"mean_diff(1-0)={r['mean_diff_1_minus_0']:.4f}"
        )
    lines.append("")
    lines.append("Figures saved under outputs/figures/ (PNG).")
    lines.append("Additional tables saved under outputs/:")
    lines.append("  - univariate_statistics.csv")
    lines.append("  - outlier_detection_iqr.csv")
    lines.append("  - hypothesis_testing_results.csv")

    out_path = OUTPUT_DIR / "eda_summary.txt"
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """Run the full pipeline end-to-end."""
    ensure_output_dirs()

    print("Loading data...")
    raw = load_raw_dataframe(DATA_PATH)

    print("Computing data quality table...")
    quality = basic_quality_report(raw)
    quality_path = OUTPUT_DIR / "data_quality_table.csv"
    quality.to_csv(quality_path, index=False)
    print(f"  Saved: {quality_path}")

    # EDA copy with simple imputations for continuous visuals
    df_eda = impute_for_eda(raw)

    print("Plotting distributions and relationships...")
    plot_target_distribution(df_eda)
    plot_correlation_heatmap(df_eda)
    plot_key_distributions(df_eda)
    plot_delinquency_risk(df_eda)
    plot_utilization_vs_default(df_eda)
    plot_bivariate_relationships(df_eda)

    print("Computing univariate, quartile, skewness, kurtosis, outlier, and hypothesis reports...")
    univariate_stats = univariate_statistics_report(df_eda)
    univariate_stats.to_csv(OUTPUT_DIR / "univariate_statistics.csv", index=False)
    outlier_stats = outlier_iqr_report(df_eda)
    outlier_stats.to_csv(OUTPUT_DIR / "outlier_detection_iqr.csv", index=False)
    hypothesis_stats = hypothesis_testing_report(df_eda)
    hypothesis_stats.to_csv(OUTPUT_DIR / "hypothesis_testing_results.csv", index=False)

    print("Training baseline models (this may take ~1–2 minutes on a laptop)...")
    X, y = build_feature_matrix(raw)  # use raw + pipeline imputation for modeling
    model_results = train_baseline_models(X, y)

    plot_feature_importance(
        model_results["random_forest"]["model"],
        model_results["random_forest"]["feature_names"],
    )

    write_summary_text(
        raw,
        quality,
        model_results,
        univariate_stats,
        outlier_stats,
        hypothesis_stats,
    )

    print("\n--- Model results (test set) ---")
    for name, m in model_results.items():
        print(f"\n{name}:")
        print(f"  ROC-AUC: {m['roc_auc']:.4f}")
        print(f"  Accuracy: {m['accuracy']:.4f}")
        print("  Confusion matrix [[TN FP],[FN TP]]:")
        print(" ", m["confusion_matrix"])

    print(f"\nSummary written to: {OUTPUT_DIR / 'eda_summary.txt'}")
    print(f"Figures written to: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
