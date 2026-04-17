# Credit scoring — EDA & baseline models

Learning project for **AI from Zero to Hero** (data science track). It explores an anonymized historical credit dataset (~150k borrowers) with exploratory analysis and simple predictive baselines for serious delinquency within two years (`SeriousDlqin2yrs`).

The narrative is framed around inclusive finance and data-driven lending (Ahadu Bank context in the course materials).

## What’s in this repo

| Item | Description |
|------|-------------|
| `eda_credit_scoring.py` | End-to-end pipeline: load data, quality report, plots, logistic regression + random forest baselines |
| `cs-training (2).csv` | Training data (Kaggle-style credit scoring features) |
| `requirements.txt` | Python dependencies |
| `outputs/` | Generated figures, `eda_summary.txt`, and `data_quality_table.csv` after you run the script |
| `dataScience_project_updated (2).pdf` | Course / assignment brief (if present) |

## Setup

Python 3.10+ recommended.

```bash
pip install -r requirements.txt
```

## Run

From the project directory:

```bash
python eda_credit_scoring.py
```

The script creates `outputs/figures/` if needed and writes PNG charts (target distribution, correlation heatmap, distributions, delinquency vs default rate, utilization deciles, random forest feature importance) plus `outputs/eda_summary.txt` and `outputs/data_quality_table.csv`.

## Models

- **Logistic regression** — median imputation, standardization, `class_weight="balanced"` for the imbalanced target.
- **Random forest** — median imputation, `class_weight="balanced_subsample"`, with feature importance plot.

Metrics on a stratified hold-out set include ROC-AUC and accuracy (see `eda_summary.txt` after a run).

## Dependencies

See `requirements.txt`: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`.

## License

Educational use. Dataset usage follows the terms of the original source you obtained it from.
