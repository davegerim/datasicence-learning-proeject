# Credit Scoring — EDA, Statistical Analysis & Baseline Models

**Project report and presentation script**  
Course: *AI from Zero to Hero* (Data Science) — **Ahadu Bank / inclusive finance** (Ethiopia), data-driven credit assessment on an anonymized historical credit dataset (~**150,000** borrowers).

This README is your **single handout** from **the original course brief** through **everything implemented so far**, including the **statistical extension** your teacher asked for and the **questions you asked** while building the project. You can read sections aloud in class.

**Related files:** place the course brief `dataScience_project_updated (2).pdf` in the project folder if you have it; this document restates its requirements in text so you are not dependent on the PDF being in the repo.

---

## Table of contents

1. [Official course brief (from the assignment PDF)](#1-official-course-brief-from-the-assignment-pdf)  
2. [The core question and the project’s answer](#2-the-core-question-and-the-projects-answer)  
3. [How the work evolved (from the start until now)](#3-how-the-work-evolved-from-the-start-until-now)  
4. [What the code does (pipeline in order)](#4-what-the-code-does-pipeline-in-order)  
5. [Dataset and every column](#5-dataset-and-every-column)  
6. [Output files and figures](#6-output-files-and-figures)  
7. [Teacher’s seven topics: skewness, kurtosis, univariate, bivariate, quartiles, outliers, hypothesis tests](#7-teachers-seven-topics)  
8. [Data quality table: `missing_pct` and `n_unique`](#8-data-quality-table-missing_pct-and-n_unique)  
9. [Class imbalance (and which figure shows it)](#9-class-imbalance-and-which-figure-shows-it)  
10. [ROC-AUC in plain language](#10-roc-auc-in-plain-language)  
11. [Baseline models and metrics](#11-baseline-models-and-metrics)  
12. [What is usually “next” after this project](#12-what-is-usually-next-after-this-project)  
13. [Beginner glossary (features, target, heatmap, outliers, …)](#13-beginner-glossary)  
14. [Log of questions from your chats (and short answers)](#14-log-of-questions-from-your-chats-and-short-answers)  
15. [How to run](#15-how-to-run)  
16. [License and data use](#16-license-and-data-use)  

---

## 1. Official course brief (from the assignment PDF)

The course project is titled **“AI from Zero to Hero — Data Science Project”** and is framed for **Ahadu Bank**.

**Overview (what the brief asks for):**

- Perform **Exploratory Data Analysis (EDA)** on the **historical credit dataset** provided for **credit scoring**.
- Support **inclusive finance** and **data-driven lending** in **Ethiopia** by finding **key risk factors** that predict the chance a borrower will experience **financial distress within the next two years**.
- Use the insights to inform a future **high-performing credit scoring model** and its packaging as a scalable **Model-as-a-Service (MaaS) API** (a long-term product idea: a service that returns a score from borrower inputs).

**Dataset (as in the brief):**

- Anonymized historical data for **150,000 borrowers** (training set for the challenge).
- Column summary in the official text:
  - **Unnamed: 0** — unique borrower ID (row index in the file).
  - **SeriousDlqin2yrs** — **target**: 1 if the borrower experienced financial distress in the **next 2 years**, 0 otherwise.
  - **RevolvingUtilizationOfUnsecuredLines** — total balance on unsecured credit lines **÷** total credit limits; values **> 1** indicate over-utilization.
  - **age** — age of the borrower.
  - **NumberOfTime30-59DaysPastDueNotWorse** — number of times 30–59 days past due (not worse) in the last 2 years.
  - **DebtRatio** — monthly debt payments **÷** monthly **gross** income.
  - **MonthlyIncome** — monthly **gross** income.
  - **NumberOfOpenCreditLinesAndLoans** — number of open credit lines and loans.
  - **NumberOfTimes90DaysLate** — number of times **90+ days** late.
  - **NumberRealEstateLoansOrLines** — number of mortgage / real-estate loans (including home equity lines).
  - **NumberOfTime60-89DaysPastDueNotWorse** — 60–89 days past due (not worse) in the last 2 years.
  - **NumberOfDependents** — dependents, excluding the borrower.

The brief’s author and submission line (as given in the course materials) include: **Tadele Melesse** and a submission target around **18 April 2026** (your teacher’s schedule may vary).

**In this repository:** the CSV is named `cs-training (2).csv`. The first column is renamed in code to **`borrower_id`** for clarity; all other column names match the story above.

---

## 2. The core question and the project’s answer

**What the “question” really is (one paragraph):**  
Use **past** borrower data to do serious **EDA** (quality, missingness, patterns), identify **drivers of serious delinquency / financial distress in the next two years** (`SeriousDlqin2yrs`), and connect that evidence to **credit scoring** and the **vision** of a **data-driven, inclusive** lending process (and, in the long term, a **MaaS**-style service). The practical question is: *What does the data show about who is at risk, and can we back that with charts, tables, and baseline models?*

**What the “answer” is in this project:**

1. **Analytical (EDA):** The data has about **150k** rows, a **minority** positive class (roughly **~6–7%** “distress” — see `outputs/eda_summary.txt` after a run), and **notable missingness** (especially **MonthlyIncome**, and some **NumberOfDependents**). Plots show that **worse delinquency history** and **high utilization** line up with **higher** observed distress rates, which matches credit intuition.
2. **Inferential (later phase):** **Univariate** tables, **IQR** outliers, **bivariate** plots, and **Welch t-tests** add statistical structure beyond eyeballing charts.
3. **Predictive (baseline):** A **stratified** train/test split and two baselines — **logistic regression** and **random forest** — with **ROC-AUC**, **accuracy**, and a **confusion matrix**, plus a random forest **importance** chart, show that the patterns are **learnable** (not a proof of causality).

---

## 3. How the work evolved (from the start until now)

| Phase | What you aimed for | What you have in the repo |
|--------|-------------------|----------------------------|
| **Start (course + dataset)** | Implement the brief: load `cs-training (2).csv`, EDA, figures, data quality, baselines, readable comments. | `eda_credit_scoring.py` loads the CSV, treats string **`NA`** as missing, renames the ID column, writes **`data_quality_table.csv`**, builds **`df_eda`** with **median** fill for EDA for income/dependents, saves figures **01–06**, trains **logistic + random forest**, writes **`eda_summary.txt`**. |
| **Teacher extension** | Add **Skewness, Univariate, Bivariate, Quartiles, Outlier detection, Hypothesis testing, Kurtosis** without breaking the rest. | Same script now also writes **`univariate_statistics.csv`**, **`outlier_detection_iqr.csv`**, **`hypothesis_testing_results.csv`**, and figure **`07_bivariate_boxplots_by_target.png`**; `eda_summary.txt` includes **top lines** from those; **`scipy`** in `requirements.txt` for **Welch t-test**. |
| **Documentation** | A README you can read for presentation and a report. | This file. |

*There is no separate “feature engineering” write-up in this document; the main script works from the **raw** predictive columns in the CSV (ID dropped), plus the statistics layers above.*

---

## 4. What the code does (pipeline in order)

1. **Load** `cs-training (2).csv` with `na_values=["NA"]` and rename **`Unnamed: 0` → `borrower_id`**.  
2. **Data quality** → `outputs/data_quality_table.csv` (dtype, missing count/**missing_pct**, **n_unique**, min/max for numerics).  
3. **EDA copy `df_eda`:** median imputation for **MonthlyIncome** and **NumberOfDependents** (for smooth plots and tables; **not** a substitute for a full production policy).  
4. **Figures 01–05:** target distribution, correlation heatmap, key histograms, delinquency vs default rate, utilization deciles.  
5. **Figure 07:** bivariate boxplots by **target** for selected features.  
6. **Univariate + quartiles + skew + kurtosis** → `univariate_statistics.csv`.  
7. **IQR outliers** → `outlier_detection_iqr.csv`.  
8. **Welch t-tests (target 1 vs 0)** → `hypothesis_testing_results.csv`.  
9. **Modeling** on `raw` with sklearn **median** impute inside pipelines: **logistic regression** (scaled) and **random forest**; **stratified** 75/25 split.  
10. **Figure 06:** random forest feature importance.  
11. **Text** → `eda_summary.txt`.  

**Main program file:** `eda_credit_scoring.py` (all logic and comments are there; click through in your editor for line-level detail).

---

## 5. Dataset and every column

**File:** `cs-training (2).csv`  
**Typical size:** ~150,000 **rows** × **12** **columns** after renaming.

| Column | Role | Plain-language meaning |
|--------|------|------------------------|
| `borrower_id` | ID (was `Unnamed: 0`) | **Row / borrower** identifier. **Not** used for modeling. |
| `SeriousDlqin2yrs` | **Target (label)** | **1** = serious delinquency / **financial distress within 2 years**; **0** = not. This is what you **predict**. |
| `RevolvingUtilizationOfUnsecuredLines` | Feature | **Balance / limit** on **unsecured** revolving lines; **&gt; 1** can mean over the limit. |
| `age` | Feature | Age in **years**. |
| `NumberOfTime30-59DaysPastDueNotWorse` | Feature | Count of **30–59** DPD (not worse), last **2 years**. |
| `NumberOfTime60-89DaysPastDueNotWorse` | Feature | Count of **60–89** DPD, last **2 years**. |
| `NumberOfTimes90DaysLate` | Feature | Count of **90+** days late (severe). |
| `DebtRatio` | Feature | **Monthly debt payments** / **monthly gross income**; can be extreme if income is tiny or missing. |
| `MonthlyIncome` | Feature | **Gross** monthly income; has **missing** in raw data. |
| `NumberOfOpenCreditLinesAndLoans` | Feature | How many open lines/loans. |
| `NumberRealEstateLoansOrLines` | Feature | **Mortgage** / real-estate–related line count. |
| `NumberOfDependents` | Feature | **Dependents** (excluding self); can be **missing**. |

**Vocabulary:** **Feature** = input column used for learning (everything except the ID; **target** = `SeriousDlqin2yrs`).

---

## 6. Output files and figures

**Tables / text (under `outputs/`):**

| File | What it is |
|------|------------|
| `data_quality_table.csv` | Per column: type, **missing_pct**, **n_unique**, min/max, etc. |
| `univariate_statistics.csv` | Univariate: mean, std, min, Q1, median, Q3, max, IQR, **skewness**, **kurtosis_excess**. |
| `outlier_detection_iqr.csv` | **IQR fences** and outlier **counts** / % of rows per feature. |
| `hypothesis_testing_results.csv` | **Welch t-test** per feature: class means, p-values, column **`significant_at_0_05`**. |
| `eda_summary.txt` | Short **narrative**: rows, default rate, missingness, model AUC/accuracy, **and** top lines from the new blocks. |

**Figures (under `outputs/figures/`):**

| File | What to say in one line |
|------|-------------------------|
| `01_target_distribution.png` | **Class imbalance**: most rows are `0`, fewer are `1` — see [§9](#9-class-imbalance-and-which-figure-shows-it). |
| `02_correlation_heatmap.png` | **Linear (Pearson)** associations between numeric **features** and **target** — a **heatmap** of correlation strengths. |
| `03_key_distributions.png` | **Univariate** histograms; some series are **capped** at a high **quantile** so the x-axis is readable. |
| `04_delinquency_vs_default_rate.png` | **Default rate** (mean **target**) vs **count** of past-due events — risk rises with worse history. |
| `05_utilization_deciles_default_rate.png` | **Deciles** of (capped) **utilization** vs **default** rate — non-linear risk. |
| `07_bivariate_boxplots_by_target.png` | **Bivariate**: key features vs **0/1** **target** (boxplots). |
| `06_feature_importance_random_forest.png` | **Random forest** “importance” (MDI) — which features the trees split on most **in this model** (not causal law). |

---

## 7. Teacher’s seven topics

- **Univariate** — one variable at a time; see `univariate_statistics.csv` and `03_*.png`.  
- **Quartiles (Q1, median, Q3)** — in `univariate_statistics.csv` as `q1`, `median_q2`, `q3`.  
- **Skewness** — `skewness` column: near **0** = more symmetric; **large positive** = long **right** tail.  
- **Kurtosis** — `kurtosis_excess` (Fisher / excess kurtosis): near **0** ≈ normal-like tails; **very large** ≈ many **extremes** / heavy tails.  
- **Bivariate (feature vs target)** — `07_*.png`, and risk curves in `04_` / `05_`.  
- **Outlier detection (IQR)** — `outlier_detection_iqr.csv` using **Q1 − 1.5×IQR** and **Q3 + 1.5×IQR** fences.  
- **Hypothesis testing** — `hypothesis_testing_results.csv`: **Welch’s t-test** of **mean(feature | target=1)** vs **mean(feature | target=0)**. **p &lt; 0.05** often read as “unlikely the mean difference is **pure luck** at 5% level”; **p ≥ 0.05** = weak evidence of a **mean** gap (nonlinear relations may still exist).

**Short “viva” line for all seven:** *“I summarized each feature alone (univariate, quartiles, skew, kurtosis), detected extremes with IQR, compared classes visually (bivariate) and with t-tests, then kept baselines to see if the signal is learnable.”*

---

## 8. Data quality table: `missing_pct` and `n_unique`

- **`missing_pct`:** percent of **rows** where that column is **empty** (missing / `NaN` after read). For example, **~19.8%** for **MonthlyIncome** means that share of rows has no income in the raw file.  
- **`n_unique`:** how many **different** non-missing values appear. **IDs** ≈ one per row; **target** is usually **2** levels (0/1). For **age**, you see how many **distinct** ages; for some count columns, it reflects **how many** distinct count levels show up.

---

## 9. Class imbalance and which figure shows it

- **Class imbalance** means one **class** of the **target** (usually **`0`**) is **much** more common than the other (`1`). In credit, **defaults/distress** are often **rare** vs **non-events**.  
- The figure that **shows** this is **`01_target_distribution.png`**: the **taller** bar (often **0**) vs the **shorter** bar (**1**).  
- Because of this, you rely on **ROC-AUC** and the **confusion matrix**, not **accuracy** alone (see [§10](#10-roc-auc-in-plain-language) and the tips inside `eda_summary.txt`).

---

## 10. ROC-AUC in plain language

- **ROC-AUC** = **A**rea **U**nder the **ROC** **C**urve; often you just say **AUC**. It ranges roughly from **0.5** (random) to **1.0** (best possible ranking in ideal data).  
- In practice for credit, it measures how well the model **ranks** applicants by risk: if you take one random **distress** and one **non-distress** row, the **probability** the model’s score is higher for the **distress** case equals the AUC (equivalent interpretation in binary problems).  
- It does **not** pick the **operating** cutoff for “approve/decline”; banks set policy separately. It also does **not** by itself fix **fairness** or **profit** — it summarizes **ranking** quality on the test split.

---

## 11. Baseline models and metrics

- **Logistic regression** — a classic **yes/no** model; **interpretable** direction of effects; uses **median impute**, **scaling**, and **`class_weight="balanced"`** to reduce ignoring the **minority** class.  
- **Random forest** — many **decision trees**; captures **nonlinear** and **interaction** effects better in many problems; **median** impute + **`class_weight="balanced_subsample"`**; **`06_*.png`** = **feature importance** (MDI).  
- **Train/test split** — **25%** test, **75%** train, with **`stratify=y`** so both sets keep similar **default** rates.  
- **Metrics in `eda_summary.txt` and console** — **ROC-AUC** and **accuracy**; the script also prints a **confusion matrix** (TN, FP, FN, TP) for what kinds of **errors** happen.

**Confusion matrix (2×2):** counts of **true negative**, **false positive**, **false negative**, **true positive** — helps you **talk** about *approving a bad customer* vs *rejecting a good one* in policy terms.

---

## 12. What is usually “next” after this project

- **For most courses:** **finish the written report / slides** and be ready to **defend** EDA + **baselines** and **limitations** (missing income, **imbalance**, IQR on **count** variables, **mean** tests not capturing all shapes).  
- **If the course asks for MaaS / API:** only **after** the **modeling story** is agreed — e.g. wrap a **frozen** model + imputation in a small **HTTP API** (e.g. FastAPI/Flask) with a clear **JSON** schema. The PDF often names **MaaS** as a **future** product **direction**, not always as a mandatory first deliverable.  
- **Possible academic next steps (not in this script unless you add them):** cross-validation, **calibration** curves, other algorithms (XGBoost/LightGBM), or **fairness** analysis across groups — all **beyond** the current baseline EDA.

---

## 13. Beginner glossary

| Term | Short meaning |
|------|----------------|
| **EDA (Exploratory Data Analysis)** | Systematically exploring data: missingness, tables, charts, sanity checks **before** strong conclusions. |
| **Feature** | An **input** column used in modeling (not the row ID; not the **target**). |
| **Target (label, dependent variable)** | The **outcome** you want to learn: `SeriousDlqin2yrs` (**0/1**). |
| **Heatmap (here)** | A **colored grid**; in `02_*.png` it is a **correlation** matrix — **darker** colors = stronger **linear** correlation. |
| **Outlier** | A value very far from the **bulk** of the data; “far” is **defined** by a rule (here, **IQR** fences) — not always a **mistake** in credit (can be a **very risky** borrower). |
| **Imputation** | Filling **missing** values. The script uses **medians** for EDA; sklearn uses **median** imputation **inside** model pipelines. |
| **Winsorize / cap (for plots)** | Temporarily **limit** very large values so **histograms** and **boxplots** are **readable** — the script caps some series at a high **percentile** for **display** only. |
| **Stratified split** | When splitting train/test, keep the **same proportion** of **0/1** in both parts (so evaluation is **fairer** if one class is rare). |
| **Logistic regression** | A model for **binary** outcomes; outputs a **probability**-like score after a sigmoid. |
| **Random forest** | Many **decision trees** on **random** subsets of data/features; “votes” or **averages** to predict **risk**; can capture **nonlinear** structure. |
| **Decision tree (intuition)** | A chain of **if–else** rules (e.g. “if late count &gt; k …”). |
| **Feature importance (in RF)** | A **heuristic** rank of which inputs drove **splits** most; great for **discussion**, not automatic **regulatory** proof. |
| **Accuracy** | **Fraction** of correct **0/1** **labels**; can look “high” when the **common** class dominates — pair with AUC. |

---

## 14. Log of questions from your chats (and short answers)

*This mirrors what you asked in earlier working sessions, so you can say “I documented these questions in the report.”*

| You asked | Short answer (where to look) |
|------------|-------------------------------|
| *Implement the project for `cs-training (2).csv`, impress the teacher, explain what and where* | A full pipeline in **`eda_credit_scoring.py`**: load **NA** correctly, **quality** CSV, **plots**, **logistic** + **RF**, **`eda_summary.txt`**. |
| *Deep / line-by-line code explanation* | Open **`eda_credit_scoring.py`**: docstring → config → load → quality → EDA impute → each `plot_*` → models → `write_summary_text` → `main()`. |
| *Explain the assignment in beginner words (PDF + dataset section)* | [§1](#1-official-course-brief-from-the-assignment-pdf) and [§2](#2-the-core-question-and-the-projects-answer) above. |
| *What is the overall question, what is the answer?* | [§2](#2-the-core-question-and-the-projects-answer). |
| *What is ROC-AUC?* | [§10](#10-roc-auc-in-plain-language). |
| *What are `missing_pct` and `n_unique`?* | [§8](#8-data-quality-table-missing_pct-and-n_unique). |
| *What is left? Is the next step the API?* | [§12](#12-what-is-usually-next-after-this-project) — **report** first; **MaaS/API** only if the **rubric** asks; often it is a **vision** in the brief. |
| *What is class imbalance; which graph?* | [§9](#9-class-imbalance-and-which-figure-shows-it) — **`01_target_distribution.png`**. |
| *Teacher asked: skewness, univariate, bivariate, quartiles, outliers, hypothesis, kurtosis* | [§3](#3-how-the-work-evolved-from-the-start-until-now), [§6–7](#6-output-files-and-figures) — CSV/figures listed. |
| *Explain each term: where, which data, how outputs found* | [§3–4](#3-how-the-work-evolved-from-the-start-until-now), [§5–6](#5-dataset-and-every-column). |
| *Brief viva: large/small skew, kurtosis, p-values* | [§7](#7-teachers-seven-topics) (skew/kurt) and p-value bullets under **hypothesis testing** there. |
| *Update README; include everything; **no** feature engineering section* | This file — **no** feature-engineering chapter; the script uses **raw** feature columns (plus the **stated** EDA median fill for plots). |
| *Create a README and push to GitHub* | This repo’s README; **publishing** steps depend on your account — use your teacher’s or your own **remote** URL; add a **`.gitignore`**, `git add`, `commit`, `push` (not repeated here to avoid wrong credentials/URLs on your machine). |

---

## 15. How to run

**Python 3.10+** recommended.

```bash
pip install -r requirements.txt
python eda_credit_scoring.py
```

**Dependencies:** see `requirements.txt` — `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, **`scipy`**.

**After running**, open **`outputs/eda_summary.txt`** for the **latest** numbers and narrative.

---

## 16. License and data use

**Educational use** for the course. Using the **dataset** must follow the **license / terms** of the source from which you obtained the file.

---

*End of report-style README. Re-run `python eda_credit_scoring.py` before your presentation to refresh all numeric outputs.*
