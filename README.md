# Credit scoring EDA — outputs guide

Run: `pip install -r requirements.txt` then `python eda_credit_scoring.py`.  
All paths below are under `outputs/`.

**Dataset:** `cs-training (2).csv` — 150,000 rows; target `SeriousDlqin2yrs` (1 = financial distress in 2 years). **Overall default rate in this file: ~6.68%.**

---

## 1. Figures (`outputs/figures/`)

Each image answers a visual question. **What the data shows** (from the same run that produced the tables below) is summarized under each.

### `01_target_distribution.png`
- **What it is:** Count of borrowers with target **0** vs **1** (bar chart).
- **What the data shows:** The **1** class is much smaller than **0** — about **6.7%** distress vs **~93.3%** no distress. This is **class imbalance**: a model can look “accurate” by mostly predicting 0, so you must use **ROC-AUC** (and the confusion matrix), not only accuracy.

### `02_correlation_heatmap.png`
- **What it is:** **Pearson correlation** between all numeric features and the target (and each other).
- **What the data shows:** Delinquency count variables **move together** (mild / moderate / severe late-payment fields are **positively** correlated with each other). The heatmap is about **linear** association; a weak cell does not mean “no risk,” only that a straight-line relationship is weak.

### `03_key_distributions.png`
- **What it is:** **Histograms** of utilization, age, debt ratio, and monthly income (for income/utilization/debt, very high values are **capped** in the plot so the graph is readable).
- **What the data shows:** **Age** is roughly **bell-shaped** (typical working-age borrowers). **Utilization, debt ratio, and income** are **heavily right-skewed** — most values sit in a moderate range, and a long tail of **extreme** values pulls means far above the median (see the univariate file for that).

### `04_delinquency_vs_default_rate.png`
- **What it is:** For each integer count of past-due events (30–59, 60–89, 90+ days, **x** capped for display), the **default rate** = share of `SeriousDlqin2yrs = 1` in that group.
- **What the data shows:** **Higher past-due counts go with higher default rate** — the monotonic “worse history → more distress” story you expect in credit. That supports using delinquency features in a score.

### `05_utilization_deciles_default_rate.png`
- **What it is:** Borrowers are sorted into **10 utilization buckets** (deciles) after capping the upper tail; for each bucket the plot shows the **mean target** (default rate).
- **What the data shows:** **Higher revolving utilization** generally lines up with **higher** distress rate in the higher buckets (non-linear risk: not always a single straight line across the full range, but a clear “stress at the top” pattern).

### `07_bivariate_boxplots_by_target.png`
- **What it is:** **Boxplots** of key numeric variables **split** by **target** (0 = no distress, 1 = distress); some y-axes are capped and outliers may be hidden.
- **What the data shows:** **Defaulters (1)** and **non-defaulters (0)** differ in **center and spread** on these variables. For example, t-tests (below) find **defaulters younger on average** and **higher** on delinquency counts. Income is **lower on average** in the distress group. Use the plot to **see** separation; use `hypothesis_testing_results.csv` to see which mean differences are **statistically** significant.

### `06_feature_importance_random_forest.png`
- **What it is:** **Random forest** “importance” (how much each feature was used in tree splits, MDI).
- **What the data shows:** A **ranking** of which raw inputs the forest relied on most for *this* model and random seed. It is **not** proof of business causality, but it tells you what the algorithm treated as most informative in the baseline.

---

## 2. Seven topics (what they mean + what *this* data shows)

| Topic | Meaning (one line) | What we see in this project |
|--------|-------------------|----------------------------|
| **Univariate** | One number column summarized alone (no pairing yet). | File `univariate_statistics.csv`. Most borrowers have **0** in the 30/60/90 delinquency counts at Q1/median; **income** middle 50% is about **$3,903–$7,400**; **age** Q1/median/Q3 about **41 / 52 / 63** years. |
| **Quartiles (Q1, median, Q3)** | Q1 = 25th percentile, **median** = 50th, Q3 = 75th. Half the data lies between Q1 and Q3 (**IQR = Q3 − Q1**). | Example — **MonthlyIncome** (after EDA imputation in the table): Q1 = **3903**, median = **5400**, Q3 = **7400**. That means: **half** of borrowers (the middle) earn between about **$3,903 and $7,400** (the IQR band). The **max** in the data is much larger than Q3, which is why the **mean** (6418) is a poor “typical” value compared to the **median** (5400). For **30–59 DPD**: Q1 = median = Q3 = **0** — the **typical** borrower has **no** such event; a minority drives all the risk signal. |
| **Skewness** | Asymmetry of a histogram. Positive and large = long **right** tail (a few very large values). | **age** is nearly symmetric (skew **~0.19**). **Debt ratio** and **income** have **huge** positive skew (90+ and 120+): a small share of people have **extreme** values. That matches the histograms. |
| **Kurtosis (excess)** | “Tail weight” vs a normal curve. **Large** positive = more extreme values / heavier tails. | **age** has kurtosis near **0** (mild tails). **Debt ratio / income** have **very large** kurtosis (thousands) — the data has **rare, extreme** magnitudes, not a tidy bell curve. |
| **Bivariate** | Two variables together; here, **feature vs target** (0 vs 1). | Figures **04, 05, 07** and the heatmap **02**. Visually, risk drivers **separate** the two groups, especially delinquency and age. |
| **Outlier detection (IQR rule)** | Flag values **below** Q1 − 1.5×IQR or **above** Q3 + 1.5×IQR. | File `outlier_detection_iqr.csv`. **Debt ratio** has the most flagged rows by this rule (**~20.9%** of all borrowers) — the ratio has a **huge** upper tail. **30–59 DPD** has **~16%** “outliers” mostly because the lower half is stuck at 0, so the fence is tight. **Age** has **almost no** IQR outliers (**&lt;0.04%**) — ages stay in a plausible band. |
| **Hypothesis testing (Welch t-test)** | Tests if the **mean** of a feature differs between **target = 1** and **target = 0** (p-value small ⇒ strong evidence of a **mean** difference). | File `hypothesis_testing_results.csv`. **Significant (p &lt; 0.05)**: e.g. **age** (distress group **~6.8 years younger** on average), all three **delinquency** count fields (much higher means in the distress group), **lower** income, etc. **Not** significant (mean test): **revolving utilization** (p **~0.22**) — the **average** is not very different between groups with this rule, even though decile plots (figure 05) can still show risk by bucket (non-mean / nonlinear effects). |

Re-run the script to refresh all numbers; small changes in metrics are normal.

---

## 3. Output files (CSVs and summary text)

### `data_quality_table.csv`
- **What it is:** One row per column: `dtype`, `non_null`, `missing_count`, **`missing_pct`**, `n_unique`, and for numerics `min`/`max`.
- **What the data shows:** **Monthly income** is missing in **~19.8%** of raw rows; **NumberOfDependents** in **~2.6%**. No other column uses the literal `NA` at that scale. That is important for any story about “income is unknown for many applicants.”

### `univariate_statistics.csv`
- **What it is:** For each **numeric** feature: `count`, `mean`, `std`, `min`, **`q1`**, **`median_q2`**, **`q3`**, `max`, **`iqr`**, **`skewness`**, **`kurtosis_excess`**.
- **What the data shows (how to read one row):** e.g. **age** — mean **52.3** years, IQR **22** (from 41 to 63) → wide middle 50% around mid-life. e.g. **Revolving utilization** — median **~0.15** (15% of limit used) but max **fifty thousands**-scale in raw numbers → **median** is the right “typical” story, not the mean. **Quartile columns** are the concrete **splitting points** of the sorted data (not a model output).

### `outlier_detection_iqr.csv`
- **What it is:** For each feature, **lower/upper IQR fence**, `outlier_count`, **`outlier_pct_of_rows`**.
- **What the data shows:** Which fields are “extreme” under a **rule-based** definition. High **%** for **debt** and **delinquency** reflects **real heavy tails and zeros**, not only errors.

### `hypothesis_testing_results.csv`
- **What it is:** `mean_target_1`, `mean_target_0`, `mean_diff_1_minus_0`, `t_statistic`, `p_value`, `significant_at_0_05`.
- **What the data shows:** A ranked list of which variables have **statistically different means** between the **distress** and **non-distress** groups. Use for report language like: on average, distressed borrowers are younger and show more past delinquency.

### `eda_summary.txt`
- **What it is:** Short run log: N rows, default %, **missing** lines, model **ROC-AUC** and **accuracy**, a **few** lines from univariate / outliers / hypothesis blocks.
- **What the data shows:** A **one-page** snapshot; open the **CSV** files for full detail.

### Figures
- All discussion above — section **1**.

---

## 4. Baseline models (numbers from last `eda_summary.txt` run)

- **Logistic regression** — ROC-AUC **~0.85**, accuracy **~0.79** (illustrative; re-run to refresh).  
- **Random forest** — ROC-AUC **~0.86**, accuracy **~0.85**.  

**Why mention them:** they show the **patterns in sections 1–3** are not only visual; a standard learner can use them to rank risk on a held-out 25% test split.

---

*Re-run `python eda_credit_scoring.py` before a presentation to align figures, tables, and `eda_summary.txt` to one fresh run.*
