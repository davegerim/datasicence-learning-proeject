# Credit scoring EDA — beginner guide + outputs

**Run:** `pip install -r requirements.txt` then `python eda_credit_scoring.py`.  
**Data file:** `cs-training (2).csv` (150,000 rows). **Outputs** live under `outputs/`.

---

## Part 1 — What the assignment is asking (the “question”), in plain language

Your teacher gave you a **project brief**. You can read the whole project as answering:

> **“Using past borrower data, how can a bank understand who is more likely to get into serious money trouble soon, and how can that support fairer, smarter lending?”**

Here is what each part of the brief usually means (wording in your PDF may differ; the **ideas** are the same).

### Title and framing (“AI from Zero to Hero”, “Data Science Project”)

- **What it means:** This is a **data science** assignment. You work with **realistic** data, **explore** it, and **draw conclusions**—not only theory on paper.

### “Ahadu Bank”

- **What it means:** The story is set at a **fictional (example) bank** (Ahadu). You write as if you are **helping that bank** make better, evidence-based decisions.

### “Exploratory Data Analysis (EDA)”

- **EDA** = **looking at the data carefully** before (or while) you build models.
- You check: Is anything **missing**? Do numbers look **weird**? What **patterns** do you see? Which things **go together** with **bad outcomes** (defaults / distress)?
- **Analogy:** Before a strong diagnosis, a doctor collects **vitals and history**. EDA is like that for your **dataset**.

### “Historical credit dataset … Credit Scoring”

- **Historical** = from the **past** (old loans / old borrowers), not a crystal ball with no data.
- **Credit scoring** = giving each borrower a **risk score** (how **risky** it is to lend to them, often later turned into a probability of default / distress).

### “Inclusive finance & data-driven lending … Ethiopia”

- **Inclusive finance** = trying to lend in a way that does **not exclude** people **unfairly**, while still **managing risk** (safety and fairness in tension).
- **Data-driven lending** = decisions that are **supported by data and models**, not only gut feeling.
- **Ethiopia** = **regional context** in the course story. Your **methods** (plots, models) are universal; the **“why it matters”** is local to the brief.

### “Identifying key risk factors … financial distress within the next two years”

- **Risk factors** = things that show up **more often** among people who later have **trouble paying** (e.g. many past late payments, very high debt compared to income).
- **Financial distress** in this dataset = the **target** column: **`SeriousDlqin2yrs`**. It is **1** if the borrower had that **bad outcome in the next 2 years**, **0** if not. That is the **outcome the bank worries about** in the brief.
- So the project question includes: **Which past signals in the data go with that bad outcome?**

### “Insights will inform … high-performing credit scoring model … Model-as-a-Service (MaaS) API”

- **Credit scoring model** = a **rule** (or a learned formula) that turns **borrower features** (age, income, late payments, etc.) into a **score** or **probability** of the bad event.
- **MaaS (Model-as-a-Service)** = putting that model **behind an API** so other software can send **borrower fields** and get a **score** back (like a small web service). For many EDA projects you **do not** have to build the API unless the teacher requires it; the brief often describes a **long-term product vision** (future bank system).

### Submission date

- **What it means:** The **due date** for the assignment.

---

## Part 2 — The “Dataset” section of the question, explained piece by piece

### “150,000 borrowers (training set)”

- **150,000 rows** ≈ **150,000 borrowers** (in this file, **one row per borrower** after cleaning).
- **Training set** = the table your **model learns from** in a challenge. In your script we also use a **train / test split** (part of the rows is **hidden** from training) to check performance fairly.

### “Anonymized”

- **What it means:** IDs are **not real private names** in a public file; the data is set up to **protect privacy** (still treat it responsibly).

### Column-by-column (normal words)

| Name in the CSV | Simple meaning |
|-----------------|----------------|
| **Unnamed: 0** | Just a **row / borrower ID** column in the file. In code it is renamed to **`borrower_id`**. It is **not** a “credit” signal; we **do not** use it as a model feature. |
| **SeriousDlqin2yrs** | **Target (label)** = the **thing you want to learn to predict**: **1** = serious delinquency / **financial distress in the next 2 years**, **0** = not. |
| **RevolvingUtilizationOfUnsecuredLines** | **How “maxed out”** the person is on unsecured revolving credit: **balance ÷ credit limit**. **High** often means stress. **> 1** can mean **over the limit** or data quirks. |
| **age** | Borrower’s **age in years**. |
| **NumberOfTime30-59DaysPastDueNotWorse** | **Count** of times they were **30–59 days late** (and “not worse” in that window) in the last **2 years** (mild / early stress). |
| **DebtRatio** | Roughly **monthly debt payments ÷ monthly gross income**. **High** = heavy **burden** (watch for very large values if income is tiny or missing). |
| **MonthlyIncome** | **Gross** income **per month**. (This column has a lot of **missing** in the raw file.) |
| **NumberOfOpenCreditLinesAndLoans** | How many **open** credit lines and loans. |
| **NumberOfTimes90DaysLate** | **Count** of **90+ days late** — a **strong** “already in trouble” signal. |
| **NumberRealEstateLoansOrLines** | **Mortgage / home / real-estate** related lines (e.g. housing exposure). |
| **NumberOfTime60-89DaysPastDueNotWorse** | **Count** of **60–89 days late** episodes (worse than 30–59, not yet 90+ bucket). |
| **NumberOfDependents** | People who **depend** on the borrower (e.g. children). Can relate to **expenses**. (Has some **missing** in the raw file.) |

**Beginner idea:**

- **Features** = **inputs** to the model (all the columns we use to predict, **except** the row ID and **except** the **target**).
- **Target** = **output to predict** → **`SeriousDlqin2yrs` (0/1)**.

**Typical run:** **~6.7%** of rows have **target = 1** in this file (class imbalance: many more **0** than **1**).

---

## Part 3 — What the implementation does (simple steps)

### Load the CSV

- The file uses the text **`NA`** for some missing values. The code tells **pandas** to read those as **empty** (NaN) so **math and counts** are correct.

### Rename the ID column

- The first column becomes **`borrower_id`** so plots and reports are **easy to read**.

### Data quality report

- **Saved as** `outputs/data_quality_table.csv`.
- Counts **missing** data, shows **types**, **n_unique** (how many different values), **min/max** for numbers, etc.  
- This answers: **“Is the data complete enough to trust, and where are the gaps?”**  
- In this data, **MonthlyIncome** and **NumberOfDependents** have noticeable **missing %** in the raw file.

### Imputation (only for smooth EDA)

- A copy of the data used for many **plots and tables** fills some missing values (e.g. **median** for income / dependents) so **histograms** and **group comparisons** are not full of empty holes. The **modeling** step uses its **own** median imputation **inside** sklearn (train-safe).

### EDA plots (pictures) — part A

- **01 Target distribution** — how many **0** vs **1** (class imbalance: fewer “bad” cases).
- **02 Correlation heatmap** — colored grid of **linear (Pearson) links** between numeric fields (not causation, **association**).
- **03 Histograms** — utilization, age, debt ratio, income (very large values are sometimes **capped** for **display** only so the x-axis is readable).
- **04 Delinquency vs default rate** — for **counts** of past-due events, the **default rate** (share of 1s) in each count group. Sanity check: **worse history → more distress** in the data.
- **05 Utilization deciles** — people split into **10** utilization bands; default rate in each band (see **non-linear** risk in higher use).

### EDA and statistics (added for the later assignment topics) — part B

- **Univariate + quartiles + skewness + kurtosis** → `outputs/univariate_statistics.csv` (one row per feature: center, spread, **Q1 / median / Q3**, IQR, skew, excess kurtosis).  
- **Outlier detection (IQR rule)** → `outputs/outlier_detection_iqr.csv` (fences, counts, % of rows).  
- **Hypothesis tests (Welch t-test on means, target 1 vs 0)** → `outputs/hypothesis_testing_results.csv`.  
- **Bivariate** figure → `07_bivariate_boxplots_by_target.png` (boxplots of key **features** by **0/1** target).  

These use **`scipy`** in `requirements.txt` for the **t-test**.

### Baseline models (simple “score like” check)

- The code **splits** rows: about **75%** **train** / **25%** **test** (**stratified** so the share of 1s stays similar in both parts).
- **Logistic regression** and **random forest** (see glossary).
- **Reports** **ROC-AUC**, **accuracy**, and prints a **confusion matrix** in the console.

### Feature importance (Random Forest)

- **Saved as** `06_feature_importance_random_forest.png`  
- A bar chart: which **features** the forest used most in its **splits** (a **storytelling** view, not automatic legal “causality”).

### Summary text

- **Saved as** `outputs/eda_summary.txt`  
- **Pulls** key **numbers** (N rows, default %, **missing** lines, model **metrics**, a **few** lines from the new tables) so you can paste into a **report**.

---

## Part 4 — Glossary: technical words (general)

### Data & columns

| Term | Meaning |
|------|--------|
| **Dataset** | A table: many **rows** (borrowers) and **columns** (variables). |
| **Feature** | One **input** column used in prediction (age, income, late counts, etc.). **Not** usually the **row ID**. |
| **Target (label, dependent variable)** | The **thing you predict** → here `SeriousDlqin2yrs` (**0/1**). |
| **EDA (Exploratory Data Analysis)** | **Exploring** and **visualizing** the data to understand it before you trust strong claims. |
| **Missing values** | **Empty** cells. In this project, **income** and **dependents** are **often** missing in the raw file. |
| **Imputation** | **Filling** missing values (here: often **median** for simple EDA continuity; models use imputation **inside** sklearn). In real banks, **policy** matters. |
| **Outlier** | A value **very far** from most people. Your script can **label** them with the **IQR** rule. **Capping** in some plots = only for **charts**, to see the **main** pattern. |

### Charts

| Term | Meaning |
|------|--------|
| **Histogram** | Bars of **how often** values fall in ranges; shows the **shape** of a variable. |
| **Heatmap (here: correlation heatmap)** | A **colored grid**: each cell is **how two** numeric columns **linearly co-move** (correlation). **Not** the same as **causation**—just **association** in this data. |
| **Class imbalance** | One **class** of the **target** (e.g. **0**) is **much** more common than the other (**1**). Default is **rare** here, so **accuracy** alone can **mislead**; **ROC-AUC** helps. |

### Modeling

| Term | Meaning |
|------|--------|
| **Model** | A learned **mapping** from **features** → **prediction** (class or **probability** of distress). |
| **Train / test split** | **Train** = learn; **Test** = check on **other** rows (no cheating on the same rows you trained on). |
| **Stratified split** | Keep roughly the **same** share of **0/1** in **train** and **test** (fairer with rare **1**s). |
| **Logistic regression** | A **classic 0/1** model. Learns **weights** per feature; **interpretable** baseline. |
| **Random forest** | **Many** **decision trees** combined; can handle **curved** and **interactive** patterns. Often **stronger AUC**, less **one-line** interpretable. |
| **Decision tree (idea)** | A chain of **if–else** rules (e.g. *if* late count high *then* higher risk branch). |
| **Probability output** | A number **0 to 1**: model’s **estimated chance** of the **bad** class. |

### Metrics

| Term | Meaning |
|------|--------|
| **Accuracy** | **Fraction** of rows where the predicted class **0/1** matches the truth. Can look “high” when **0s dominate**. |
| **ROC-AUC** | A **0–1** style score: how well the model **ranks** risky people **above** safe people ( **0.5** ≈ random, **1.0** = perfect **ranking** in ideal data). Common when classes are **imbalanced**. |
| **Confusion matrix** | **2×2** table: true negatives, false positives, false negatives, true positives — which **kinds of mistakes** happen. |
| **Feature importance (Random Forest)** | A **rank** of which inputs drove **splits** most. Good for **discussion**; not automatic **regulatory** proof. |

### Combined table: seven “statistics / EDA” topics

| Topic | In plain language | In this project | What this data shows |
|--------|-------------------|-----------------|----------------------|
| **Univariate** | You study **one** column at a time (typical value, spread, **shape**). | `univariate_statistics.csv` and histograms in `03_key_distributions.png`. | Most borrowers have **0** in mild delinquency at median; income middle 50% is about **$3,903–$7,400**; age quartiles are about **41 / 52 / 63** years. |
| **Bivariate** | You study **two** things: here **one feature** vs the **target (0/1)**. | Boxplots in `07_bivariate_boxplots_by_target.png` and default-rate lines in `04_delinquency_vs_default_rate.png` and `05_utilization_deciles_default_rate.png`. | Risk factors visually separate **0 vs 1** groups; worse delinquency/utilization regions align with higher distress. |
| **Quartiles (Q1, median, Q3)** | Sort a column from smallest to largest. **Q1** is where 25% of values are below, **median (Q2)** is the middle value, and **Q3** is where 75% are below. So quartiles split data into 4 equal parts and show the **typical range**. **IQR = Q3 − Q1** is the middle 50%. | Columns `q1`, `median_q2`, `q3` in `univariate_statistics.csv`. | For **MonthlyIncome**, Q1 = **3903**, median = **5400**, Q3 = **7400**. This means the middle 50% of borrowers are roughly between **3903 and 7400**. For `30-59 DPD`, Q1=median=Q3=**0**, showing most borrowers have no such late event and only a smaller group drives risk in the tail. |
| **Skewness** | Skewness is a number that tells you whether a graph (usually a histogram) is balanced or leaning to one side. If both sides look similar, skewness is near 0. If there is a long tail to the right (a few very large values), skewness is positive. If there is a long tail to the left (a few very small values), skewness is negative. The number comes from one column by taking all values, finding the center (mean) and spread (standard deviation), measuring pull to one side (standard 3rd-moment formula), then converting that into one skewness value. Quick rule: around 0 = almost symmetric; 0.5 to 1 = mild right skew; >1 = strong right skew; very large positive (10, 20, 100+) = extremely long right tail; negative = left tail. | Column `skewness` in `univariate_statistics.csv`. | From `univariate_statistics.csv`: `age` skew ≈ **0.19** (close to symmetric), `NumberOfDependents` skew ≈ **1.63** (clear right skew), `DebtRatio` skew ≈ **95+**, and `MonthlyIncome` skew ≈ **127+** (extremely right-skewed: most rows are in a lower/normal band, but a small number of very large values create a huge right tail). |
| **Kurtosis (excess in pandas)** | Tells you about **extreme tails** vs a bell curve. Big = more extreme values. | Column `kurtosis_excess` in `univariate_statistics.csv`. | `age` is near **0** (mild tails). Debt/income fields have **huge** kurtosis, showing extreme tail values. |
| **Outlier detection (IQR)** | Rule: flag below **Q1 − 1.5×IQR** or above **Q3 + 1.5×IQR**. | `outlier_detection_iqr.csv`. | Debt ratio has about **~21%** rows flagged; age is **<0.04%**. Delinquency counts with many zeros produce tight fences, so many flagged points are not necessarily data errors. |
| **Hypothesis testing (Welch t-test)** | Tests if the **mean** differs between target **1** and **0** more than random chance; small p-value (often <0.05) indicates strong evidence of a mean difference. | `hypothesis_testing_results.csv`. | `age`, delinquency fields, and income are significant (**p < 0.05**). Revolving utilization is not significant on **mean difference** (**p ~ 0.22**), even though decile plots can still show risk patterns. |

---

## Part 5 — Figures (`outputs/figures/`)

Each image answers a visual question. **What the data shows** in your current run (see CSVs) is summarized under each.

### `01_target_distribution.png`
- **What it is:** Count of borrowers with target **0** vs **1** (bar chart).
- **What the data shows:** The **1** class is much smaller than **0** — about **6.7%** distress vs **~93.3%** no distress. This is **class imbalance**: a model can look “accurate” by mostly predicting 0, so you must use **ROC-AUC** (and the confusion matrix), not only accuracy.

### `02_correlation_heatmap.png`
- **What it is:** **Pearson correlation** between all numeric features and the target (and each other).
- **What the data shows:** Delinquency count variables **move together** (mild / moderate / severe late-payment fields are **positively** correlated with each other). The heatmap is about **linear** association; a weak cell does not mean “no risk,” only that a straight-line relationship is weak.

### `03_key_distributions.png`
- **What it is:** **Histograms** of utilization, age, debt ratio, and monthly income (for income/utilization/debt, very high values are **capped** in the plot so the graph is readable).
- **What the data shows:** **Age** is roughly **bell-shaped** (typical working-age borrowers). **Utilization, debt ratio, and income** are **heavily right-skewed** — most values sit in a moderate range, and a long tail of **extreme** values pulls means far above the median (see the univariate file for that).
- **Revolving Utilization (Top-Left, Blue):**
  - **What it measures:** The ratio of a person's total balance on credit cards and personal lines of credit to their total credit limit.
  - **Observation:** This is highly skewed to the left. The vast majority of individuals use very little of their available credit (close to 0.0). However, there is a notable spike right at 1.0, representing people who have maxed out their credit lines.
- **2. Age (Top-Right, Teal):**
  - **What it measures:** The age of the individuals in years.
  - **Observation:** This follows a roughly normal (bell-shaped) distribution, though it is slightly \"jagged.\" The bulk of the population in this dataset is between 35 and 65 years old, with the peak frequency occurring around age 50.
  - **EDA Note:** It shows a sensible range for credit applicants, starting around age 21 and trailing off after age 80.
- **3. Debt Ratio (Bottom-Left, Orange):**
  - **What it measures:** Monthly debt payments divided by gross monthly income.
  - **Observation:** This is extremely skewed. Almost the entire dataset has a debt ratio very close to 0. Because of the \"cap at 99.5th pct,\" we can see a tiny sliver of data points extending all the way out to 6000+, which likely represents extreme outliers or perhaps errors in the raw data.
  - **EDA Note:** In credit modeling, an extremely high debt ratio is usually a strong indicator of high risk.
- **4. Monthly Income (Bottom-Right, Purple):**
  - **What it measures:** The monthly income of the individual.
  - **Observation:** There is a general distribution peaking around 5,000, but there is a massive, sharp spike right around the 5,000–6,000 mark.
  - **EDA Note:** The title mentions \"imputed.\" This huge spike suggests that many individuals had missing income data, and the data scientist \"imputed\" (filled in) those missing values using a single value, likely the median or mean income of the group.
- **Summary for Credit Risk:**
  - In the context of the \"Target Distribution\" and \"Delinquency charts\" mentioned in your text:
    - **Utilization and Debt Ratio:** Higher values in these two charts generally correlate with a higher \"bad\" outcome rate (1s).
    - **Age and Income:** These are often used as stabilizing factors; for example, older individuals or those with higher steady incomes might be statistically less likely to default.

### `04_delinquency_vs_default_rate.png`
- **What it is:** For each integer count of past-due events (30–59, 60–89, 90+ days, **x** capped for display), the **default rate** = share of `SeriousDlqin2yrs = 1` in that group.
- **What the data shows:** **Higher past-due counts go with higher default rate** — the monotonic “worse history → more distress” story you expect in credit. That supports using delinquency features in a score.

### `05_utilization_deciles_default_rate.png`
- **What it is:** Borrowers are sorted into **10 utilization buckets** (deciles) after capping the upper tail; for each bucket the plot shows the **mean target** (default rate).
- **What the data shows:** **Higher revolving utilization** generally lines up with **higher** distress rate in the higher buckets (non-linear risk: not always a single straight line across the full range, but a clear “stress at the top” pattern).

### `07_bivariate_boxplots_by_target.png`
- **What it is:** **Boxplots** of key numeric variables **split** by **target** (0 = no distress, 1 = distress); some y-axes are capped and outliers may be hidden.
- **What the data shows:** **Defaulters (1)** and **non-defaulters (0)** differ in **center and spread** on these variables. T-tests (below) find **defaulters younger on average** and **higher** on delinquency counts; **lower** mean income in distress. Use the plot to **see** separation; use `hypothesis_testing_results.csv` for which **mean** differences are **statistically** significant.
- **First: What you are looking at**
  - Each chart compares two groups of people:
    - Target = 0 -> No distress (good customers)
    - Target = 1 -> Distress (customers likely to default)
  - Each box plot shows how a variable is distributed in those two groups.
- **Quick refresher: How to read a box plot**
  - Each box shows:
    - Middle line -> median (the "typical" value)
    - Box -> where most data lies (middle 50%)
    - Whiskers -> range (min to max, roughly)
    - If box is higher -> values are generally higher
- **1. Utilization by target (TOP LEFT)**
  - This is credit utilization (how much of your credit you are using).
  - What we see:
    - Target = 0 (good users): low median (~0.1), most values are small.
    - Target = 1 (distressed users): much higher median (~0.85), many people use a lot of their credit.
  - Meaning:
    - People who use too much of their credit are more likely to be in financial trouble.
  - Simple example:
    - Using 10% of credit -> safer
    - Using 90% of credit -> riskier
- **2. Debt ratio by target (TOP RIGHT)**
  - Debt ratio = how much debt you have compared to income.
  - What we see:
    - Both groups look somewhat similar.
    - Distressed group (1) has a slightly higher median and wider spread (more extreme values).
  - Meaning:
    - Debt ratio matters, but it is not as strong a predictor as utilization in this view.
- **3. Monthly income by target (BOTTOM LEFT)**
  - What we see:
    - Target = 0 (good users): slightly higher median income and wider range.
    - Target = 1 (distressed users): slightly lower median income.
  - Meaning:
    - Lower income is linked to distress, but not always; some high-income people also default.
- **4. 90+ days late count (BOTTOM RIGHT)**
  - This is very important.
  - What we see:
    - Target = 0: almost always 0 (no serious late payments).
    - Target = 1: many have 1 or more late payments; some go higher.
  - Meaning:
    - If someone is 90+ days late, they are much more likely to default.
    - This is one of the strongest signals in the dataset.
- **Final Summary**
  - Strong predictors of default:
    - High credit utilization
    - History of 90+ days late
  - Medium predictor:
    - Debt ratio
  - Weak/moderate predictor:
    - Income

### `06_feature_importance_random_forest.png`
- **What it is:** **Random forest** “importance” (how much each feature was used in tree splits, MDI).
- **What the data shows:** A **ranking** of which raw inputs the forest relied on most for *this* model and random seed. It is **not** proof of business causality, but it tells you what the algorithm treated as most informative in the baseline.

---

## Part 6 — Output files (CSVs and summary text)

### `data_quality_table.csv`
- **What it is:** One row per column: `dtype`, `non_null`, `missing_count`, **`missing_pct`**, `n_unique`, and for numerics `min`/`max`.
- **What the data shows:** **Income** ~**19.8%** missing; **dependents** ~**2.6%** in raw `NA` cells.

### `univariate_statistics.csv`
- **What it is:** Per numeric feature: `mean`, `std`, `min`, **`q1`**, **`median_q2`**, **`q3`**, `max`, `iqr`, **`skewness`**, **`kurtosis_excess`**.
- **What the data shows:** The **quartile columns** are the **real cut points** in the data after sorting. Compare **mean vs median** to see **tail** effects (debt, income, utilization).

### `outlier_detection_iqr.csv`
- **What it is:** IQR **fences**, outlier **counts** and **% of rows** per feature.
- **What the data shows:** Which fields are “extreme” by the **1.5×IQR** rule; high % often means **real tail risk** or many **zeros** on count variables, not only typos.

### `hypothesis_testing_results.csv`
- **What it is:** Mean in **target=1** vs **target=0**, **difference**, **p-value**, **significant** flag.
- **What the data shows:** Which **average** feature levels **differ** by distress status at usual significance levels.

### `eda_summary.txt`
- **What it is:** Short **narrative** snapshot: N rows, default %, **missing** lines, **model** ROC-AUC / accuracy, top lines from the statistics blocks.
- **What the data shows:** A **one-page** paste for your **write-up**; use **CSVs** for the full table.

### Figures
- All discussion under **Part 5**.

---

## Part 7 — Baseline models (illustrative; re-run to refresh)

- **Logistic regression** — example order of ROC-AUC in high **0.8**s; accuracy in high **0.7**s–**0.8**s.  
- **Random forest** — example ROC-AUC a bit **higher**; accuracy in **0.8**s.  

Exact numbers: open **`outputs/eda_summary.txt`** after you run the script. They show the **same patterns** in Parts 5–7 are not only for pictures; a standard model can **use** them to **rank** risk on held-out data.

---

*Re-run `python eda_credit_scoring.py` before a grade or presentation so figures, `eda_summary.txt`, and CSVs match one run.*
