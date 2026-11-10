# 0. General Information

## Project Title

**Airbnb Price Prediction in New York City Using Tree Ensembles and Segment-Level Price Deviation Analysis**

## Project Description

This project investigates Airbnb listing price prediction in New York City using machine learning regression models. It compares six algorithms, including Linear Regression, Ridge Regression, Lasso Regression, Random Forest, XGBoost, and Gradient Boosting.

The study evaluates predictive performance and examines model-relative price deviations across different listing segments. It aims to identify how listing characteristics relate to predicted prices and whether particular segments exhibit consistent deviations from model predictions.

## Research Team

- **Group:** Group 9
- **Institution:** FPT University
- **Campus:** Ho Chi Minh City
- **Course:** ADY201m
- **Semester:** FA-26
- **Supervisor / Instructor:** Lê Nhật Tùng

### Team Members

| No. | Full Name             | Student ID | Main Responsibilities |
| --- | --------------------- | ---------- | --------------------- |
| 1   | Nguyễn Ngọc Đăng Khoa | SE210374   | Code                  |
| 2   | Huỳnh Gia Phú         | SE210275   | Paper                 |

## Repository

- **GitHub Repository:** https://github.com/nguyenngocdangkhoa16122007-collab/airbnb-price-prediction
- **Paper Version:** [TODO: Final paper version]
- **Repository Version / Commit:** [TODO: Commit hash or release tag]
- **Last Updated:** [TODO: Date]

---
# 1. Quick Start

This section provides the fastest way to reproduce the main results
without manually running the research notebook.
The script reproduces the main model evaluation and analytical results, including the injected-shift sensitivity analysis.

## 1.1 Clone the repository

Open a terminal and run:

```bash
git clone https://github.com/nguyenngocdangkhoa16122007-collab/airbnb-price-prediction.git
cd airbnb-price-prediction
```

## 1.2 Set up the Python environment

Create and activate a virtual environment.

**Windows:**

```bash
python -m venv .venv
.venv\Scripts\activate
```

**macOS / Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the required packages:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 1.3 Check the dataset

Ensure that the dataset is available at:

data/raw/AB_NYC_2019.csv

The script expects the original NYC Airbnb Open Data file
containing 48,895 rows and 16 columns.

## 1.4 Run the reproduction script

From the repository root, execute:

```bash
python reproduce.py
```

The script automatically loads and validates the dataset, performs
preprocessing, trains the six configured models across five
train/test splits, and generates the evaluation and analysis outputs
(including the injected-shift sensitivity analysis).

No hyperparameter tuning is performed during this run. The script
uses the final hyperparameters specified in the source code.

## 1.5 Locate the generated results

After successful execution, the results are saved in:

- `outputs/tables/`: per-split results, feature importance,
pricing-deviation results, sensitivity results, and other detailed tables.
- `outputs/metrics/`: aggregated model metrics, cross-validation
summaries, smearing factor, sensitivity summaries, and run metadata.
- `outputs/figures/`: model comparison, calibration,
feature-importance, and pricing-deviation figures.

The script prints progress messages and elapsed time in the terminal.
A successful run ends with:

Quick Reproduction finished. Outputs are in outputs/tables,
outputs/metrics, outputs/figures.

---

# 2. Folder Structure

The repository is organized as follows:

```text
airbnb-price-prediction/
├── README.md
├── requirements.txt
├── reproduce.py
├── notebooks/
│ ├── .gitkeep
│ └── Full_notebookk.ipynb
├── data/
│ ├── raw/
│ │ └── AB_NYC_2019.csv
│ └── processed/
│ └── .gitkeep
├── configs/
│ └── .gitkeep
├── outputs/
│ ├── tables/
│ │ └── .gitkeep
│ ├── figures/
│ │ └── .gitkeep
│ └── metrics/
│ └── .gitkeep
└── checkpoints/
└── .gitkeep
```

## Directory Description

| Path               | Description                                                                                                                                                        |
| ------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `README.md`        | Provides an overview of the project and basic repository instructions.                                                                                             |
| `requirements.txt` | Lists the Python packages and their pinned versions required by the project.                                                                                       |
| `reproduce.py`     | Runs the main training and evaluation pipeline using the fixed hyperparameters documented in this guide, including the injected-shift sensitivity analysis.         |
| `notebooks/`       | Contains the complete research notebook, including exploratory analysis, preprocessing, hyperparameter optimization, model evaluation, and supplementary analyses. |
| `data/raw/`        | Stores the original dataset used in the study.                                                                                                                     |
| `data/processed/`  | Reserved for processed datasets generated during preprocessing.                                                                                                    |
| `configs/`         | Reserved for external configuration files. Final hyperparameters are currently defined directly in `reproduce.py`.                                                 |
| `outputs/tables/`  | Stores generated result tables.                                                                                                                                    |
| `outputs/figures/` | Stores generated figures and visualizations.                                                                                                                       |
| `outputs/metrics/` | Stores model evaluation metrics and related numerical results.                                                                                                     |
| `checkpoints/`     | Reserved for saved model checkpoints or trained model artifacts.                                                                                                   |

The complete research workflow is documented in `notebooks/Full_notebookk.ipynb`, which serves as the primary reference for the original experiments and supplementary analyses. The separate `reproduce.py` script provides a shorter reproduction pipeline using the fixed hyperparameters selected during the research phase.

---
# 3. Hardware & Software Requirements

## 3.1 Hardware Requirements

The project was developed and executed using Google Colab. Since Colab allocates computational resources dynamically, the actual hardware configuration may vary between sessions.

The following hardware is recommended for running the experiments:
- CPU: Multi-core processor.
- RAM: At least 8 GB (16 GB recommended).
- Storage: At least 2 GB of available space.
- GPU: Not required; the experiments can run on CPU.

Actual execution time may vary depending on hardware and system load.

## 3.2 Software Requirements

- Python: 3.13.18
- pip: Python package installer
- Git: Required to clone the repository

## 3.3 Python Dependencies

The project uses the following package versions:

| Package      | Version |
| ------------ | ------: |
| numpy        |   2.4.6 |
| pandas       |   2.2.3 |
| duckdb       |   1.3.2 |
| matplotlib   |  3.10.0 |
| seaborn      |  0.13.2 |
| scikit-learn |   1.6.1 |
| statsmodels  |  0.15.0 |
| xgboost      |   3.4.1 |

All dependencies are listed in `requirements.txt`.

---
# 4. Environment Setup

## 4.1 Clone the Repository

Open a terminal and run the following commands:

```bash
git clone https://github.com/nguyenngocdangkhoa16122007-collab/airbnb-price-prediction.git
cd airbnb-price-prediction
```

## 4.2 Create a Virtual Environment

A virtual environment is recommended to avoid conflicts with packages installed in other Python projects.

**Windows:**

```bash
python -m venv .venv
.venv\Scripts\activate
```

**macOS / Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## 4.3 Install Dependencies

After activating the virtual environment, install the required packages:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The `requirements.txt` file specifies the package versions used in this project.

## 4.4 Verify the Installation

Run the following command to check whether all required packages can be imported:

```bash
python -c "import numpy, pandas, duckdb, matplotlib, seaborn, sklearn, statsmodels, xgboost; print('All packages imported successfully.')"
```

If the command finishes without an import error, the package installation is successful.

## 4.5 Verify the Dataset

Before running the reproduction script, ensure that the dataset is available at:

data/raw/AB_NYC_2019.csv

The expected original dataset contains 48,895 rows and 16 columns.

If the file is missing, place the original `AB_NYC_2019.csv` dataset in the `data/raw/` directory.

## 4.6 Run the Reproduction Script

From the repository root directory, execute:

```bash
python reproduce.py
```

The script will perform the main model evaluation and analyses (including the injected-shift sensitivity analysis), then save generated outputs in the following directories:

- `outputs/tables/`
- `outputs/metrics/`
- `outputs/figures/`

For the complete research workflow, including the original hyperparameter optimization and exploratory analyses, refer to `notebooks/Full_notebookk.ipynb`.

---
# 5. Data

## 5.1 Dataset Source

This project uses the New York City Airbnb Open Data dataset by Gomonov (2019), redistributed through Kaggle from publicly available listings collected by the Inside Airbnb project.

- Dataset: New York City Airbnb Open Data
- Kaggle dataset page: https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data
- License: CC0 1.0 (Public Domain)
- File name: `AB_NYC_2019.csv`
- Original size: 48,895 rows × 16 columns
- Repository path: `data/raw/AB_NYC_2019.csv`

The raw CSV file is included in the GitHub repository. No separate download is required when cloning the repository.

## 5.2 Data Cleaning

The reproduction script loads the original CSV file and performs the following cleaning operations:

1. Removes listings with a price of zero or less (11 rows in the original dataset).
2. Fills missing values in `reviews_per_month` with zero. These missing values correspond to listings with no reviews.
3. Derives a binary `has_review` feature to indicate whether a listing has received reviews.

The variables `last_review` and `reviews_per_month` contain missing values for 10,052 listings (approximately 20.6% of the original dataset). These missing values are treated as structural rather than arbitrary missing observations because the corresponding listings have `number_of_reviews = 0`.

## 5.3 Feature Engineering and Preprocessing

The following transformations are applied during preprocessing:

**Numerical features**
- Applies `log1p(x)` transformations to `minimum_nights`, `number_of_reviews`, `reviews_per_month`, and `calculated_host_listings_count`.
- Caps the target price and `minimum_nights` at their respective 99th percentiles before applying the target log transformation. The percentile thresholds are estimated from the training data only.
- Transforms the target variable using `log1p(price)`.
- Converts `last_review` into `days_since_last_review`, measured relative to 31 December 2019. Listings without reviews receive a sentinel value estimated from the training data.
- Creates `is_unavailable`, indicating whether `availability_365` equals zero.

**Categorical encoding and scaling**
- One-hot encodes `neighbourhood_group` and `room_type`.
- Drops one reference category for Linear Regression, Ridge, and Lasso; retains all indicators for tree-based models.
- Standardizes numerical features using training-partition statistics for the linear models. Latitude and longitude were standardized during preprocessing for all six models. For Linear Regression, Ridge, and Lasso, an additional `StandardScaler` was applied to the complete feature matrix within the modeling pipeline. Therefore, the two geographic coordinates underwent an additional scaling step in the linear-model pipeline.
- Encodes `neighbourhood` using a five-fold cross-fitted target encoder with automatic m-estimate smoothing. Training observations are encoded using out-of-fold estimates, while validation and test observations are transformed using an encoder fitted on the full training partition. Unseen neighbourhood categories are mapped to the training-target mean.

**Geographic features**
- Calculates Haversine distances in kilometres to:
- Times Square (40.75636, -73.98644)
- Wall Street financial district (40.7074, -74.0113)

**Title-derived features**
Ten features are extracted from the listing title (`name`):
- `n_bed_name`: stated bedroom count; "studio" is encoded as zero, while -1 indicates that the count cannot be extracted.
- `n_bed_name_missing`: indicates whether the bedroom count could not be extracted.
- Eight binary keyword indicators: `luxury`, `penthouse`, `loft`, `townhouse`, `duplex`, `cozy`, `bunk`, and `hostel`.

Raw `host_name` and `name` text columns are not directly included as predictors. Only the numerical and binary features derived from `name` are retained. The raw target and price-derived quantities are excluded from the predictor variables.

## 5.4 Data Leakage Prevention

All data-dependent preprocessing steps are fitted using the relevant training partition only. This includes:

- 99th-percentile capping thresholds;
- numerical scaling statistics;
- neighbourhood target encoding;
- the sentinel value for listings without a recorded review date.

The fitted transformations are then applied unchanged to the corresponding validation or test partition.

Each fitted transformation is applied unchanged to the corresponding validation or test partition. This ensures that data-dependent preprocessing parameters are estimated exclusively from the relevant training data.

## 5.5 Data Integrity

The SHA256 checksum of the raw dataset is provided below to help verify file integrity.

- File: `data/raw/AB_NYC_2019.csv`
- Algorithm: SHA256
- Checksum: `e420db40ff10fcb40efc1b5b1648ee0b18a48f4e4537155cecc59fe95d18783a`

To calculate the checksum on Windows PowerShell, run:

```bash
Get-FileHash .\data\raw\AB_NYC_2019.csv -Algorithm SHA256
```

On macOS or Linux, run:

```bash
sha256sum data/raw/AB_NYC_2019.csv
```

The calculated checksum should match the value reported in this section.
---
# 6. Config & Hyperparameters

## 6.1 Configuration Management

The final hyperparameters used for reproduction are explicitly defined in `reproduce.py`. The current repository does not use separate YAML configuration files.

The original hyperparameter optimization was performed during the research phase. The selected configurations were subsequently fixed in the reproduction script to avoid repeating the computationally expensive search process.

## 6.2 Hyperparameter Optimization

Hyperparameter optimization was conducted using two approaches:

**Linear models**

- Ridge Regression: Grid Search over 25 logarithmically spaced alpha values from 0.0001 to 100.
- Lasso Regression: Grid Search over 30 logarithmically spaced alpha values from 0.000001 to 10.
- Linear Regression: No hyperparameter search was performed.

**Tree-based models**

Randomized hyperparameter sampling was implemented using scikit-learn's `ParameterSampler`, with 25 sampled configurations for each model. Each sampled configuration was evaluated through a custom five-fold cross-validation procedure:

- Random Forest
- XGBoost
- Gradient Boosting

The sampling process used `random_state=42`.

All candidate configurations were evaluated using five-fold stratified cross-validation on the training data. The mean MAE in the original USD price scale was used as the primary selection metric. The final configuration for each tuned model was selected based on the lowest mean cross-validation MAE.

The held-out test set was not used for hyperparameter selection.

**Reproduction note on HPO verification.** The hyperparameter search procedure and its intermediate search-result tables are documented in the research notebook. The current `reproduce.py` run does **not** re-execute or re-log the original hyperparameter search; it only uses the final fixed configurations listed in Section 6.4. Consequently, the original HPO search logs and candidate rankings have **not** been independently re-confirmed by the present reproduction outputs.

## 6.3 Search Spaces

The search spaces for the tree-based models were:

**Random Forest**

| Hyperparameter     | Candidate values    |
| ------------------ | ------------------- |
| `n_estimators`     | {300, 500}          |
| `max_depth`        | {8, 10, 12, 15, 18} |
| `min_samples_leaf` | {1, 3, 5, 10, 20}   |
| `max_features`     | {sqrt, 0.5, 0.7}    |

**XGBoost**

| Hyperparameter     | Candidate values      |
| ------------------ | --------------------- |
| `n_estimators`     | {300, 500, 700, 1000} |
| `max_depth`        | {6, 8, 10, 12}        |
| `learning_rate`    | {0.01, 0.03, 0.05}    |
| `subsample`        | {0.6, 0.8, 1.0}       |
| `colsample_bytree` | {0.6, 0.8, 1.0}       |
| `min_child_weight` | {1, 3, 5}             |
| `reg_lambda`       | {1, 2, 5}             |

**Gradient Boosting**

| Hyperparameter     | Candidate values        |
| ------------------ | ----------------------- |
| `n_estimators`     | {100, 150, 200, 300}    |
| `max_depth`        | {3, 4, 5, 6, 8}         |
| `learning_rate`    | {0.01, 0.03, 0.05, 0.1} |
| `subsample`        | {0.6, 0.8, 1.0}         |
| `min_samples_leaf` | {5, 10, 20, 50}         |

## 6.4 Final Model Hyperparameters

The following configurations are fixed in `reproduce.py`:

| Model             | Final hyperparameters                                                                                                                   |
| ----------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| Linear Regression | Default scikit-learn configuration                                                                                                      |
| Ridge             | `alpha=1.0`                                                                                                                             |
| Lasso             | `alpha=0.0001`                                                                                                                          |
| Random Forest     | `n_estimators=500`, `max_depth=18`, `min_samples_leaf=1`, `max_features=0.5`                                                            |
| XGBoost           | `n_estimators=1000`, `max_depth=6`, `learning_rate=0.03`, `subsample=0.8`, `colsample_bytree=0.6`, `min_child_weight=1`, `reg_lambda=5` |
| Gradient Boosting | `n_estimators=300`, `max_depth=8`, `learning_rate=0.1`, `subsample=1.0`, `min_samples_leaf=5`                                           |

XGBoost was selected as the best-performing model based on mean cross-validation MAE.

## 6.5 Reproduction Notes

The reproduction script does not repeat the original hyperparameter search. Instead, it uses the fixed hyperparameters listed in Section 6.4.

The original search-result DataFrames were not exported as `outputs/hpo_results.csv`. Therefore, this file is not included in the current reproduction package.

The preprocessing pipeline is implemented directly in `reproduce.py`. Its data-dependent transformations are fitted using the relevant training partition, as described in Section 5.

One-hot encoding was configured differently for linear and tree-based models. Linear models used reference-category encoding (drop_first=True) to avoid perfect multicollinearity, whereas tree-based models retained all dummy variables (drop_first=False). Latitude and longitude were standardized during preprocessing for all models. Additionally, linear models used a StandardScaler within the modeling pipeline to standardize the complete feature matrix.

The hyperparameter search and model-selection procedures used mean cross-validation MAE measured on the original USD price scale. Predictions were transformed back from the log scale using the inverse `expm1` transformation and clipped at zero before evaluation.

---
## 7. Training

### 7.1. Training Procedure

All six regression models reported in the study were trained using the `reproduce.py` script. The script uses the final hyperparameter configurations specified directly in the source code; therefore, hyperparameter tuning is not repeated during reproduction.

To execute the training and reproduction pipeline, run the following command from the repository root:

```bash
python reproduce.py
```

The pipeline evaluates the six models: Linear Regression, Ridge Regression, Lasso Regression, Random Forest, XGBoost, and Gradient Boosting. Training and evaluation are repeated using five random seeds: 42, 1, 7, 123, and 2024. Each model is trained using the corresponding training partition, with preprocessing fitted only on the relevant training data to prevent information leakage.

Each split follows the 85:15 training-to-test ratio, with stratification based on quantile-based price bins.

### 7.2. Model Configuration

All models use the final hyperparameters documented in Section 6. The reproduction script loads these configurations directly and does not perform additional hyperparameter searches.

### 7.3. Output Artifacts

Generated results are saved in the following directories:

- `outputs/tables/`: generated result tables.
- `outputs/metrics/`: evaluation metrics and related numerical outputs.
- `outputs/figures/`: generated visualizations.

The script does not rely on separately saved deep-learning checkpoints, epoch-based training logs, or TensorBoard/W&B tracking.

The script does not save trained model checkpoints.

### 7.4. Computational Resources

The experiments were conducted using Google Colab. The actual execution environment and runtime depend on the allocated Colab resources.

- Environment: Google Colab.
- CPU: x86_64.
- Available RAM: Approximately 12.67 GB in the recorded execution environment.
- GPU: Not used; all experiments were executed on CPU.
- Total execution time: approximately 5,300 seconds for `reproduce.py` (may increase after inclusion of the sensitivity analysis with B = 4000).

---
## 8. Evaluation and Inference

### 8.1. Full Evaluation Workflow

The complete research workflow is documented in `notebooks/Full_notebookk.ipynb`, which serves as the primary source of the experimental procedures and analysis. It covers exploratory data analysis (EDA), preprocessing, feature engineering, hyperparameter tuning, model selection, final evaluation, and supplementary analyses.

To reproduce the complete study, open the notebook in Google Colab or a compatible Jupyter environment, ensure that the dataset is available at the expected path, and execute the relevant cells in their documented order. The notebook should be used to reproduce the full experimental process, including hyperparameter searches that are not repeated by the quick reproduction script.

### 8.2. Quick Evaluation Workflow

For a shorter verification of the main experimental pipeline, run:

```bash
python reproduce.py
```

This script performs training and evaluation using the fixed hyperparameters documented in Section 6. It reproduces the six-model comparison across five train/test splits, RQ1 feature importance analysis, RQ2 model comparison, RQ3 segment-level deviation analysis, the injected-shift sensitivity analysis, host-type comparisons, the StratifiedKFold versus GroupKFold comparison, and calibration analysis.

The script does not repeat EDA or hyperparameter tuning. It also does not save trained model checkpoints or provide a separate evaluation-only mode. Therefore, running it again repeats the training and evaluation procedures.

### 8.3. Evaluation Metrics

Regression performance is evaluated using the following metrics:

- **Mean Absolute Error (MAE):** Calculated using `sklearn.metrics.mean_absolute_error`; measures the average absolute prediction error in USD.
- **Root Mean Squared Error (RMSE):** Calculated as the square root of `sklearn.metrics.mean_squared_error`; measures the prediction error in USD while assigning greater weight to larger errors.
- **Coefficient of Determination (R²):** Calculated using `sklearn.metrics.r2_score`. The study reports R² on the original price scale, capped-price scale, and log-price scale as separate metrics.

Predictions are transformed back from the log scale to the original USD scale before calculating MAE and RMSE. Model selection uses mean cross-validation MAE on the original USD scale, while final performance is evaluated on held-out test data.

Results are reported for five random train/test splits, with mean and standard deviation used to summarize performance across splits.

**Reproduction note on train–test R² gap.** The manuscript discusses the gap between training and test R². The current `reproduce.py` outputs focus on held-out test metrics (MAE, RMSE, R² on log / capped / raw scales) across the five splits. Explicit train-side R² (and therefore a fully logged train–test R² gap table) is **not** regenerated by the present reproduction script and has **not** been re-confirmed from the current reproduction logs. Users who need the train-side R² values should consult the research notebook.

### 8.4. Statistical and Supplementary Evaluation

In addition to regression performance, the study conducts the following analyses:

- **Feature importance (RQ1):** Lasso coefficients, Random Forest impurity-based importance, and XGBoost gain-based importance, using the split with seed 42.
- **Model comparison (RQ2):** Paired MAE comparisons between XGBoost and Random Forest, and between XGBoost and Gradient Boosting, across the five splits.
- **Segment-level deviation analysis (RQ3):** Host-clustered bootstrap with 4,000 replicates, basic bootstrap confidence intervals, p-values, Holm correction, a 5% practical threshold, and cross-split direction consistency.
- **Injected-shift sensitivity analysis:** Paired host-cluster bootstrap (B = 4,000) on two selected segments with injected shifts of 5%, 10%, and 20%, across the five splits, with Holm correction and practical-threshold checks.
- **Host-type analysis:** Multi-host versus single-host comparisons using host-clustered bootstrap with 4,000 replicates, both overall and within each room type.
- **Cross-validation comparison:** StratifiedKFold versus GroupKFold for Random Forest, XGBoost, and Gradient Boosting, using the training partition from the split with seed 42.
- **Calibration analysis:** An out-of-fold smearing factor estimated from XGBoost residuals and decile-level comparisons of median predictions and smearing-corrected mean predictions.

### 8.5. Output Artifacts

The quick reproduction script saves its generated results in the following directories:

- `outputs/tables/`: per-split evaluation results, feature importance tables, segment-level results, sensitivity results, host-type results, and calibration tables.
- `outputs/metrics/`: model performance summaries, paired MAE comparisons, cross-validation summaries, smearing factor, sensitivity summaries, host-type summaries, and run metadata.
- `outputs/figures/`: model comparison, feature importance, segment deviation, host-type, and calibration figures.

The generated files include CSV tables, JSON summaries, and PNG figures. Their filenames are defined directly in `reproduce.py`.

Key sensitivity outputs produced by `reproduce.py` include:

- `outputs/tables/sensitivity_all_segments_by_split.csv`
- `outputs/tables/sensitivity_target_detail_by_split.csv`
- `outputs/metrics/sensitivity_target_summary_5_splits.csv`

The complete notebook may also contain intermediate analysis results that are not exported by the quick reproduction script.

### 8.6. Inference on New Data

The project focuses on experimental evaluation and model comparison rather than deployment as a prediction service. Neither the quick reproduction script nor the documented workflow provides a standalone command-line interface, REST API, or interactive web application for predicting prices of arbitrary new listings.

Accordingly, no separate inference command or demo interface is provided.

---
## 9. Randomness and Determinism

The experiments use fixed random seeds to improve reproducibility across repeated runs.

- **Repeated train/test splits:** Five seeds are used: `42`, `1`, `7`, `123`, and `2024`.
- **Cross-validation:** Stratified K-fold splitting uses a fixed random state of `42`.
- **Hyperparameter sampling:** `ParameterSampler` uses `random_state=42` for the Random Forest, XGBoost, and Gradient Boosting searches.
- **Model-level randomness:** Random Forest, XGBoost, Gradient Boosting, and Lasso use `random_state=42` where applicable.
- **Target encoding:** The cross-fitted target encoder uses `random_state=42`.
- **Bootstrap analysis:** The main RQ3, host-type, and injected-shift sensitivity analyses use fixed random seeds (and B = 4,000 replicates) to make the resampling procedures reproducible.

The `reproduce.py` script uses the final, predefined hyperparameters and does not rerun hyperparameter optimization. It generates results for the five specified train/test seeds and saves the resulting evaluation outputs.

Exact numerical results may still vary slightly across different software versions, hardware configurations, or computational environments. The project does not use PyTorch, GPU-based training, or DataLoader workers; therefore, PyTorch-specific deterministic settings are not applicable.

No universal numerical tolerance is prescribed for cross-environment reproduction. Small differences may occur due to software and hardware variations; the reported metrics should be compared using the same dataset, preprocessing procedure, split seeds, and evaluation definitions.

---
## 10. Mapping Results to the Report

The following table maps the reported results to their corresponding reproduction procedures, configurations, and generated files.

| Report item                                       | Reproduction command / procedure  | Configuration                                                                                              | Output file(s)                                                                                                                                           | Key result                                                                                                                                                     |
| ------------------------------------------------- | --------------------------------- | ---------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| RQ2 – Model performance comparison                | `python reproduce.py`             | Six models; five train/test seeds; fixed hyperparameters                                                   | `outputs/tables/rq2_model_results_by_seed.csv`; `outputs/metrics/rq2_model_performance_summary.csv`; `outputs/figures/model_comparison.png`              | XGBoost achieved the best overall predictive performance.                                                                                                      |
| RQ2 – Paired MAE comparison                       | `python reproduce.py`             | Paired comparison across the five test splits                                                              | `outputs/tables/rq2_paired_mae_by_seed.csv`; `outputs/metrics/rq2_paired_mae_summary.csv`                                                                | Compares model-level MAE differences across repeated splits.                                                                                                   |
| RQ1 – Lasso coefficients                          | `python reproduce.py`             | Final Lasso configuration                                                                                  | `outputs/tables/rq1_lasso_coefficients.csv`; `outputs/figures/lasso_coefficients.png`                                                                    | Reports the selected linear-model coefficients.                                                                                                                |
| RQ1 – Feature importance                          | `python reproduce.py`             | Random Forest and XGBoost with final hyperparameters                                                       | `outputs/tables/rq1_feature_importance.csv`; `outputs/figures/RQ1_Feature_Importance2.png`                                                               | Entire-home room type was among the most important predictors.                                                                                                 |
| RQ3 – Segment-level pricing deviations            | `python reproduce.py`             | Five test splits; host-clustered bootstrap (B = 4000); Holm correction; practical threshold                | `outputs/tables/rq3_segment_results_by_split.csv`; `outputs/tables/rq3_segment_summary_5_splits.csv`; `outputs/figures/rq3_dotplot_deviation_en.png`     | No segment satisfied all four criteria for a systematic pricing deviation.                                                                                     |
| Injected-shift sensitivity analysis               | `python reproduce.py`             | Two selected segments; shifts of 5%, 10%, 20%; paired host-cluster bootstrap (B = 4000); Holm correction   | `outputs/tables/sensitivity_all_segments_by_split.csv`; `outputs/tables/sensitivity_target_detail_by_split.csv`; `outputs/metrics/sensitivity_target_summary_5_splits.csv` | Both segments showed positive changes across all five splits. The practical threshold was reached in 5/5 splits for 10% and 20% shifts, but not for 5% shifts. |
| Host-type comparison                              | `python reproduce.py`             | Direct multi-host versus single-host comparison; host-clustered bootstrap (B = 4000)                       | `outputs/tables/host_type_overall_by_seed.csv`; `outputs/metrics/host_type_overall_consistency.json`; `outputs/figures/host_type_forest_plot.png`        | The direct host-type differences were not statistically supported.                                                                                             |
| Host-type comparison by room type                 | `python reproduce.py`             | Separate analysis by room type; host-clustered bootstrap (B = 4000)                                        | `outputs/tables/host_type_by_room_type_by_split.csv`; `outputs/metrics/host_type_by_room_type_summary.csv`; `outputs/figures/host_type_by_room_type.png` | Exploratory host-type differences across room types.                                                                                                           |
| GroupKFold generalization analysis                | `python reproduce.py`             | GroupKFold and StratifiedKFold evaluation                                                                  | `outputs/tables/groupkfold_vs_stratifiedkfold_fold_mae.csv`; `outputs/metrics/groupkfold_vs_stratifiedkfold_summary.csv`                                 | Compares generalization performance under listing-level and host-grouped validation.                                                                           |
| Prediction calibration                            | `python reproduce.py`             | Selected XGBoost model; decile-based calibration and smearing correction                                   | `outputs/tables/calibration_by_decile.csv`; `outputs/metrics/smearing_factor.json`; `outputs/figures/calibration.png`                                    | Examines calibration across predicted-price deciles and mean-price correction.                                                                                 |
| Run metadata                                      | `python reproduce.py`             | Recorded execution environment and experiment metadata                                                     | `outputs/metrics/run_metadata.json`                                                                                                                      | Records metadata associated with the reproduction run.                                                                                                         |

**Reproduction scope and limitations**

- Running `python reproduce.py` regenerates the outputs listed for the analyses above, including the injected-shift sensitivity analysis. It does not rerun hyperparameter optimization because the final hyperparameters are predefined in the script.
- The original hyperparameter search logs and an explicit train–test R² gap table are **not** regenerated by the current reproduction script and have **not** been re-confirmed from the present reproduction outputs (see Sections 6.2 and 8.3).
- Output filenames are listed as generated by the current version of `reproduce.py`.

---
## 11. Limitations and Troubleshooting

### 11.1. Reproduction Limitations

The main experimental workflow can be rerun using `reproduce.py` and the dataset included in the repository. However, exact numerical agreement with the manuscript is not guaranteed in every analysis.

The following limitations should be considered:

- **GroupKFold variability:** The GroupKFold evaluation may produce slightly different results across executions due to differences in the grouping and fold assignment process. In the reproduction run, the mean MAE was approximately **$55.15 ± $3.70**, compared with **$55.09 ± $3.75** reported in the manuscript. The difference in mean MAE is approximately $0.06.
- **Computational cost:** A complete execution of `reproduce.py` took approximately **4,600 seconds (76 minutes and 40 seconds)** in a newly created Google Colab notebook before the sensitivity analysis was added. With the injected-shift analysis and host-type bootstraps both using B = 4,000, total runtime is expected to increase. This measurement represents one recorded execution and should be treated as a reference rather than a guaranteed runtime. Actual execution time may vary depending on the CPU resources allocated by Google Colab and the computational environment.
- **Hyperparameter search not re-logged:** The original HPO search is not re-executed by `reproduce.py`. Intermediate search rankings and HPO logs have therefore not been independently re-confirmed by the current reproduction outputs.
- **Train–test R² gap not re-logged:** Explicit train-side R² (and the associated train–test R² gap) is not exported by the current `reproduce.py` run and has not been re-confirmed from the present reproduction logs.
- **Environment-dependent variation:** Small numerical differences may occur due to differences in software versions, hardware, and computational execution.

The reproduced outputs were observed to be broadly consistent with the reported findings. However, exact agreement across every numerical result has not been independently verified.

### 11.2. Troubleshooting

No specific installation or execution errors have been documented during the preparation of this reproduction guide. The following table lists potential issues and suggested solutions.

| Issue                                              | Possible cause                                                                    | Suggested solution                                                                           |
| -------------------------------------------------- | --------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| `ModuleNotFoundError`                              | Required package is not installed or the wrong Python environment is active.      | Activate the project environment and run `pip install -r requirements.txt`.                  |
| `FileNotFoundError` for the dataset                | The script is executed without the expected dataset file or repository structure. | Verify that `data/raw/AB_NYC_2019.csv` exists in the repository.                             |
| `PermissionError` when saving outputs              | The output directory is not writable.                                             | Check directory permissions and ensure the project is stored in a writable location.         |
| Execution takes substantially longer than expected | Limited CPU resources or competing processes; sensitivity analysis uses B = 4000. | Close unnecessary processes and allow sufficient execution time.                             |
| Different GroupKFold metrics                       | Differences in grouping, fold assignment, or execution environment.               | Check the recorded metadata and compare the evaluation procedure and grouping configuration. |

### 11.3. Known Numerical Differences

The GroupKFold results may differ slightly from those reported in the manuscript. The reproduced mean MAE was approximately $55.09, compared with $55.15 in the reported results. The corresponding standard deviations were $3.75 and $3.70, respectively.

These differences are small in magnitude and do not materially change the interpretation of the generalization analysis. Nevertheless, the reproduced values should be distinguished from the originally reported values rather than treated as exact matches.

---
# 12. Third-Party Libraries and References

The project uses the following Python libraries and tools:

| Library / Tool | Main purpose                                                       | Official reference              |
| -------------- | ------------------------------------------------------------------ | ------------------------------- |
| NumPy          | Numerical computing and array operations                           | https://numpy.org/              |
| pandas         | Data manipulation and preprocessing                                | https://pandas.pydata.org/      |
| DuckDB         | Data querying and analytical operations                            | https://duckdb.org/             |
| Matplotlib     | Plotting and visualization                                         | https://matplotlib.org/         |
| Seaborn        | Statistical data visualization                                     | https://seaborn.pydata.org/     |
| scikit-learn   | Preprocessing, cross-validation, regression models, and evaluation | https://scikit-learn.org/       |
| statsmodels    | Statistical procedures and multiple-testing correction             | https://www.statsmodels.org/    |
| XGBoost        | Gradient-boosted tree modeling                                     | https://xgboost.readthedocs.io/ |

The exact package versions used in the project are specified in `requirements.txt`.

The project uses these libraries through their documented APIs. No third-party source code is intentionally presented as original work by the authors. Any externally sourced or adapted code should be identified and attributed in the relevant source file.

**Dataset source:**

- Gomonov, A. (2019). *New York City Airbnb Open Data*. Kaggle. https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data
- Original data source: Inside Airbnb. https://insideairbnb.com/get-the-data/

The dataset is distributed under the CC0 1.0 Universal license as indicated by its Kaggle dataset metadata. Users should also consult the original data source and its applicable terms.

---
### Generative AI Declaration

Generative AI tools were used as supporting tools during the development of the project, including programming, literature exploration, manuscript preparation, visualization, and notebook review.

The tools and their respective roles are summarized below:

| AI tool        | Usage in the project                                                                                                                                                                      |
| -------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| ChatGPT        | Assisted with code development and debugging, manuscript drafting and revision, methodological discussions, brainstorming, and explanations of statistical and machine-learning concepts. |
| Claude         | Assisted with code development, manuscript revision, literature exploration and reference checking, and critical review of the manuscript.                                                |
| Dola           | Assisted with writing code for simple figures and visualizations.                                                                                                                         |
| DeepSeek       | Assisted with code development, debugging, and brainstorming.                                                                                                                             |
| NotebookLM     | Assisted with summarizing and organizing information from research papers and reference materials.                                                                                        |
| Gemini         | Assisted with manuscript drafting and writing-related tasks.                                                                                                                              |
| Grok           | Assisted with reviewing and inspecting notebook files; updating `reproduce.py` (synchronizing bootstrap replicates to 4000, integrating the injected-shift sensitivity analysis, removing outdated notebook-cell comments); revising the reproduction guide for consistency with the updated script. |
| GitHub Copilot | Assisted with reviewing and evaluating figures and visualizations.                                                                                                                        |

AI-generated suggestions, code, explanations, and manuscript content were reviewed by the project authors before being incorporated into the final work.

Code was checked through execution and comparison with the intended analytical procedures. Statistical methods, numerical results, and interpretations were reviewed against the implemented workflow and experimental outputs. Literature references were checked against their original publications or authoritative bibliographic records whenever applicable.

The authors retain responsibility for the accuracy, originality, reproducibility, and integrity of the submitted work. AI tools are acknowledged as supporting tools and are not considered authors of the research.

---
### Citing This Project

If this work is formally published, cite the corresponding publication using its official bibliographic information.

A BibTeX entry can be added after publication:

```bibtex
@inproceedings{airbnb_price_prediction,
author = {Author One and Author Two and Author Three},
title = {Airbnb Price Prediction in New York City Using Tree Ensembles and Segment-Level Price Deviation Analysis},
booktitle = {Conference Proceedings},
year = {YEAR},
pages = {XX--XX},
doi = {DOI}
}
```

The placeholders should be replaced with the actual publication metadata. Do not use this template as a finalized citation before publication.

---
### License

**Source code:** The repository does not currently include a `LICENSE` file. Therefore, no specific open-source license has been declared for the project's original source code.

**Dataset:** The Airbnb dataset is distributed under the CC0 1.0 Universal license according to the Kaggle dataset metadata. The original data source is Inside Airbnb.

**Third-party libraries:** All third-party libraries remain subject to their respective licenses and terms of use. Their licenses are not replaced or modified by this project.
