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

| No. | Full Name | Student ID | Main Responsibilities |
|---|---|---|---|
| 1 | Nguyễn Ngọc Đăng Khoa | SE210374 | Code |
| 2 | Huỳnh Gia Phú | SE210275 | Paper |

## Repository

- **GitHub Repository:** https://github.com/nguyenngocdangkhoa16122007-collab/airbnb-price-prediction
- **Paper Version:** [TODO: Final paper version]
- **Repository Version / Commit:** [TODO: Commit hash or release tag]
- **Last Updated:** [TODO: Date]

---
# 1. Quick Start

This section provides the fastest way to reproduce the main results
without manually running the research notebook.
The script reproduces the main model evaluation and analytical results. The supplementary injected-shift sensitivity analysis is not included in this quick reproduction script and must be reproduced separately using the research notebook.

## 1.1 Clone the repository

Open a terminal and run:

    git clone https://github.com/nguyenngocdangkhoa16122007-collab/airbnb-price-prediction.git
    cd airbnb-price-prediction

## 1.2 Set up the Python environment

Create and activate a virtual environment.

**Windows:**

    python -m venv .venv
    .venv\Scripts\activate

**macOS / Linux:**

    python3 -m venv .venv
    source .venv/bin/activate

Install the required packages:

    python -m pip install --upgrade pip
    pip install -r requirements.txt

## 1.3 Check the dataset

Ensure that the dataset is available at:

    data/raw/AB_NYC_2019.csv

The script expects the original NYC Airbnb Open Data file
containing 48,895 rows and 16 columns.

## 1.4 Run the reproduction script

From the repository root, execute:

    python reproduce.py

The script automatically loads and validates the dataset, performs
preprocessing, trains the six configured models across five
train/test splits, and generates the evaluation and analysis outputs.

No hyperparameter tuning is performed during this run. The script
uses the final hyperparameters specified in the source code.

## 1.5 Locate the generated results

After successful execution, the results are saved in:

- `outputs/tables/`: per-split results, feature importance,
  pricing-deviation results, and other detailed tables.
- `outputs/metrics/`: aggregated model metrics, cross-validation
  summaries, smearing factor, and run metadata.
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
│   ├── .gitkeep
│   └── Full_notebookk.ipynb
├── data/
│   ├── raw/
│   │   └── AB_NYC_2019.csv
│   └── processed/
│       └── .gitkeep
├── configs/
│   └── .gitkeep
├── outputs/
│   ├── tables/
│   │   └── .gitkeep
│   ├── figures/
│   │   └── .gitkeep
│   └── metrics/
│       └── .gitkeep
└── checkpoints/
    └── .gitkeep
```

## Directory Description

| Path | Description |
|---|---|
| `README.md` | Provides an overview of the project and basic repository instructions. |
| `requirements.txt` | Lists the Python packages and their pinned versions required by the project. |
| `reproduce.py` | [TODO: Describe the reproduction script after its implementation is finalized.] |
| `notebooks/` | Contains the Jupyter Notebook used for data analysis, model development, evaluation, and visualization. |
| `data/raw/` | Stores the original dataset used in the study. |
| `data/processed/` | Reserved for processed datasets generated during preprocessing. |
| `configs/` | Reserved for configuration files and experiment settings. |
| `outputs/tables/` | Stores generated result tables. |
| `outputs/figures/` | Stores generated figures and visualizations. |
| `outputs/metrics/` | Stores model evaluation metrics and related numerical results. |
| `checkpoints/` | Reserved for saved model checkpoints or trained model artifacts. |

The main analysis notebook is `notebooks/Full_notebookk.ipynb`. It contains the workflow used in the research, while the reproduction script is maintained separately.

---
# 3. Hardware & Software Requirements

## 3.1 Hardware Requirements

The project was developed and executed using Google Colab. 
The following hardware is recommended:

- CPU: Multi-core processor
- RAM: At least 8 GB (16 GB recommended)
- Storage: At least 2 GB of available space
- GPU: Not required; the experiments can run on CPU.

Actual execution time may vary depending on hardware and system load.

## 3.2 Software Requirements

- Python: A compatible Python 3 environment
- pip: Python package installer
- Git: Required to clone the repository

## 3.3 Python Dependencies

The project uses the following package versions:

| Package | Version |
|---|---:|
| numpy | 2.4.6 |
| pandas | 2.2.3 |
| duckdb | 1.3.2 |
| matplotlib | 3.10.0 |
| seaborn | 0.13.2 |
| scikit-learn | 1.6.1 |
| statsmodels | 0.15.0 |
| xgboost | 3.4.1 |

All dependencies are listed in `requirements.txt`.

---
# 4. Environment Setup

## 4.1 Clone the Repository

Open a terminal and run the following commands:

    git clone https://github.com/nguyenngocdangkhoa16122007-collab/airbnb-price-prediction.git
    cd airbnb-price-prediction

## 4.2 Create a Virtual Environment

A virtual environment is recommended to avoid conflicts with packages installed in other Python projects.

**Windows:**

    python -m venv .venv
    .venv\Scripts\activate

**macOS / Linux:**

    python3 -m venv .venv
    source .venv/bin/activate

## 4.3 Install Dependencies

After activating the virtual environment, install the required packages:

    python -m pip install --upgrade pip
    pip install -r requirements.txt

The `requirements.txt` file specifies the package versions used in this project.

## 4.4 Verify the Installation

Run the following command to check whether all required packages can be imported:

    python -c "import numpy, pandas, duckdb, matplotlib, seaborn, sklearn, statsmodels, xgboost; print('All packages imported successfully.')"

If the command finishes without an import error, the package installation is successful.

## 4.5 Verify the Dataset

Before running the reproduction script, ensure that the dataset is available at:

    data/raw/AB_NYC_2019.csv

The expected original dataset contains 48,895 rows and 16 columns.

If the file is missing, place the original `AB_NYC_2019.csv` dataset in the `data/raw/` directory.

## 4.6 Run the Reproduction Script

From the repository root directory, execute:

    python reproduce.py

The script will perform the main model evaluation and analyses, then save generated outputs in the following directories:

- `outputs/tables/`
- `outputs/metrics/`
- `outputs/figures/`

Note: The supplementary injected-shift sensitivity analysis is not included in `reproduce.py`. It must be reproduced separately using the research notebook.

---
# 5. Data

## 5.1 Dataset Source

This project uses the New York City Airbnb Open Data dataset by Gomonov (2019), redistributed through Kaggle from publicly available listings collected by the Inside Airbnb project.

- Dataset: New York City Airbnb Open Data
- Original source: https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data
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
- Standardizes numerical features using training-partition statistics for the linear models. Latitude and longitude are standardized for all six models.
- Encodes `neighbourhood` using a five-fold cross-fitted target encoder. The encoder is fitted using training data only, with unseen neighbourhoods mapped to the training-target mean.

**Geographic features**
- Calculates Haversine distances in kilometres to:
  - Times Square (40.75636, -73.98644)
  - Wall Street financial district (40.7074, -74.0113)

**Title-derived features**
Ten features are extracted from the listing title (`name`):
- `n_bed_name`: stated bedroom count; "studio" is encoded as zero, while -1 indicates that the count cannot be extracted.
- `n_bed_name_missing`: indicates whether the bedroom count could not be extracted.
- Eight binary keyword indicators: `luxury`, `penthouse`, `loft`, `townhouse`, `duplex`, `cozy`, `bunk`, and `hostel`.

Raw host names and listing titles are not directly used as predictors. Only the title-derived features are retained. The raw target and price-derived quantities are excluded from the predictor variables.

## 5.4 Data Leakage Prevention

All data-dependent preprocessing steps are fitted using the relevant training partition only. This includes:

- 99th-percentile capping thresholds;
- numerical scaling statistics;
- neighbourhood target encoding;
- the sentinel value for listings without a recorded review date.

The fitted transformations are then applied unchanged to the corresponding validation or test partition.

This procedure prevents information from validation and test observations from influencing preprocessing.

## 5.5 Data Integrity

The SHA256 checksum of the raw dataset is provided below to help verify file integrity.

- File: `data/raw/AB_NYC_2019.csv`
- Algorithm: SHA256
- Checksum: `e420db40ff10fcb40efc1b5b1648ee0b18a48f4e4537155cecc59fe95d18783a`

To calculate the checksum on Windows PowerShell, run:

    Get-FileHash .\data\raw\AB_NYC_2019.csv -Algorithm SHA256

On macOS or Linux, run:

    sha256sum data/raw/AB_NYC_2019.csv

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

Randomized Search was implemented using scikit-learn's `ParameterSampler`, with 25 sampled configurations for each model:

- Random Forest
- XGBoost
- Gradient Boosting

The sampling process used `random_state=42`.

All candidate configurations were evaluated using five-fold stratified cross-validation on the training data. The mean MAE in the original USD price scale was used as the primary selection metric. The final configuration for each tuned model was selected based on the lowest mean cross-validation MAE.

The held-out test set was not used for hyperparameter selection.

## 6.3 Search Spaces

The search spaces for the tree-based models were:

**Random Forest**

| Hyperparameter | Candidate values |
|---|---|
| `n_estimators` | {300, 500} |
| `max_depth` | {8, 10, 12, 15, 18} |
| `min_samples_leaf` | {1, 3, 5, 10, 20} |
| `max_features` | {sqrt, 0.5, 0.7} |

**XGBoost**

| Hyperparameter | Candidate values |
|---|---|
| `n_estimators` | {300, 500, 700, 1000} |
| `max_depth` | {6, 8, 10, 12} |
| `learning_rate` | {0.01, 0.03, 0.05} |
| `subsample` | {0.6, 0.8, 1.0} |
| `colsample_bytree` | {0.6, 0.8, 1.0} |
| `min_child_weight` | {1, 3, 5} |
| `reg_lambda` | {1, 2, 5} |

**Gradient Boosting**

| Hyperparameter | Candidate values |
|---|---|
| `n_estimators` | {100, 150, 200, 300} |
| `max_depth` | {3, 4, 5, 6, 8} |
| `learning_rate` | {0.01, 0.03, 0.05, 0.1} |
| `subsample` | {0.6, 0.8, 1.0} |
| `min_samples_leaf` | {5, 10, 20, 50} |

## 6.4 Final Model Hyperparameters

The following configurations are fixed in `reproduce.py`:

| Model | Final hyperparameters |
|---|---|
| Linear Regression | Default scikit-learn configuration |
| Ridge | `alpha=1.0` |
| Lasso | `alpha=0.0001` |
| Random Forest | `n_estimators=500`, `max_depth=18`, `min_samples_leaf=1`, `max_features=0.5` |
| XGBoost | `n_estimators=1000`, `max_depth=6`, `learning_rate=0.03`, `subsample=0.8`, `colsample_bytree=0.6`, `min_child_weight=1`, `reg_lambda=5` |
| Gradient Boosting | `n_estimators=300`, `max_depth=8`, `learning_rate=0.1`, `subsample=1.0`, `min_samples_leaf=5` |

XGBoost was selected as the best-performing model based on mean cross-validation MAE.

## 6.5 Reproduction Notes

The reproduction script does not repeat the original hyperparameter search. Instead, it uses the fixed hyperparameters listed in Section 6.4.

The original search-result DataFrames were not exported as `outputs/hpo_results.csv`. Therefore, this file is not included in the current reproduction package.

The preprocessing pipeline is implemented directly in `reproduce.py`. Its data-dependent transformations are fitted using the relevant training partition, as described in Section 5.

One-hot encoding was configured differently for linear and tree-based models. Linear models used reference-category encoding (drop_first=True) to avoid perfect multicollinearity, whereas tree-based models retained all dummy variables (drop_first=False). Latitude and longitude were standardized during preprocessing for all models. Additionally, linear models used a StandardScaler within the modeling pipeline to standardize the complete feature matrix.

All candidate configurations were evaluated using five-fold stratified cross-validation, with mean MAE on the original USD price scale as the primary selection metric. Although the target variable was log-transformed during model training, predictions were transformed back to the original price scale before MAE calculation.
---
## 7. Training

### 7.1. Training Procedure

All six regression models reported in the study were trained using the `reproduce.py` script. The script uses the final hyperparameter configurations specified directly in the source code; therefore, hyperparameter tuning is not repeated during reproduction.

To execute the training and reproduction pipeline, run the following command from the repository root:

    python reproduce.py

The pipeline evaluates the six models: Linear Regression, Ridge Regression, Lasso Regression, Random Forest, XGBoost, and Gradient Boosting. Training and evaluation are repeated using five random seeds: 42, 1, 7, 123, and 2024. Each model is trained using the corresponding training partition, with preprocessing fitted only on the relevant training data to prevent information leakage.

### 7.2. Model Configuration

All models use the final hyperparameters documented in Section 6. The reproduction script loads these configurations directly and does not perform additional hyperparameter searches.

### 7.3. Output Artifacts

Generated results are saved in the following directories:

- `outputs/tables/`: generated result tables.
- `outputs/metrics/`: evaluation metrics and related numerical outputs.
- `outputs/figures/`: generated visualizations.

The script does not rely on separately saved deep-learning checkpoints, epoch-based training logs, or TensorBoard/W&B tracking.

### 7.4. Computational Resources

The experiments were conducted using Google Colab. The actual execution environment and runtime depend on the allocated Colab resources.

- Environment: Google Colab.
- CPU: x86_64.
- RAM: 12.67.
- GPU: Not used / unavailable.
- Total execution time: [record actual runtime].

---
