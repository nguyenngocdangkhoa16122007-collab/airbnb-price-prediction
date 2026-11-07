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
│   └── ADYm_analysis.ipynb
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

# 5. Data

## 5.1 Dataset Information

This study uses the **New York City Airbnb Open Data (2019)** dataset originally published by Gomonov (2019) on Kaggle.

- **Dataset name:** New York City Airbnb Open Data
- **Original source:** Kaggle
- **Dataset author:** Gomonov
- **Dataset URL:** https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data
- **File used:** `AB_NYC_2019.csv`
- **Original size:** 48,895 rows × 16 columns
- **File location in this repository:** `data/raw/AB_NYC_2019.csv`
- **Dataset license:** [TODO: Verify and specify the original dataset license]
- **File checksum (SHA-256):** [TODO: Add checksum]

## 5.2 Dataset Description

The dataset contains Airbnb listing information for New York City in 2019. It includes listing characteristics, host-related information, geographical attributes, room types, review information, availability, and listing prices.

The original dataset contains 48,895 observations and 16 columns.

The target variable in this study is the listing price (`price`).

## 5.3 Data Preprocessing

The following preprocessing operations are applied during the analysis:

1. **Invalid price removal:** 11 observations with a price of zero are removed.
2. **Price capping:** The price variable is capped at its 99th percentile to reduce the influence of extreme high-price observations.
3. **Target transformation:** The target variable is transformed using `log1p(price)` for model training.
4. **Feature preparation:** Categorical variables and numerical variables are processed according to the requirements of each model.
5. **Target encoding:** Neighbourhood-related target encoding is performed using training data only, with cross-fitting to reduce target leakage.
6. **Text-derived features:** Additional listing features are extracted from the listing title (`name`) using regular expressions.

Preprocessing operations that learn parameters from data are fitted using training data only to prevent information leakage from the test set.

## 5.4 Data Access and Usage

The original dataset is included in the repository at:

```text
data/raw/AB_NYC_2019.csv
```

Users can obtain the dataset by cloning the repository. No separate manual download is required if the dataset file remains available in the repository.

**Dataset modification before repository upload:** [TODO: Confirm whether the uploaded CSV is identical to the original Kaggle file.]

**Processed dataset storage:** [TODO: Specify whether processed data are saved to `data/processed/` or generated dynamically during notebook execution.]

## 5.5 Data Integrity

To help verify that users are working with the same dataset version, the SHA-256 checksum of the original CSV file can be provided below.

```text
SHA-256: [TODO: Insert checksum]
```

The dataset should be treated as the original input to the analysis. Preprocessing and feature engineering are performed by the project workflow rather than by manually modifying the raw CSV file.
