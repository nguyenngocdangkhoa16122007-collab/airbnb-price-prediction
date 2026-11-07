
# ============================================================
# 1. ENVIRONMENT SETUP
# ============================================================

import os
import re
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, GroupKFold
from sklearn.preprocessing import StandardScaler, TargetEncoder
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

# Project paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(
    PROJECT_ROOT, "data", "raw", "AB_NYC_2019.csv"
)

OUTPUT_PATH = os.path.join(PROJECT_ROOT, "outputs")

TABLE_PATH = os.path.join(OUTPUT_PATH, "tables")
FIGURE_PATH = os.path.join(OUTPUT_PATH, "figures")
METRIC_PATH = os.path.join(OUTPUT_PATH, "metrics")

# Create output directories
for path in [TABLE_PATH, FIGURE_PATH, METRIC_PATH]:
    os.makedirs(path, exist_ok=True)

# Check dataset
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}"
    )

print("Environment setup completed.")
print("Dataset:", DATA_PATH)
print("Output:", OUTPUT_PATH)
