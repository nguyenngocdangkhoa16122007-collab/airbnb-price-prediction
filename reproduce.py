#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
reproduce.py - Quick Reproduction of the NYC Airbnb price-prediction study
==========================================================================

Source of truth: the research notebook (Full_notebookk.ipynb).

What this script reproduces (no hyper-parameter tuning is run):
    RQ2   6 models x 5 train/test splits, MAE/RMSE/R2 (log, capped, raw),
          paired MAE comparison XGBoost vs Gradient Boosting / Random Forest.
    RQ3   Model-relative pricing deviation D = log(actual / predicted) per
          (neighbourhood_group x room_type) segment: host-clustered bootstrap
          (B = 4000), Holm correction, +-5 % practical threshold,
          direction consistency in >= 3 of 5 splits.
    Host  Multi-host vs single-host comparison (overall and within each room
          type) with host-clustered bootstrap.
    RQ1   Lasso coefficients, Random Forest impurity importance and XGBoost
          gain importance (split seed 42).
    CV    StratifiedKFold vs GroupKFold(host_id) MAE for XGBoost, Random
          Forest and Gradient Boosting (train part of split seed 42).
    Calib Smearing factor (out-of-fold residuals, split seed 42) and decile
          calibration: median prediction (naive back-transform) and mean
          prediction (smearing-corrected).

Run from the repository root (or anywhere):

    python reproduce.py

Outputs are written to outputs/tables, outputs/metrics and outputs/figures.
"""

# ============================================================
# 1. ENVIRONMENT SETUP AND IMPORTS
# ============================================================

import gc
import json
import os
import platform
import re
import sys
import time

# Fail early, with a readable message, if a required package is missing.
_REQUIRED = {
    "numpy": "numpy",
    "pandas": "pandas",
    "matplotlib": "matplotlib",
    "seaborn": "seaborn",
    "sklearn": "scikit-learn>=1.3",   # TargetEncoder needs scikit-learn >= 1.3
    "statsmodels": "statsmodels",
    "xgboost": "xgboost",
}
_missing = []
for _module, _pip_name in _REQUIRED.items():
    try:
        __import__(_module)
    except ImportError:
        _missing.append(_pip_name)
if _missing:
    sys.exit(
        "Missing required packages: " + ", ".join(_missing) + "\n"
        "Install them with:  pip install -r requirements.txt"
    )

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")                       # headless backend (no display needed)
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns

try:
    from sklearn.preprocessing import StandardScaler, TargetEncoder
except ImportError:
    sys.exit("scikit-learn >= 1.3 is required (sklearn.preprocessing.TargetEncoder).")
from sklearn.model_selection import train_test_split, StratifiedKFold, GroupKFold
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from statsmodels.stats.multitest import multipletests
from xgboost import XGBRegressor

import sklearn
import statsmodels
import xgboost

# ============================================================
# 2. PROJECT PATH CONFIGURATION (relative to the repository root)
# ============================================================

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "AB_NYC_2019.csv")

OUTPUT_PATH = os.path.join(PROJECT_ROOT, "outputs")
TABLE_PATH = os.path.join(OUTPUT_PATH, "tables")
METRIC_PATH = os.path.join(OUTPUT_PATH, "metrics")
FIGURE_PATH = os.path.join(OUTPUT_PATH, "figures")

for _path in (TABLE_PATH, METRIC_PATH, FIGURE_PATH):
    os.makedirs(_path, exist_ok=True)

# Global seed (notebook cell 3). Every estimator/split below also receives an
# explicit random_state, exactly as in the notebook.
SEED = 42
np.random.seed(SEED)

# Figure style that was active in the notebook when the paper figures were made
# (notebook cells 8 and 15: seaborn whitegrid theme, figure.dpi = 110).
sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams.update({"figure.dpi": 110, "figure.figsize": (16, 6)})

_T0 = time.time()


def log(msg):
    """Print a message with the elapsed time since the script started."""
    print(f"[{time.time() - _T0:8.1f}s] {msg}", flush=True)


def save_table(df_, name, folder):
    path = os.path.join(folder, name)
    df_.to_csv(path, index=False)
    log(f"saved {os.path.relpath(path, PROJECT_ROOT)}")


# ============================================================
# 3. DATASET LOADING AND VALIDATION
#    (notebook cells 3, 49, 65, 66)
# ============================================================

REQUIRED_COLUMNS = [
    "id", "name", "host_id", "host_name", "neighbourhood_group",
    "neighbourhood", "latitude", "longitude", "room_type", "price",
    "minimum_nights", "number_of_reviews", "last_review",
    "reviews_per_month", "calculated_host_listings_count", "availability_365",
]


def load_and_clean_dataset(path):
    """Load AB_NYC_2019.csv and apply the notebook's cleaning steps."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found: {path}\n"
            "Place the original Kaggle file AB_NYC_2019.csv in data/raw/."
        )

    raw = pd.read_csv(path)
    log(f"loaded dataset: {raw.shape[0]:,} rows x {raw.shape[1]} columns")

    missing_cols = [c for c in REQUIRED_COLUMNS if c not in raw.columns]
    if missing_cols:
        raise ValueError(f"Dataset is missing required columns: {missing_cols}")
    if raw.shape != (48895, 16):
        print(f"WARNING: expected shape (48895, 16) as in the notebook, got {raw.shape}.")

    # Cell 49 / 65: drop price == 0, reviews_per_month NaN -> 0, reset the index.
    # (The notebook also adds price_capped / is_unavailable / has_review here;
    #  all three are re-created inside preprocess_base(), so they are not needed.)
    cleaned = raw[raw["price"] > 0].copy()
    cleaned["reviews_per_month"] = cleaned["reviews_per_month"].fillna(0)
    cleaned = cleaned.reset_index(drop=True)
    log(f"cleaning: removed {len(raw) - len(cleaned)} rows with price <= 0 "
        f"-> {len(cleaned):,} rows")
    return cleaned


df_clean = load_and_clean_dataset(DATA_PATH)


# ============================================================
# 4. DATA PREPROCESSING AND FEATURE ENGINEERING
#    (notebook cells 69, 70, 71 - unchanged)
# ============================================================

SEEDS = [42, 1, 7, 123, 2024]
TIMES_SQUARE = (40.75636, -73.98644)
WALL_STREET = (40.7074, -74.0113)
FIXED_REF_DATE = pd.Timestamp('2019-12-31')


def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))


KEYWORDS = ['luxury', 'penthouse', 'loft', 'townhouse', 'duplex', 'cozy', 'bunk', 'hostel']


def bedrooms_from_name(s):
    s = str(s).lower()
    if 'studio' in s:
        return 0
    m = re.search(r'(\d+)\s*(?:br\b|bd\b|bed(?:room)?s?\b)', s)
    if m and int(m.group(1)) < 10:
        return int(m.group(1))
    return np.nan


def add_name_features(d):
    d = d.copy()
    d['n_bed_name'] = d['name'].map(bedrooms_from_name)
    d['n_bed_name_missing'] = d['n_bed_name'].isna().astype(int)
    d['n_bed_name'] = d['n_bed_name'].fillna(-1)  # -1 = "could not extract", model learns it
    for kw in KEYWORDS:
        d[f'kw_{kw}'] = d['name'].astype(str).str.lower().str.contains(kw, na=False).astype(int)
    return d


NAME_FEATURE_COLS = ['n_bed_name', 'n_bed_name_missing'] + [f'kw_{kw}' for kw in KEYWORDS]


def preprocess_base(d, p99_price, p99_min_nights, scaler_latlon, fit_scaler, drop_first, date_sentinel):
    d = d.copy()
    d['price_capped'] = d['price'].clip(upper=p99_price)
    d['price_log'] = np.log1p(d['price_capped'])

    d = pd.get_dummies(d, columns=['neighbourhood_group', 'room_type'], prefix=['ng', 'room'], drop_first=drop_first)

    if fit_scaler:
        scaler_latlon.fit(d[['latitude', 'longitude']])
    d[['latitude_scaled', 'longitude_scaled']] = scaler_latlon.transform(d[['latitude', 'longitude']])

    d['number_of_reviews_log'] = np.log1p(d['number_of_reviews'])
    d['reviews_per_month'] = d['reviews_per_month'].fillna(0)
    d['reviews_per_month_log'] = np.log1p(d['reviews_per_month'])
    d['has_review'] = (d['number_of_reviews'] > 0).astype(int)

    # Sentinel is a fixed PARAMETER computed from train (shared by train/val/test).
    last_review_dt = pd.to_datetime(d['last_review'])
    d['days_since_last_review'] = (FIXED_REF_DATE - last_review_dt).dt.days
    d['days_since_last_review'] = d['days_since_last_review'].fillna(date_sentinel)

    d['minimum_nights_capped'] = d['minimum_nights'].clip(upper=p99_min_nights)
    d['minimum_nights_log'] = np.log1p(d['minimum_nights_capped'])
    d['is_unavailable'] = (d['availability_365'] == 0).astype(int)
    d['host_listings_log'] = np.log1p(d['calculated_host_listings_count'])

    d['dist_times_square'] = haversine(d['latitude'], d['longitude'], *TIMES_SQUARE)
    d['dist_wall_street'] = haversine(d['latitude'], d['longitude'], *WALL_STREET)

    d = add_name_features(d)
    return d


def prepare_fold_fixed(tr_raw, va_raw, drop_first=False, use_name_features=True):
    p99_price = tr_raw['price'].quantile(0.99)
    p99_min_nights = tr_raw['minimum_nights'].quantile(0.99)
    scaler = StandardScaler()

    # Sentinel computed ONCE from train, shared by train and val/test.
    tr_recency = (FIXED_REF_DATE - pd.to_datetime(tr_raw['last_review'])).dt.days
    date_sentinel = tr_recency.max() + 1 if tr_recency.notna().any() else 9999

    tr_p = preprocess_base(tr_raw, p99_price, p99_min_nights, scaler, fit_scaler=True, drop_first=drop_first, date_sentinel=date_sentinel)
    va_p = preprocess_base(va_raw, p99_price, p99_min_nights, scaler, fit_scaler=False, drop_first=drop_first, date_sentinel=date_sentinel)

    # Cross-fitted target encoding of `neighbourhood` (fit_transform on train = 5-fold cross-fitting).
    te = TargetEncoder(target_type='continuous', smooth='auto', cv=5, random_state=42)
    tr_p['neighbourhood_target_enc'] = te.fit_transform(tr_raw[['neighbourhood']], tr_p['price_log']).ravel()
    va_p['neighbourhood_target_enc'] = te.transform(va_raw[['neighbourhood']]).ravel()

    cols = [c for c in tr_p.columns if c.startswith(('ng_', 'room_'))] + [
        'latitude_scaled', 'longitude_scaled', 'number_of_reviews_log',
        'reviews_per_month_log', 'has_review', 'days_since_last_review',
        'minimum_nights_log', 'availability_365', 'is_unavailable', 'host_listings_log',
        'neighbourhood_target_enc', 'dist_times_square', 'dist_wall_street',
    ]
    if use_name_features:
        cols += NAME_FEATURE_COLS

    X_tr = tr_p.reindex(columns=cols, fill_value=0)
    X_va = va_p.reindex(columns=cols, fill_value=0)
    return X_tr, tr_p['price_log'], X_va, va_p['price_log'], va_raw['price'].values


# ============================================================
# 5. MODEL INITIALISATION - FINAL SELECTED HYPER-PARAMETERS
#    (notebook cells 90 and 91; NO tuning is performed here)
# ============================================================

best_params_rf = {'n_estimators': 500, 'max_depth': 18, 'min_samples_leaf': 1, 'max_features': 0.5}
best_params_xgb = {'n_estimators': 1000, 'max_depth': 6, 'learning_rate': 0.03, 'subsample': 0.8,
                   'colsample_bytree': 0.6, 'min_child_weight': 1.0, 'reg_lambda': 5.0}
best_params_gbm = {'n_estimators': 300, 'max_depth': 8, 'learning_rate': 0.1, 'subsample': 1.0,
                   'min_samples_leaf': 5}

LINEAR_MODELS = {'Linear Regression', 'Ridge', 'Lasso'}
best_alpha_lasso = 0.0001
best_alpha_ridge = 1.0

MODEL_NAMES = ['Linear Regression', 'Ridge', 'Lasso', 'Random Forest', 'XGBoost', 'Gradient Boosting']


def build_model_fn(name):
    if name == 'Linear Regression':
        return lambda: Pipeline([('scaler', StandardScaler()), ('model', LinearRegression())])
    if name == 'Ridge':
        return lambda: Pipeline([('scaler', StandardScaler()), ('model', Ridge(alpha=best_alpha_ridge))])
    if name == 'Lasso':
        return lambda: Pipeline([('scaler', StandardScaler()), ('model', Lasso(alpha=best_alpha_lasso, random_state=42, max_iter=50000))])
    if name == 'Random Forest':
        return lambda: RandomForestRegressor(**best_params_rf, n_jobs=-1, random_state=42)
    if name == 'XGBoost':
        return lambda: XGBRegressor(**best_params_xgb, objective='reg:squarederror', tree_method='hist', n_jobs=-1, random_state=42)
    if name == 'Gradient Boosting':
        return lambda: GradientBoostingRegressor(**best_params_gbm, random_state=42)
    raise ValueError(f"Unknown model name: {name}")


# ============================================================
# 6. TRAINING AND EVALUATION ON THE 5 FIXED SPLITS
#    RQ2 table (notebook cell 92) + per-split XGBoost predictions for
#    RQ3 / host-type analyses (notebook cell 97)
# ============================================================

def evaluate_once(model, X_tr, y_tr, X_te, price_te_raw, p99_price_train):
    model.fit(X_tr, y_tr)
    pred_log = model.predict(X_te)
    pred_raw = np.maximum(np.expm1(pred_log), 0)
    mae = mean_absolute_error(price_te_raw, pred_raw)
    rmse = np.sqrt(mean_squared_error(price_te_raw, pred_raw))
    r2_log = r2_score(np.log1p(price_te_raw), pred_log)
    r2_raw = r2_score(price_te_raw, pred_raw)
    actual_capped = np.minimum(price_te_raw, p99_price_train)
    pred_capped = np.minimum(pred_raw, p99_price_train)
    r2_capped = r2_score(actual_capped, pred_capped)
    return mae, rmse, r2_log, r2_capped, r2_raw


# Columns of each test split that RQ3 / host-type analyses need.
_TEST_COLS = ["neighbourhood_group", "room_type", "host_id", "calculated_host_listings_count"]

all_results = []     # RQ2: one row per (model, seed)
split_models = {}    # RQ3 / host type: seed -> {'test', 'pred_log', 'price_te_raw'}

# Kept from the seed-42 split only (SEEDS[0]): RQ1 importances/coefficients are
# read from the RQ2 fits (identical configuration to notebook cell 123), and the
# train part is reused by the GroupKFold / smearing section.
rq1_store = {}
train_full0 = None

log("training 6 models on each of the 5 splits (this is the slow step)")
for seed in SEEDS:
    price_bins = pd.qcut(df_clean['price'], q=10, labels=False, duplicates='drop')
    train_full, test = train_test_split(df_clean, test_size=0.15, random_state=seed, stratify=price_bins)
    p99_price_train = train_full['price'].quantile(0.99)

    # prepare_fold_fixed is deterministic, so the two variants (one-hot with /
    # without drop_first) are built once per split and shared by the models.
    prepared = {
        drop_first: prepare_fold_fixed(train_full, test, drop_first=drop_first, use_name_features=True)
        for drop_first in (False, True)
    }

    xgb_pred_log_rs42 = None
    for model_name in MODEL_NAMES:
        drop_first = model_name in LINEAR_MODELS
        X_tr, y_tr, X_te, y_te, price_te_raw = prepared[drop_first]
        model = build_model_fn(model_name)()
        mae, rmse, r2_log, r2_capped, r2_raw = evaluate_once(model, X_tr, y_tr, X_te, price_te_raw, p99_price_train)
        all_results.append({'model': model_name, 'seed': seed, 'mae': mae, 'rmse': rmse,
                            'r2_log': r2_log, 'r2_capped': r2_capped, 'r2_raw': r2_raw})
        if model_name == 'XGBoost':
            xgb_pred_log_rs42 = model.predict(X_te)
        if seed == 42:   # RQ1 (notebook cells 121 / 123 use the seed-42 split)
            if model_name == 'Random Forest':
                rq1_store['RF'] = pd.Series(model.feature_importances_, index=X_tr.columns)
            elif model_name == 'XGBoost':
                rq1_store['XGB'] = pd.Series(model.feature_importances_, index=X_tr.columns)  # gain
            elif model_name == 'Lasso':
                rq1_store['Lasso'] = pd.Series(model.named_steps['model'].coef_, index=X_tr.columns)
        del model

    # RQ3 / host-type predictions: XGBoost fitted with random_state = split seed
    # (notebook cell 97). For seed 42 this is the identical configuration to the
    # RQ2 XGBoost fit above, so its predictions are reused instead of re-fitting.
    X_tr, y_tr, X_te, y_te, price_te_raw = prepared[False]
    if seed == 42:
        pred_log_te = xgb_pred_log_rs42
    else:
        model = XGBRegressor(**best_params_xgb, objective='reg:squarederror', tree_method='hist',
                             n_jobs=-1, random_state=seed)
        model.fit(X_tr, y_tr)
        pred_log_te = model.predict(X_te)
        del model

    split_models[seed] = {
        'test': test[_TEST_COLS].copy(),
        'pred_log': pred_log_te,
        'price_te_raw': price_te_raw,
    }

    if seed == 42:
        train_full0 = train_full

    del prepared, X_tr, y_tr, X_te, y_te, train_full, test
    gc.collect()
    log(f"finished split seed={seed}")

all_results_df = pd.DataFrame(all_results)


# ============================================================
# 7. RQ2 - MODEL EVALUATION AND COMPARISON
#    (notebook cells 92-95)
# ============================================================

log("RQ2: aggregating results")

# ---- 7.1 Six-model table across the five splits (cell 92 / 93) ----
summary = all_results_df.groupby('model')[['mae', 'rmse', 'r2_log', 'r2_capped', 'r2_raw']].agg(['mean', 'std'])
summary = summary.sort_values(('mae', 'mean'))
summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
summary = summary.reset_index()

print("\n=== RQ2: model performance, mean / std across 5 splits ===")
print(summary.round(4).to_string(index=False))

save_table(all_results_df, "rq2_model_results_by_seed.csv", TABLE_PATH)
save_table(summary, "rq2_model_performance_summary.csv", METRIC_PATH)

# ---- 7.2 Paired MAE comparison across seeds (cells 94 / 95) ----
pairs = [
    ('XGBoost', 'Gradient Boosting'),
    ('XGBoost', 'Random Forest')
]

models_needed = {'XGBoost', 'Gradient Boosting', 'Random Forest'}
df_pair = all_results_df[all_results_df['model'].isin(models_needed)].copy()

if df_pair.duplicated(['model', 'seed']).any():
    raise ValueError("More than one row for the same model and seed.")

mae_wide = df_pair.pivot(index='seed', columns='model', values='mae')

expected_models = ['XGBoost', 'Gradient Boosting', 'Random Forest']
missing_models = set(expected_models) - set(mae_wide.columns)
if missing_models:
    raise ValueError(f"Missing results for models: {missing_models}")
if mae_wide[expected_models].isna().any().any():
    raise ValueError("MAE is missing for one or more seeds.")

mae_wide = mae_wide.sort_index()

print("\n=== MAE per seed (USD) ===")
print(mae_wide.round(2).to_string())

pair_rows = []
for winner_model, competitor_model in pairs:
    # Positive => XGBoost has lower MAE
    delta = mae_wide[competitor_model] - mae_wide[winner_model]
    improvement_pct = delta / mae_wide[competitor_model] * 100

    for seed in mae_wide.index:
        pair_rows.append({
            'comparison': f'{winner_model} vs {competitor_model}',
            'seed': seed,
            'mae_xgboost': mae_wide.loc[seed, 'XGBoost'],
            'mae_competitor': mae_wide.loc[seed, competitor_model],
            'delta_mae_comp_minus_xgb': delta.loc[seed],
            'xgb_mae_improvement_pct': improvement_pct.loc[seed]
        })

pair_results_df = pd.DataFrame(pair_rows)

summary_rows = []
for winner_model, competitor_model in pairs:
    sub = pair_results_df[pair_results_df['comparison'] == f'{winner_model} vs {competitor_model}']
    delta = sub['delta_mae_comp_minus_xgb']
    improvement = sub['xgb_mae_improvement_pct']

    summary_rows.append({
        'comparison': f'{winner_model} vs {competitor_model}',
        'n_seeds': len(sub),
        'mean_mae_xgb': sub['mae_xgboost'].mean(),
        'mean_mae_competitor': sub['mae_competitor'].mean(),
        'mean_delta_mae': delta.mean(),
        'std_delta_mae': delta.std(ddof=1),
        'mean_improvement_pct': improvement.mean(),
        'seeds_xgb_better': int((delta > 0).sum()),
        'seeds_tied': int((delta == 0).sum()),
        'seeds_xgb_worse': int((delta < 0).sum())
    })

pair_summary_df = pd.DataFrame(summary_rows)

print("\n=== RQ2: paired MAE comparison (summary) ===")
print(pair_summary_df.round(4).to_string(index=False))

save_table(pair_results_df, "rq2_paired_mae_by_seed.csv", TABLE_PATH)
save_table(pair_summary_df, "rq2_paired_mae_summary.csv", METRIC_PATH)

# ---- 7.3 Figure: model comparison (cell 93) ----
plot_summary = summary.copy()
model_order = ['XGBoost', 'Gradient Boosting', 'Random Forest',
               'Linear Regression', 'Ridge', 'Lasso']
plot_summary['model'] = pd.Categorical(plot_summary['model'], categories=model_order, ordered=True)
# reset_index keeps label-based lookups below aligned with the display order
# (identical to the notebook whenever the MAE ranking equals model_order).
plot_summary = plot_summary.sort_values('model').reset_index(drop=True)

colors = ['#1a365d' if m == 'XGBoost' else '#b8b8b8' for m in plot_summary['model']]

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.dpi': 300,
})

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()
metrics = [
    ('mae_mean', 'mae_std', 'MAE'),
    ('rmse_mean', 'rmse_std', 'RMSE'),
    ('r2_log_mean', 'r2_log_std', 'R² Log'),
    ('r2_raw_mean', 'r2_raw_std', 'R² Raw'),
]

x = np.arange(len(plot_summary))
width = 0.6

for idx, (mean_col, std_col, title) in enumerate(metrics):
    ax = axes[idx]
    bars = ax.bar(x, plot_summary[mean_col], width,
                  yerr=plot_summary[std_col], capsize=5,
                  color=colors, edgecolor='#333333', linewidth=0.5)

    bars[0].set_edgecolor('#0f2038')
    bars[0].set_linewidth(1.2)

    ax.set_title(title, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(plot_summary['model'], rotation=30, ha='right')
    ax.grid(axis='y', linestyle=':', alpha=0.5)

    for i, v in enumerate(plot_summary[mean_col]):
        ax.text(i, v + (plot_summary[std_col][i] * 0.5), f'{v:.4f}',
                ha='center', va='bottom', fontsize=8,
                fontweight='bold' if plot_summary['model'][i] == 'XGBoost' else 'normal')

fig.suptitle('Model performance comparison (mean ± standard deviation across 5 splits)',
             fontsize=14, fontweight='bold', y=0.98)
legend_elements = [Patch(facecolor='#1a365d', edgecolor='#0f2038', label='XGBoost — Best Model'),
                   Patch(facecolor='#b8b8b8', label='Other Models')]
fig.legend(handles=legend_elements, loc='upper center', ncol=2,
           bbox_to_anchor=(0.5, 0.94), framealpha=1.0)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(os.path.join(FIGURE_PATH, 'model_comparison.png'), dpi=300, bbox_inches='tight')
plt.close(fig)
log("saved outputs/figures/model_comparison.png")


# ============================================================
# 8. RQ3 - PRICING DEVIATION ANALYSIS (HOST-CLUSTER BOOTSTRAP)
#    (notebook cells 106-110)
# ============================================================
# D = log(actual / predicted); estimand = median(D_segment) - median(D_overall);
# the bootstrap resamples hosts over the WHOLE test set and recomputes both medians.

B_BOOT = 4000
MIN_SEGMENT_N = 25
PRACTICAL_THRESHOLD = np.log(1.05)
MIN_VALID_BOOTSTRAP_RATE = 0.99


def cluster_bootstrap_segment_gap(
    df,
    segment_mask,
    value_col="D",
    group_col="host_id",
    B=4000,
    seed=42,
):
    """
    Host-clustered bootstrap over the WHOLE test set.

    In each replicate:
      1. Resample host_id with replacement.
      2. Recompute the overall median of D.
      3. Recompute the segment median of D.
      4. theta* = median_segment* - median_overall*.

    Returns theta_hat, boot_theta, valid_rate, n_hosts.
    """
    rng = np.random.default_rng(seed)

    required = [value_col, group_col]
    if any(c not in df.columns for c in required):
        raise ValueError(f"Missing required columns: {required}")

    if df[group_col].isna().any():
        raise ValueError(f"{group_col} contains missing values; handle them before bootstrapping.")

    values = df[value_col].to_numpy(dtype=float)
    mask = np.asarray(segment_mask, dtype=bool)

    if len(mask) != len(df):
        raise ValueError("segment_mask length does not match df.")

    if not np.isfinite(values).all():
        raise ValueError(f"{value_col} contains NaN or infinite values.")

    if mask.sum() == 0:
        raise ValueError("The segment has no observations.")

    theta_hat = (
        np.median(values[mask])
        - np.median(values)
    )

    # Keep every listing of a host together: if a host is drawn several times,
    # all of its listings appear several times in the bootstrap sample.
    host_indices = [
        np.flatnonzero(df[group_col].to_numpy() == host)
        for host in pd.unique(df[group_col])
    ]

    n_hosts = len(host_indices)
    if n_hosts < 2:
        raise ValueError("Not enough unique hosts to bootstrap.")

    boot_theta = np.full(B, np.nan)

    for b in range(B):
        sampled_hosts = rng.integers(
            low=0,
            high=n_hosts,
            size=n_hosts,
        )

        sampled_indices = np.concatenate([
            host_indices[j] for j in sampled_hosts
        ])

        sampled_values = values[sampled_indices]
        sampled_mask = mask[sampled_indices]

        # The segment can be absent from a replicate when very few hosts represent it.
        if not sampled_mask.any():
            continue

        median_overall_b = np.median(sampled_values)
        median_segment_b = np.median(
            sampled_values[sampled_mask]
        )

        boot_theta[b] = median_segment_b - median_overall_b

    valid = np.isfinite(boot_theta)
    valid_rate = valid.mean()
    boot_theta = boot_theta[valid]

    return theta_hat, boot_theta, valid_rate, n_hosts


def basic_bootstrap_ci_and_pvalue(
    theta_hat,
    boot_theta,
    alpha=0.05,
):
    """
    Basic bootstrap CI and approximate p-value from the centred bootstrap error
    distribution.

      CI:      [2*theta_hat - q_(1-alpha/2), 2*theta_hat - q_(alpha/2)]
      p-value: compares |theta_hat| with |theta* - theta_hat| (Monte-Carlo
               corrected so that p is never exactly 0).
    """
    boot_theta = np.asarray(boot_theta, dtype=float)
    boot_theta = boot_theta[np.isfinite(boot_theta)]

    if boot_theta.size < 100:
        return np.nan, np.nan, np.nan

    q_low, q_high = np.quantile(
        boot_theta,
        [alpha / 2, 1 - alpha / 2],
    )

    ci_low = 2 * theta_hat - q_high
    ci_high = 2 * theta_hat - q_low

    bootstrap_errors = boot_theta - theta_hat

    p_value = (
        1 + np.sum(
            np.abs(bootstrap_errors) >= abs(theta_hat)
        )
    ) / (len(bootstrap_errors) + 1)

    return float(ci_low), float(ci_high), float(p_value)


# ---- 8.1 Segment-level analysis in each of the five splits (cell 107) ----
log("RQ3: host-clustered bootstrap per segment (B = 4000)")

all_split_segment_results = []

for split_idx, seed in enumerate(SEEDS):
    test_s = split_models[seed]["test"]

    price_te_raw = np.asarray(split_models[seed]["price_te_raw"]).reshape(-1)
    pred_log = np.asarray(split_models[seed]["pred_log"]).reshape(-1)

    if not (len(test_s) == len(price_te_raw) == len(pred_log)):
        raise ValueError(f"Seed {seed}: test rows and predictions do not match.")

    pred_naive = np.maximum(np.expm1(pred_log), 0)

    gap_df = test_s[
        ["neighbourhood_group", "room_type", "host_id"]
    ].copy().reset_index(drop=True)

    gap_df["actual_price"] = price_te_raw
    gap_df["predicted_price"] = pred_naive

    gap_df["D"] = np.log(
        gap_df["actual_price"].clip(lower=1)
        / gap_df["predicted_price"].clip(lower=1)
    )

    if not np.isfinite(gap_df["D"]).all():
        raise ValueError(f"Seed {seed}: D contains NaN or infinite values.")

    overall_median = gap_df["D"].median()

    seg_rows = []

    for (ng, rt), sub in gap_df.groupby(
        ["neighbourhood_group", "room_type"],
        sort=True,
    ):
        n = len(sub)

        # Minimum segment-size requirement
        if n < MIN_SEGMENT_N:
            continue

        segment_mask = (
            (gap_df["neighbourhood_group"].to_numpy() == ng)
            & (gap_df["room_type"].to_numpy() == rt)
        )

        theta_hat, boot_theta, valid_rate, n_hosts = (
            cluster_bootstrap_segment_gap(
                df=gap_df,
                segment_mask=segment_mask,
                value_col="D",
                group_col="host_id",
                B=B_BOOT,
                # Separate reproducible seed for every split/segment
                seed=42 + split_idx * 1000 + len(seg_rows),
            )
        )

        if valid_rate < MIN_VALID_BOOTSTRAP_RATE:
            # Unreliable p-values are kept out of the Holm correction.
            ci_low = ci_high = p_value = np.nan
            inference_status = "too_many_invalid_bootstrap_replicates"
        else:
            ci_low, ci_high, p_value = (
                basic_bootstrap_ci_and_pvalue(
                    theta_hat=theta_hat,
                    boot_theta=boot_theta,
                    alpha=0.05,
                )
            )
            inference_status = "ok"

        seg_rows.append({
            "seed": seed,
            "neighbourhood_group": ng,
            "room_type": rt,
            "n": n,
            "n_hosts": n_hosts,
            "overall_median": overall_median,
            "median_centered": theta_hat,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "p_value": p_value,
            "valid_bootstrap_rate": valid_rate,
            "inference_status": inference_status,
        })

    seg_df = pd.DataFrame(seg_rows)

    # Holm correction within each split, valid p-values only.
    seg_df["p_holm"] = np.nan
    seg_df["significant_holm"] = False

    valid_p = (
        seg_df["p_value"].notna()
        & np.isfinite(seg_df["p_value"])
    )

    if valid_p.any():
        reject, p_holm, _, _ = multipletests(
            seg_df.loc[valid_p, "p_value"].to_numpy(),
            alpha=0.05,
            method="holm",
        )

        seg_df.loc[valid_p, "p_holm"] = p_holm
        seg_df.loc[valid_p, "significant_holm"] = reject

    seg_df["passes_practical_threshold"] = (
        seg_df["median_centered"].abs()
        >= PRACTICAL_THRESHOLD
    )

    seg_df["below_model"] = (
        seg_df["significant_holm"]
        & seg_df["passes_practical_threshold"]
        & (seg_df["median_centered"] < 0)
    )

    seg_df["above_model"] = (
        seg_df["significant_holm"]
        & seg_df["passes_practical_threshold"]
        & (seg_df["median_centered"] > 0)
    )

    all_split_segment_results.append(seg_df)

    print(
        f"Seed {seed}: overall median D = {overall_median:.4f}; "
        f"valid segments = {len(seg_df)}; "
        f"invalid inference = "
        f"{(seg_df['inference_status'] != 'ok').sum()}",
        flush=True,
    )
    log(f"RQ3 split seed={seed} done")

full_segment_df = pd.concat(
    all_split_segment_results,
    ignore_index=True,
)
full_segment_df = full_segment_df.sort_values(
    ["seed", "neighbourhood_group", "room_type"]
).reset_index(drop=True)

# ---- 8.2 Aggregation over the five splits (cell 109) ----
# A split without a valid result for a segment counts as NOT passing.
segments = full_segment_df[
    ["neighbourhood_group", "room_type"]
].drop_duplicates()

split_grid = segments.merge(
    pd.DataFrame({"seed": list(SEEDS)}),
    how="cross",
)

split_grid = split_grid.merge(
    full_segment_df,
    on=["seed", "neighbourhood_group", "room_type"],
    how="left",
    validate="one_to_one",
)

for col in [
    "significant_holm",
    "passes_practical_threshold",
    "below_model",
    "above_model",
]:
    split_grid[col] = (
        split_grid[col]
        .astype("boolean")
        .fillna(False)
        .astype(bool)
    )

split_grid["both_conditions"] = (
    split_grid["significant_holm"]
    & split_grid["passes_practical_threshold"]
)

rq3_summary = split_grid.groupby(
    ["neighbourhood_group", "room_type"],
    as_index=False,
).agg(
    n_splits_present=("n", "count"),
    median_n=("n", "median"),
    mean_median_centered=("median_centered", "mean"),
    n_holm=("significant_holm", "sum"),
    n_practical=("passes_practical_threshold", "sum"),
    n_both=("both_conditions", "sum"),
    n_below=("below_model", "sum"),
    n_above=("above_model", "sum"),
)

# Direction consistency: same direction (Holm-significant AND >= 5 %) in >= 3 of 5 splits.
rq3_summary["stable_below_model"] = rq3_summary["n_below"] >= 3
rq3_summary["stable_above_model"] = rq3_summary["n_above"] >= 3

# Convenience flag = the four criteria combined (Holm, +-5 %, >= 3/5 splits,
# segment size >= 25 in the splits counted). Pure OR of the two columns above.
rq3_summary["meets_all_criteria"] = (
    rq3_summary["stable_below_model"] | rq3_summary["stable_above_model"]
)

rq3_summary = rq3_summary.sort_values("mean_median_centered").reset_index(drop=True)

print("\n===== RQ3: 5-split summary =====")
print(rq3_summary.to_string(index=False))
print(f"\nSegments meeting all criteria: {int(rq3_summary['meets_all_criteria'].sum())}")

save_table(full_segment_df, "rq3_segment_results_by_split.csv", TABLE_PATH)
save_table(rq3_summary, "rq3_segment_summary_5_splits.csv", TABLE_PATH)

# ---- 8.3 Figure: observed deviation by segment (cell 110) ----
plot_df = rq3_summary.copy()

# Mean log-ratio -> exact percentage: 100 * (exp(D) - 1)
plot_df['pct_dev'] = 100.0 * (np.exp(plot_df['mean_median_centered']) - 1.0)

plot_df['label'] = (
    plot_df['neighbourhood_group'] + ' — ' + plot_df['room_type']
    + '  (' + plot_df['n_splits_present'].astype(str) + '/5)'
)

plot_df = plot_df.sort_values('pct_dev', ascending=True).reset_index(drop=True)

plt.rcParams.update({
    'font.size': 11,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 300,
})

THRESH_PCT = 100.0 * (np.exp(np.log(1.05)) - 1.0)   # = 5.0
MARKER_COLOR = '#1f4e79'

fig, ax = plt.subplots(figsize=(10.5, 6.8))

ax.axvspan(-THRESH_PCT, THRESH_PCT, color='#f0f0f0', zorder=0)
ax.axvline(0, color='black', lw=0.8, zorder=1)
ax.axvline(-THRESH_PCT, color='#999999', lw=0.9, ls='--', zorder=1)
ax.axvline(THRESH_PCT, color='#999999', lw=0.9, ls='--', zorder=1)

y = np.arange(len(plot_df))
for i, row in plot_df.iterrows():
    ax.scatter(row['pct_dev'], i,
               marker='o', s=90, color=MARKER_COLOR,
               edgecolor='white', linewidth=0.8, zorder=3)
    ax.plot([0, row['pct_dev']], [i, i],
            color=MARKER_COLOR, lw=1.0, alpha=0.35, zorder=2)

    if row['pct_dev'] >= 0:
        ha = 'left'
        x_offset = 0.4
    else:
        ha = 'right'
        x_offset = -0.4

    txt = f"{row['pct_dev']:+.2f}%  ·  {int(row['n_splits_present'])}/5 splits"
    ax.text(row['pct_dev'] + x_offset, i, txt,
            va='center', ha=ha, fontsize=9.5, color='#333333')

ax.set_yticks(y)
ax.set_yticklabels(plot_df['label'], fontsize=10)
ax.set_xlabel('Mean split-level deviation (%)', fontsize=11, labelpad=10)

xmin = min(-THRESH_PCT * 1.6, plot_df['pct_dev'].min() * 1.5)
xmax = max(THRESH_PCT * 1.6, plot_df['pct_dev'].max() * 1.25)
ax.set_xlim(xmin, xmax)
ax.grid(axis='x', color='#dddddd', lw=0.5, zorder=0)
ax.set_axisbelow(True)

legend_handles = [
    Patch(facecolor='#f0f0f0', edgecolor='none',
          label=f'Within practical threshold ±{THRESH_PCT:.0f}%'),
]
ax.legend(handles=legend_handles, loc='lower right', fontsize=9,
          frameon=True, framealpha=0.95)

plt.tight_layout()
plt.savefig(os.path.join(FIGURE_PATH, 'rq3_dotplot_deviation_en.png'), dpi=300, bbox_inches='tight')
plt.close(fig)
log("saved outputs/figures/rq3_dotplot_deviation_en.png")


# ============================================================
# 9. HOST-TYPE ANALYSIS: MULTI-HOST vs SINGLE-HOST
#    (notebook cells 112-119)
# ============================================================
# host_type = 'multi' if calculated_host_listings_count > 1 else 'single'.
# Estimand: median(D_multi) - median(D_single); hosts are resampled with
# replacement separately inside each group, keeping all listings of a host.

# ---- 9.1 Overall comparison, one bootstrap per split (cell 112) ----
log("host type: overall comparison (B = 2000)")

B = 2000
BASE_BOOTSTRAP_SEED = 42

# The five seeds in ascending order (this order sets the RNG seed offset).
seeds = sorted(split_models.keys())[:5]

if len(seeds) != 5:
    raise ValueError(f"Five seeds are required, but split_models has {len(seeds)}.")

results_host_type = []

for seed_idx, seed in enumerate(seeds):

    model_data = split_models[seed]

    test = model_data['test']
    price_te_raw = np.asarray(model_data['price_te_raw']).reshape(-1)
    pred_log = np.asarray(model_data['pred_log']).reshape(-1)

    if len(test) != len(price_te_raw) or len(test) != len(pred_log):
        raise ValueError(
            f"Seed {seed}: test rows, actual prices and predictions do not match."
        )

    pred_price = np.maximum(np.expm1(pred_log), 0)

    host_gap_df = test[
        ['calculated_host_listings_count', 'host_id']
    ].copy()

    host_gap_df['actual_price'] = price_te_raw
    host_gap_df['pred_price'] = pred_price

    host_gap_df['D'] = np.log(
        np.maximum(host_gap_df['actual_price'], 1)
        / np.maximum(host_gap_df['pred_price'], 1)
    )

    host_gap_df['host_type'] = np.where(
        host_gap_df['calculated_host_listings_count'] > 1,
        'multi',
        'single'
    )

    host_gap_df = host_gap_df[
        host_gap_df['host_id'].notna()
        & np.isfinite(host_gap_df['D'])
    ].copy()

    multi_df = host_gap_df[host_gap_df['host_type'] == 'multi']
    single_df = host_gap_df[host_gap_df['host_type'] == 'single']

    multi_groups = [
        g['D'].to_numpy()
        for _, g in multi_df.groupby('host_id', sort=False)
    ]

    single_groups = [
        g['D'].to_numpy()
        for _, g in single_df.groupby('host_id', sort=False)
    ]

    if len(multi_groups) < 2 or len(single_groups) < 2:
        raise ValueError(
            f"Seed {seed}: not enough hosts in one of the two groups to bootstrap."
        )

    median_multi = np.median(multi_df['D'])
    median_single = np.median(single_df['D'])

    # Multi minus single
    observed_diff = median_multi - median_single

    rng = np.random.default_rng(BASE_BOOTSTRAP_SEED + seed_idx)

    boot_diffs = np.empty(B)

    for b in range(B):

        m_idx = rng.integers(
            0, len(multi_groups), size=len(multi_groups)
        )
        s_idx = rng.integers(
            0, len(single_groups), size=len(single_groups)
        )

        boot_multi = np.concatenate(
            [multi_groups[j] for j in m_idx]
        )
        boot_single = np.concatenate(
            [single_groups[j] for j in s_idx]
        )

        boot_diffs[b] = (
            np.median(boot_multi) - np.median(boot_single)
        )

    # Percentile bootstrap 95 % CI
    ci_low, ci_high = np.percentile(boot_diffs, [2.5, 97.5])

    results_host_type.append({
        'seed': seed,
        'n_multi_listings': len(multi_df),
        'n_single_listings': len(single_df),
        'n_multi_hosts': len(multi_groups),
        'n_single_hosts': len(single_groups),
        'median_D_multi': median_multi,
        'median_D_single': median_single,
        'diff_multi_minus_single': observed_diff,
        'ci95_low': ci_low,
        'ci95_high': ci_high,
        'ci_excludes_zero': (ci_low > 0) or (ci_high < 0)
    })

host_type_results = pd.DataFrame(results_host_type)

print('\n=== HOST TYPE: MULTI-HOST - SINGLE-HOST ===')
print(host_type_results.round(4).to_string(index=False))

# ---- 9.2 Consistency summary across the five seeds (cell 115) ----
host_type_consistency = {
    'mean_observed_difference': round(float(host_type_results['diff_multi_minus_single'].mean()), 4),
    'median_observed_difference': round(float(host_type_results['diff_multi_minus_single'].median()), 4),
    'seeds_positive_difference': int((host_type_results['diff_multi_minus_single'] > 0).sum()),
    'seeds_negative_difference': int((host_type_results['diff_multi_minus_single'] < 0).sum()),
    'seeds_ci_excludes_zero': int(host_type_results['ci_excludes_zero'].sum()),
    'n_seeds': int(len(host_type_results)),
}
print('\n=== SUMMARY ACROSS 5 SEEDS ===')
for k, v in host_type_consistency.items():
    print(f"{k}: {v}")

save_table(host_type_results, "host_type_overall_by_seed.csv", TABLE_PATH)
with open(os.path.join(METRIC_PATH, "host_type_overall_consistency.json"), "w", encoding="utf-8") as fh:
    json.dump(host_type_consistency, fh, indent=2)
log("saved outputs/metrics/host_type_overall_consistency.json")

# ---- 9.3 Forest plot: overall difference per seed (cell 114) ----
# The notebook typed these numbers in by hand (rounded to 4 decimals);
# here they come straight from host_type_results (same values, unrounded).
seed_labels = [str(s) for s in host_type_results['seed']]
diff = host_type_results['diff_multi_minus_single'].to_numpy()
ci_low_arr = host_type_results['ci95_low'].to_numpy()
ci_high_arr = host_type_results['ci95_high'].to_numpy()

xerr = np.array([
    diff - ci_low_arr,
    ci_high_arr - diff
])

fig, ax = plt.subplots(figsize=(10, 5.5))

y = np.arange(len(seed_labels))

ax.axvline(
    0,
    color='black',
    linestyle='--',
    linewidth=1.3,
    alpha=0.7,
    label='No difference'
)

ax.errorbar(
    diff,
    y,
    xerr=xerr,
    fmt='o',
    color='#2878B5',
    ecolor='#2878B5',
    elinewidth=2,
    capsize=5,
    markersize=8,
    markeredgecolor='white',
    markeredgewidth=1
)

ax.set_yticks(y)
ax.set_yticklabels(seed_labels)
ax.invert_yaxis()

ax.set_xlabel(
    'Difference in Median Pricing Deviation '
    '(Multi-host − Single-host)',
    fontsize=11
)

ax.set_ylabel('Random Seed', fontsize=11)

ax.set_title(
    'Multi-host vs Single-host: Difference in Median Deviation',
    fontsize=13,
    fontweight='bold',
    pad=15
)

ax.set_xlim(-0.03, 0.04)
ax.set_xticks(np.arange(-0.03, 0.041, 0.01))
ax.set_xticklabels(
    [f'{v:.0%}' for v in np.arange(-0.03, 0.041, 0.01)]
)

ax.grid(axis='x', linestyle=':', alpha=0.4)
ax.spines[['top', 'right']].set_visible(False)

plt.tight_layout()
plt.savefig(
    os.path.join(FIGURE_PATH, 'host_type_forest_plot.png'),
    dpi=300,
    bbox_inches='tight'
)
plt.close(fig)
log("saved outputs/figures/host_type_forest_plot.png")

# ---- 9.4 Host type within each room type (cell 116) ----
log("host type: within each room type (B = 2000)")

B_HOST = 2000
BASE_HOST_SEED = 20261029
MIN_HOSTS_PER_GROUP = 2

host_type_room_results = []

# Same order of the five splits as in the earlier analyses.
seeds = list(SEEDS)

if len(seeds) != 5:
    raise ValueError(f"Exactly five SEEDS are required, got {len(seeds)}.")

for split_idx, seed in enumerate(seeds):

    model_data = split_models[seed]
    test = model_data["test"].copy().reset_index(drop=True)

    price_actual = np.asarray(
        model_data["price_te_raw"]
    ).reshape(-1)

    pred_log = np.asarray(
        model_data["pred_log"]
    ).reshape(-1)

    if not (
        len(test) == len(price_actual) == len(pred_log)
    ):
        raise ValueError(
            f"Seed {seed}: test rows, actual prices and predictions do not match."
        )

    required_cols = [
        "room_type",
        "calculated_host_listings_count",
        "host_id",
    ]
    missing_cols = [
        c for c in required_cols if c not in test.columns
    ]
    if missing_cols:
        raise ValueError(
            f"Seed {seed}: missing columns {missing_cols}."
        )

    host_df = test[required_cols].copy()
    host_df["actual_price"] = price_actual
    host_df["pred_price"] = np.maximum(
        np.expm1(pred_log), 0
    )

    host_df["D"] = np.log(
        np.maximum(host_df["actual_price"], 1)
        / np.maximum(host_df["pred_price"], 1)
    )

    host_df["host_type"] = np.where(
        host_df["calculated_host_listings_count"] > 1,
        "multi",
        "single",
    )

    host_df = host_df.dropna(
        subset=[
            "host_id",
            "room_type",
            "calculated_host_listings_count",
        ]
    ).copy()

    host_df = host_df[
        np.isfinite(host_df["D"])
    ].copy()

    for room_idx, room_type in enumerate(
        sorted(host_df["room_type"].unique())
    ):

        room_df = host_df[
            host_df["room_type"] == room_type
        ].copy()

        multi_df = room_df[
            room_df["host_type"] == "multi"
        ]
        single_df = room_df[
            room_df["host_type"] == "single"
        ]

        multi_hosts = [
            group["D"].to_numpy()
            for _, group in multi_df.groupby(
                "host_id", sort=False
            )
        ]

        single_hosts = [
            group["D"].to_numpy()
            for _, group in single_df.groupby(
                "host_id", sort=False
            )
        ]

        n_multi_hosts = len(multi_hosts)
        n_single_hosts = len(single_hosts)

        # Not enough hosts: record the status instead of treating it as 0.
        if (
            n_multi_hosts < MIN_HOSTS_PER_GROUP
            or n_single_hosts < MIN_HOSTS_PER_GROUP
        ):
            host_type_room_results.append({
                "seed": seed,
                "room_type": room_type,
                "n_multi_listings": len(multi_df),
                "n_single_listings": len(single_df),
                "n_multi_hosts": n_multi_hosts,
                "n_single_hosts": n_single_hosts,
                "median_D_multi": np.nan,
                "median_D_single": np.nan,
                "diff_multi_minus_single": np.nan,
                "ci95_low": np.nan,
                "ci95_high": np.nan,
                "ci_excludes_zero": np.nan,
                "status": "insufficient_hosts",
            })
            continue

        median_multi = np.median(multi_df["D"])
        median_single = np.median(single_df["D"])

        # Positive: multi-host median D is higher than single-host in the same room type.
        observed_diff = median_multi - median_single

        rng = np.random.default_rng(
            BASE_HOST_SEED
            + split_idx * 100
            + room_idx
        )

        boot_diffs = np.empty(B_HOST)

        for b in range(B_HOST):

            idx_multi = rng.integers(
                0, n_multi_hosts, size=n_multi_hosts
            )
            idx_single = rng.integers(
                0, n_single_hosts, size=n_single_hosts
            )

            boot_multi = np.concatenate([
                multi_hosts[j] for j in idx_multi
            ])
            boot_single = np.concatenate([
                single_hosts[j] for j in idx_single
            ])

            boot_diffs[b] = (
                np.median(boot_multi)
                - np.median(boot_single)
            )

        # Percentile bootstrap 95 % CI
        ci_low, ci_high = np.quantile(
            boot_diffs, [0.025, 0.975]
        )

        ci_excludes_zero = (
            (ci_low > 0) or (ci_high < 0)
        )

        host_type_room_results.append({
            "seed": seed,
            "room_type": room_type,
            "n_multi_listings": len(multi_df),
            "n_single_listings": len(single_df),
            "n_multi_hosts": n_multi_hosts,
            "n_single_hosts": n_single_hosts,
            "median_D_multi": median_multi,
            "median_D_single": median_single,
            "diff_multi_minus_single": observed_diff,
            "ci95_low": ci_low,
            "ci95_high": ci_high,
            "ci_excludes_zero": ci_excludes_zero,
            "status": "ok",
        })

host_type_room_df = pd.DataFrame(host_type_room_results)

print("\n=== HOST TYPE WITHIN EACH ROOM TYPE: PER SPLIT ===")
print(
    host_type_room_df.sort_values(["room_type", "seed"]).round(4).to_string(index=False)
)

# ---- 9.5 Summary over the five splits (cell 118); CIs are NOT pooled ----
valid_host_results = host_type_room_df[
    host_type_room_df["status"] == "ok"
].copy()

host_type_room_summary = (
    valid_host_results
    .groupby("room_type", as_index=False)
    .agg(
        n_splits_valid=("seed", "nunique"),
        mean_diff=("diff_multi_minus_single", "mean"),
        sd_diff=("diff_multi_minus_single", "std"),
        median_diff=("diff_multi_minus_single", "median"),
        n_ci_excludes_zero=("ci_excludes_zero", "sum"),
        n_positive=("diff_multi_minus_single",
                    lambda x: (x > 0).sum()),
        n_negative=("diff_multi_minus_single",
                    lambda x: (x < 0).sum()),
    )
    .sort_values("room_type")
    .reset_index(drop=True)
)

print("\n=== HOST TYPE WITHIN EACH ROOM TYPE: 5-SPLIT SUMMARY ===")
print(host_type_room_summary.round(4).to_string(index=False))

save_table(host_type_room_df, "host_type_by_room_type_by_split.csv", TABLE_PATH)
save_table(host_type_room_summary, "host_type_by_room_type_summary.csv", METRIC_PATH)

# ---- 9.6 Figure: host-type difference by room type (cell 119) ----
# The notebook typed mean/SD/CI counts in by hand; here they come from
# host_type_room_summary (same values, unrounded).
room_types = host_type_room_summary['room_type'].tolist()
mean_diff = host_type_room_summary['mean_diff'].to_numpy(dtype=float)
sd_diff = host_type_room_summary['sd_diff'].to_numpy(dtype=float)
n_ci = [int(v) for v in host_type_room_summary['n_ci_excludes_zero']]

y = np.arange(len(room_types))

fig, ax = plt.subplots(figsize=(10, 5))

ax.axvline(
    0,
    color='black',
    linestyle='--',
    linewidth=1.3,
    alpha=0.7
)

ax.errorbar(
    mean_diff,
    y,
    xerr=sd_diff,
    fmt='o',
    color='#2878B5',
    ecolor='#2878B5',
    elinewidth=2,
    capsize=5,
    markersize=9,
    markeredgecolor='white',
    markeredgewidth=1
)

ax.set_yticks(y)
ax.set_yticklabels(room_types)
ax.invert_yaxis()

ax.set_xlabel(
    'Mean Difference in Median Pricing Deviation\n'
    '(Multi-host − Single-host)',
    fontsize=11
)

ax.set_ylabel('Room Type', fontsize=11)

ax.set_title(
    'Host-type Difference Across Room Types',
    fontsize=13,
    fontweight='bold',
    pad=15
)

ax.grid(axis='x', linestyle=':', alpha=0.4)
ax.spines[['top', 'right']].set_visible(False)

for i, count in enumerate(n_ci):
    ax.text(
        0.98, i,
        f'{count}/5 CI exclude zero',
        transform=ax.get_yaxis_transform(),
        ha='right',
        va='center',
        fontsize=9
    )

plt.tight_layout()
plt.savefig(
    os.path.join(FIGURE_PATH, 'host_type_by_room_type.png'),
    dpi=300,
    bbox_inches='tight'
)
plt.close(fig)
log("saved outputs/figures/host_type_by_room_type.png")


# ============================================================
# 10. RQ1 - FEATURE IMPORTANCE (split seed 42)
#     (notebook cells 121-124)
# ============================================================
# Lasso (alpha = 1e-4, drop_first=True), Random Forest impurity importance and
# XGBoost gain importance, all taken from the seed-42 fits of section 6.
# The notebook typed the Lasso coefficients into cell 122 by hand; here they
# come straight from the fitted model.

log("RQ1: Lasso coefficients and tree feature importance")

lasso_coef = rq1_store['Lasso'].sort_values(key=abs, ascending=False)
print(lasso_coef)
print('\nFeatures dropped by Lasso (coef=0):', (lasso_coef == 0).sum(), '/', len(lasso_coef))

importance_df = pd.DataFrame({
    'Feature': rq1_store['RF'].index,
    'RF_Importance': rq1_store['RF'].to_numpy(),
    'XGB_Importance': rq1_store['XGB'].reindex(rq1_store['RF'].index).to_numpy(),
}).sort_values('RF_Importance', ascending=False)

lasso_abs = rq1_store['Lasso'].abs().rename('Lasso_Coef_Abs')
importance_df = importance_df.merge(
    lasso_abs.rename_axis('Feature').reset_index(), on='Feature', how='left'
)
print(importance_df.head(20).to_string(index=False))

save_table(lasso_coef.rename_axis('Feature').reset_index(name='Lasso_Coefficient'),
           "rq1_lasso_coefficients.csv", TABLE_PATH)
save_table(importance_df, "rq1_feature_importance.csv", TABLE_PATH)

# ---- 10.1 Figure: Lasso coefficients (cell 122) ----
coefs_sorted = lasso_coef.sort_values(key=abs, ascending=True)

fig, ax = plt.subplots(figsize=(10, 10), dpi=300)

colors = ['#2b5c8f' if val >= 0 else '#d95f02' for val in coefs_sorted.values]

bars = ax.barh(coefs_sorted.index, coefs_sorted.values,
               color=colors, alpha=0.85, edgecolor='black', linewidth=0.5)

ax.axvline(0, color='black', linewidth=0.8, linestyle='--')

ax.set_xlabel('Lasso Coefficient Value', fontsize=12, fontweight='bold')
ax.set_ylabel('Features', fontsize=12, fontweight='bold')
ax.tick_params(axis='both', labelsize=9)

for bar in bars:
    width = bar.get_width()
    align = 'left' if width >= 0 else 'right'
    x_pos = width + 0.005 if width >= 0 else width - 0.005
    ax.text(x_pos, bar.get_y() + bar.get_height()/2,
            f'{width:.4f}', ha=align, va='center', fontsize=7)

plt.tight_layout()
# The notebook has this savefig commented out; the file name is taken from it.
plt.savefig(os.path.join(FIGURE_PATH, 'lasso_coefficients.png'), dpi=300, bbox_inches='tight')
plt.close(fig)
log("saved outputs/figures/lasso_coefficients.png")

# ---- 10.2 Figure: top-20 RF / XGBoost importance (cell 124) ----
top20_rf = importance_df.sort_values('RF_Importance', ascending=False).head(20).sort_values('RF_Importance', ascending=True)
top20_xgb = importance_df.sort_values('XGB_Importance', ascending=False).head(20).sort_values('XGB_Importance', ascending=True)

fig, axes = plt.subplots(1, 2, figsize=(12, 7), sharey=False)

axes[0].barh(top20_rf['Feature'], top20_rf['RF_Importance'], color='#4C72B0')
axes[0].set_title('Random Forest - Top 20 Features', fontsize=13, fontweight='bold', pad=10)
axes[0].set_xlabel('Importance', fontsize=11)
axes[0].tick_params(axis='both', labelsize=10)
axes[0].grid(axis='x', linestyle='--', alpha=0.5)

axes[1].barh(top20_xgb['Feature'], top20_xgb['XGB_Importance'], color='#DD8452')
axes[1].set_title('XGBoost - Top 20 Features', fontsize=13, fontweight='bold', pad=10)
axes[1].set_xlabel('Importance', fontsize=11)
axes[1].tick_params(axis='both', labelsize=10)
axes[1].grid(axis='x', linestyle='--', alpha=0.5)

plt.tight_layout()
# The notebook has this savefig commented out; the file name is taken from it.
plt.savefig(os.path.join(FIGURE_PATH, 'RQ1_Feature_Importance2.png'), dpi=300, bbox_inches='tight')
plt.close(fig)
log("saved outputs/figures/RQ1_Feature_Importance2.png")


# ============================================================
# 11. GROUPKFOLD vs STRATIFIEDKFOLD AND SMEARING FACTOR
#     (notebook cells 98-102)
# ============================================================
# Everything here uses the TRAIN part of the first split (seed = SEEDS[0] = 42):
#   * StratifiedKFold(5, shuffle=True, random_state=42) on 10 price bins of train
#   * GroupKFold(5) on host_id
# The out-of-fold XGBoost residuals (log scale, StratifiedKFold folds) give ONE
# smearing factor S = mean(exp(residual)); it is NOT estimated per split.
# Each stratified fold is prepared and fitted once and feeds both the MAE and
# the residuals (the notebook fitted the same model twice; the fit is identical).

log("GroupKFold vs StratifiedKFold (3 tree models) and out-of-fold smearing (slow step)")

# NOTE on reproducibility: GroupKFold breaks ties between groups of equal size
# (most hosts own a single listing) with numpy's argsort, which is not stable.
# The resulting fold composition - and therefore the GroupKFold MAE and its SD -
# can differ slightly between scikit-learn / numpy versions and CPUs. The
# StratifiedKFold results, the smearing factor and everything else are unaffected.

seed0 = SEEDS[0]
price_bins0 = pd.qcut(train_full0['price'], q=10, labels=False, duplicates='drop')
skf0 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
gkf = GroupKFold(n_splits=5)

CV_MODELS = ['XGBoost', 'Random Forest', 'Gradient Boosting']
cv_fold_rows = []
residuals_oof = []


def run_cv_scheme(scheme, splits, collect_residuals):
    for fold_no, (tr_idx, va_idx) in enumerate(splits, start=1):
        X_tr_f, y_tr_f, X_va_f, y_va_f, price_va_raw_f = prepare_fold_fixed(
            train_full0.iloc[tr_idx], train_full0.iloc[va_idx], use_name_features=True
        )
        for name in CV_MODELS:
            m = build_model_fn(name)()
            m.fit(X_tr_f, y_tr_f)
            pred_log_f = m.predict(X_va_f)
            pred_f = np.maximum(np.expm1(pred_log_f), 0)
            cv_fold_rows.append({
                'model': name, 'scheme': scheme, 'fold': fold_no,
                'mae': mean_absolute_error(price_va_raw_f, pred_f),
            })
            if collect_residuals and name == 'XGBoost':
                residuals_oof.append(y_va_f.values - pred_log_f)
            del m
        del X_tr_f, y_tr_f, X_va_f, y_va_f
        gc.collect()
        log(f"  {scheme} fold {fold_no}/5 done")


run_cv_scheme('StratifiedKFold', skf0.split(train_full0, price_bins0), collect_residuals=True)
run_cv_scheme('GroupKFold', gkf.split(train_full0, groups=train_full0['host_id']), collect_residuals=False)

# ---- 11.1 Smearing factor (cell 98) ----
residuals_oof = np.concatenate(residuals_oof)
smearing_factor = np.mean(np.exp(residuals_oof))
print(f'\nSmearing factor S = {smearing_factor:.4f}')

with open(os.path.join(METRIC_PATH, "smearing_factor.json"), "w", encoding="utf-8") as fh:
    json.dump({
        "smearing_factor": float(smearing_factor),
        "definition": "mean(exp(out-of-fold residual on log1p(capped price))), XGBoost best_params_xgb, "
                      "StratifiedKFold(5, shuffle, random_state=42) on the train part of split seed 42",
        "n_oof_residuals": int(len(residuals_oof)),
    }, fh, indent=2)
log("saved outputs/metrics/smearing_factor.json")

# ---- 11.2 GroupKFold vs StratifiedKFold summary (cells 99-102) ----
cv_fold_df = pd.DataFrame(cv_fold_rows)
save_table(cv_fold_df, "groupkfold_vs_stratifiedkfold_fold_mae.csv", TABLE_PATH)


def _cv_mae(model_name, scheme):
    return cv_fold_df[(cv_fold_df['model'] == model_name) & (cv_fold_df['scheme'] == scheme)]['mae'].to_numpy()


groupkfold_summary = pd.DataFrame([
    {
        'Model': model_name,
        'StratifiedKFold_MAE_mean': np.mean(_cv_mae(model_name, 'StratifiedKFold')),
        'StratifiedKFold_MAE_sd': np.std(_cv_mae(model_name, 'StratifiedKFold')),
        'GroupKFold_MAE_mean': np.mean(_cv_mae(model_name, 'GroupKFold')),
        'GroupKFold_MAE_sd': np.std(_cv_mae(model_name, 'GroupKFold')),
    }
    for model_name in ['Random Forest', 'XGBoost', 'Gradient Boosting']
])

groupkfold_summary['MAE_change'] = (
    groupkfold_summary['GroupKFold_MAE_mean']
    - groupkfold_summary['StratifiedKFold_MAE_mean']
)

print("\n=== GroupKFold vs StratifiedKFold (MAE, USD; sd = np.std over 5 folds) ===")
print(groupkfold_summary.round(2).to_string(index=False))
save_table(groupkfold_summary, "groupkfold_vs_stratifiedkfold_summary.csv", METRIC_PATH)


# ============================================================
# 12. CALIBRATION - MEDIAN PREDICTION AND SMEARING-CORRECTED MEAN PREDICTION
#     (notebook cells 126-127)
# ============================================================
# The model predicts log1p(price).
#   median prediction : naive back-transform  expm1(pred_log)
#   mean prediction   : smearing-corrected    S * exp(pred_log) - 1  (S from section 11)
# Both are evaluated on the TEST part of split seed 42 by prediction decile.

log("calibration by prediction decile")

pred_log = split_models[seed0]['pred_log']
price_te_raw = split_models[seed0]['price_te_raw']
pred_naive = np.maximum(np.expm1(pred_log), 0)
pred_smeared = np.maximum(smearing_factor * np.exp(pred_log) - 1, 0)

cal_df = pd.DataFrame({'actual': price_te_raw, 'pred_naive': pred_naive, 'pred_smeared': pred_smeared})
cal_df['decile'] = pd.qcut(cal_df['pred_naive'], 10, labels=[f'D{i}' for i in range(1, 11)])

calibration = cal_df.groupby('decile', observed=False).agg(
    n=('actual', 'size'),
    actual_median=('actual', 'median'),
    pred_median=('pred_naive', 'median'),
    actual_mean=('actual', 'mean'),
    pred_mean_smeared=('pred_smeared', 'mean'),
)
calibration['ratio_median'] = calibration['actual_median'] / calibration['pred_median']
calibration['ratio_mean_smeared'] = calibration['actual_mean'] / calibration['pred_mean_smeared']
print('\n=== CALIBRATION (median-vs-median, and mean-vs-mean with smearing) ===')
print(calibration.to_string())

save_table(calibration.reset_index(), "calibration_by_decile.csv", TABLE_PATH)

fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

# --- Left panel: median-to-median (naive) ---
ax = axes[0]
ax.scatter(calibration['pred_median'], calibration['actual_median'], color='tab:blue', zorder=3)
lims = [0, max(calibration['pred_median'].max(), calibration['actual_median'].max()) * 1.05]
ax.plot(lims, lims, '--', color='gray', label='Perfect calibration ($y=x$)')
for dec, row in calibration.iterrows():
    ax.annotate(dec, (row['pred_median'], row['actual_median']), fontsize=8,
                textcoords="offset points", xytext=(4, 4))
ax.set_xlabel('Median predicted price ($)')
ax.set_ylabel('Median actual price ($)')
ax.set_title('(a) Median-to-median (naive predictions)')
ax.legend()

# --- Right panel: mean-to-mean (smearing-corrected) ---
ax = axes[1]
ax.scatter(calibration['pred_mean_smeared'], calibration['actual_mean'], color='tab:orange', zorder=3)
lims = [0, max(calibration['pred_mean_smeared'].max(), calibration['actual_mean'].max()) * 1.05]
ax.plot(lims, lims, '--', color='gray', label='Perfect calibration ($y=x$)')
for dec, row in calibration.iterrows():
    ax.annotate(dec, (row['pred_mean_smeared'], row['actual_mean']), fontsize=8,
                textcoords="offset points", xytext=(4, 4))
ax.set_xlabel('Mean predicted price, smearing-corrected ($)')
ax.set_ylabel('Mean actual price ($)')
ax.set_title('(b) Mean-to-mean (smearing-corrected)')
ax.legend()

plt.tight_layout()
plt.savefig(os.path.join(FIGURE_PATH, 'calibration.png'), dpi=200, bbox_inches='tight')
plt.close(fig)
log("saved outputs/figures/calibration.png")

del train_full0
gc.collect()


# ============================================================
# 13. RUN METADATA (for traceability of the reproduction)
# ============================================================

run_metadata = {
    "script": "reproduce.py",
    "seeds": SEEDS,
    "n_rows_after_cleaning": int(len(df_clean)),
    "hyperparameters": {
        "random_forest": best_params_rf,
        "xgboost": best_params_xgb,
        "gradient_boosting": best_params_gbm,
        "ridge_alpha": best_alpha_ridge,
        "lasso_alpha": best_alpha_lasso,
    },
    "rq3": {
        "B_boot": B_BOOT,
        "min_segment_n": MIN_SEGMENT_N,
        "practical_threshold_log": float(PRACTICAL_THRESHOLD),
        "min_valid_bootstrap_rate": MIN_VALID_BOOTSTRAP_RATE,
        "holm_alpha": 0.05,
        "min_splits_same_direction": 3,
    },
    "host_type": {"B_overall": B, "B_by_room_type": B_HOST},
    "smearing_factor": float(smearing_factor),
    "environment": {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit-learn": sklearn.__version__,
        "statsmodels": statsmodels.__version__,
        "xgboost": xgboost.__version__,
        "matplotlib": matplotlib.__version__,
        "seaborn": sns.__version__,
    },
    "runtime_seconds": round(time.time() - _T0, 1),
}
with open(os.path.join(METRIC_PATH, "run_metadata.json"), "w", encoding="utf-8") as fh:
    json.dump(run_metadata, fh, indent=2)

log("Quick Reproduction finished. Outputs are in outputs/tables, outputs/metrics, outputs/figures.")
