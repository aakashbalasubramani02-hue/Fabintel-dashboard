"""
FABINTEL — SECOM Process Inference Backend
Loads the XGBoost model + preprocessing pipeline and provides
prediction + SHAP explanations.
"""
import os
import json
import pickle
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_BASE = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
MODEL_PATH = os.path.join(_BASE, 'models', 'secom', 'secom_xgboost.pkl')
IMPUTER_PATH = os.path.join(_BASE, 'models', 'secom', 'secom_imputer.pkl')
SCALER_PATH = os.path.join(_BASE, 'models', 'secom', 'secom_scaler.pkl')
FEATURE_COLS_PATH = os.path.join(_BASE, 'models', 'secom', 'secom_feature_cols.json')
GLOBAL_SHAP_PATH = os.path.join(_BASE, 'results', 'phase12', 'top_process_risk_factors.csv')
SECOM_DATA_PATH = os.path.join(_BASE, 'secom_data', 'secom.data')

# ---------------------------------------------------------------------------
# Lazy-loaded artifacts
# ---------------------------------------------------------------------------
_model = None
_imputer = None
_scaler = None
_feature_cols = None

def _load_artifacts():
    global _model, _imputer, _scaler, _feature_cols
    if _model is None:
        with open(MODEL_PATH, 'rb') as f:
            _model = pickle.load(f)
        with open(IMPUTER_PATH, 'rb') as f:
            _imputer = pickle.load(f)
        with open(SCALER_PATH, 'rb') as f:
            _scaler = pickle.load(f)
        with open(FEATURE_COLS_PATH, 'r') as f:
            _feature_cols = json.load(f)
    return _model, _imputer, _scaler, _feature_cols

# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
def preprocess_record(df_row):
    """
    Preprocess a single SECOM record (pd.DataFrame with 590 columns).
    Returns the imputed (unscaled) numpy array suitable for tree models.
    """
    model, imputer, scaler, feature_cols = _load_artifacts()
    X = df_row[feature_cols]
    X_imp = imputer.transform(X)
    return X_imp

def preprocess_csv(uploaded_df):
    """
    Preprocess an uploaded CSV of SECOM records.
    Expects a DataFrame with at least 590 numeric columns (column indices 0-589).
    Returns imputed array and metadata.
    """
    model, imputer, scaler, feature_cols = _load_artifacts()

    # Ensure columns are integers
    uploaded_df.columns = list(range(len(uploaded_df.columns)))

    # Check we have enough columns
    max_col_needed = max(feature_cols)
    if len(uploaded_df.columns) < max_col_needed + 1:
        raise ValueError(
            f"Expected at least {max_col_needed + 1} columns, got {len(uploaded_df.columns)}. "
            f"Ensure the uploaded file matches SECOM format (590 features)."
        )

    X = uploaded_df[feature_cols]
    missing_before = int(X.isnull().sum().sum())
    X_imp = imputer.transform(X)

    return X_imp, {
        'n_records': len(uploaded_df),
        'n_features_used': len(feature_cols),
        'n_features_total': len(uploaded_df.columns),
        'missing_values_imputed': missing_before,
    }

# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
def predict_process(X_imp):
    """
    Run XGBoost prediction on preprocessed data.
    Returns list of result dicts.
    """
    model, _, _, _ = _load_artifacts()
    probs = model.predict_proba(X_imp)[:, 1]
    preds = (probs >= 0.5).astype(int)

    results = []
    for i in range(len(probs)):
        prob = float(probs[i])
        pred = int(preds[i])
        if prob < 0.3:
            risk = 'LOW'
        elif prob < 0.6:
            risk = 'MEDIUM'
        else:
            risk = 'HIGH'

        results.append({
            'index': i,
            'prediction': 'FAIL' if pred == 1 else 'PASS',
            'prediction_int': pred,
            'failure_probability': prob,
            'risk': risk,
        })
    return results

# ---------------------------------------------------------------------------
# SHAP
# ---------------------------------------------------------------------------
def compute_shap(X_imp):
    """
    Compute SHAP values for preprocessed records using TreeExplainer.
    Returns shap_values array and feature_cols list.
    """
    import shap
    model, _, _, feature_cols = _load_artifacts()
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_imp)
    base_value = float(explainer.expected_value)
    return shap_values, feature_cols, base_value

def get_top_shap_features(shap_values_row, feature_cols, top_n=20):
    """
    Get ranked list of features by absolute SHAP for a single record.
    """
    abs_shap = np.abs(shap_values_row)
    sorted_idx = np.argsort(abs_shap)[::-1][:top_n]

    features = []
    for idx in sorted_idx:
        features.append({
            'feature': f'F{feature_cols[idx]}',
            'feature_index': int(feature_cols[idx]),
            'shap_value': float(shap_values_row[idx]),
            'abs_shap': float(abs_shap[idx]),
            'direction': '↑' if shap_values_row[idx] > 0 else '↓',
        })
    return features

def get_global_shap_ranking():
    """Load pre-computed global SHAP ranking from Phase 12."""
    if os.path.exists(GLOBAL_SHAP_PATH):
        df = pd.read_csv(GLOBAL_SHAP_PATH)
        return df
    return None

# ---------------------------------------------------------------------------
# Demo data
# ---------------------------------------------------------------------------
def load_demo_records(n=5):
    """Load a few demo SECOM records for demonstration."""
    if not os.path.exists(SECOM_DATA_PATH):
        return None
    data = pd.read_csv(SECOM_DATA_PATH, sep=' ', header=None, nrows=n)
    return data

# ---------------------------------------------------------------------------
# Model metadata
# ---------------------------------------------------------------------------
def get_model_info():
    return {
        'name': 'XGBoost (scale_pos_weight)',
        'task': 'Process Failure Prediction',
        'input_features': '434 (after filtering)',
        'original_features': '590',
        'validation_strategy': 'Chronological 70/15/15',
        'test_accuracy': '95.76%',
        'test_fail_f1': '0.00%',
        'test_roc_auc': '0.668',
        'test_pr_auc': '0.067',
        'status': 'EXPERIMENTAL',
        'artifact': 'secom_xgboost.pkl',
        'limitation': 'Model failed to detect any failures in the chronological test set due to extreme class imbalance and temporal process drift.',
    }
