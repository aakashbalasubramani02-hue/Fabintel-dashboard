import os
import json
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, balanced_accuracy_score, roc_auc_score, average_precision_score, confusion_matrix

import xgboost as xgb

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

def load_and_split():
    print("Loading data...")
    data = pd.read_csv('d:/ALK/secom_data/secom.data', sep=' ', header=None)
    try:
        labels_df = pd.read_csv('d:/ALK/secom_data/secom_labels.data', delim_whitespace=True, header=None, parse_dates=[[1, 2]])
        labels_df.columns = ['Timestamp', 'Label']
    except Exception:
        with open('d:/ALK/secom_data/secom_labels.data', 'r') as f:
            lines = f.readlines()
        labels = [int(line.split(' ')[0]) for line in lines]
        timestamps = [' '.join(line.split(' ')[1:]).strip().strip('"') for line in lines]
        labels_df = pd.DataFrame({'Label': labels, 'Timestamp': pd.to_datetime(timestamps, format='%d/%m/%Y %H:%M:%S', errors='coerce')})
        
    df = data.copy()
    df['Label'] = (labels_df['Label'] == 1).astype(int) # 0 for pass, 1 for fail
    df['Timestamp'] = labels_df['Timestamp']
    
    # Sort chronologically
    df = df.sort_values(by='Timestamp').reset_index(drop=True)
    
    n = len(df)
    train_idx = int(0.7 * n)
    val_idx = int(0.85 * n)
    
    train_df = df.iloc[:train_idx].copy()
    val_df = df.iloc[train_idx:val_idx].copy()
    test_df = df.iloc[val_idx:].copy()
    
    print(f"Train size: {len(train_df)} (Fail: {train_df['Label'].sum()})")
    print(f"Val size: {len(val_df)} (Fail: {val_df['Label'].sum()})")
    print(f"Test size: {len(test_df)} (Fail: {test_df['Label'].sum()})")
    
    return train_df, val_df, test_df

def create_preprocessing_pipeline(train_df):
    print("Creating preprocessing pipeline...")
    # Identify constant and >50% missing features from training data only to prevent leakage
    missing_pct = train_df.drop(columns=['Label', 'Timestamp']).isnull().sum() / len(train_df)
    high_missing_cols = missing_pct[missing_pct > 0.5].index.tolist()
    
    constant_cols = []
    for col in train_df.drop(columns=['Label', 'Timestamp']).columns:
        if col not in high_missing_cols:
            if train_df[col].dropna().nunique() <= 1 or train_df[col].dropna().var() < 1e-6:
                constant_cols.append(col)
                
    drop_cols = list(set(high_missing_cols + constant_cols))
    print(f"Dropping {len(drop_cols)} features based on training data.")
    
    feature_cols = [c for c in train_df.drop(columns=['Label', 'Timestamp']).columns if c not in drop_cols]
    
    # We will just return the columns to keep and standard sklearn transformers
    imputer = SimpleImputer(strategy='median')
    scaler = StandardScaler()
    
    # Fit imputer and scaler on training
    X_train_raw = train_df[feature_cols]
    imputer.fit(X_train_raw)
    X_train_imp = imputer.transform(X_train_raw)
    scaler.fit(X_train_imp)
    
    # Save artifacts
    os.makedirs('d:/ALK/models/secom', exist_ok=True)
    with open('d:/ALK/models/secom/secom_feature_cols.json', 'w') as f:
        json.dump(feature_cols, f)
    with open('d:/ALK/models/secom/secom_imputer.pkl', 'wb') as f:
        pickle.dump(imputer, f)
    with open('d:/ALK/models/secom/secom_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
        
    return feature_cols, imputer, scaler

def apply_preprocessing(df, feature_cols, imputer, scaler, scale=True):
    X = df[feature_cols]
    X_imp = imputer.transform(X)
    if scale:
        X_out = scaler.transform(X_imp)
    else:
        X_out = X_imp
    return X_out, df['Label'].values

def evaluate_model(y_true, y_pred, y_prob, name, out_dir):
    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    if y_prob is not None:
        try:
            roc = roc_auc_score(y_true, y_prob)
            pr = average_precision_score(y_true, y_prob)
        except:
            roc = 0
            pr = 0
    else:
        roc = 0
        pr = 0
        
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Pass (0)', 'Fail (1)'], 
                yticklabels=['Pass (0)', 'Fail (1)'])
    plt.title(f'Confusion Matrix: {name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f'{out_dir}/{name.replace(" ", "_")}_cm.png')
    plt.close()
    
    return {
        'Model': name,
        'Accuracy': acc,
        'Balanced Accuracy': bal_acc,
        'Fail Precision': prec,
        'Fail Recall': rec,
        'Fail F1': f1,
        'ROC-AUC': roc,
        'PR-AUC': pr
    }

def main():
    out_dir = 'd:/ALK/results/phase12'
    os.makedirs(out_dir, exist_ok=True)
    
    train_df, val_df, test_df = load_and_split()
    feature_cols, imputer, scaler = create_preprocessing_pipeline(train_df)
    
    # Create datasets
    X_train_scaled, y_train = apply_preprocessing(train_df, feature_cols, imputer, scaler, scale=True)
    X_val_scaled, y_val = apply_preprocessing(val_df, feature_cols, imputer, scaler, scale=True)
    X_test_scaled, y_test = apply_preprocessing(test_df, feature_cols, imputer, scaler, scale=True)
    
    X_train_unscaled, _ = apply_preprocessing(train_df, feature_cols, imputer, scaler, scale=False)
    X_val_unscaled, _ = apply_preprocessing(val_df, feature_cols, imputer, scaler, scale=False)
    X_test_unscaled, _ = apply_preprocessing(test_df, feature_cols, imputer, scaler, scale=False)
    
    results = []
    
    print("--- 1. Logistic Regression ---")
    lr = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
    lr.fit(X_train_scaled, y_train)
    y_pred_lr = lr.predict(X_test_scaled)
    y_prob_lr = lr.predict_proba(X_test_scaled)[:, 1]
    results.append(evaluate_model(y_test, y_pred_lr, y_prob_lr, "Logistic Regression", out_dir))
    
    print("--- 2. Random Forest ---")
    rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, max_depth=10)
    rf.fit(X_train_unscaled, y_train)
    y_pred_rf = rf.predict(X_test_unscaled)
    y_prob_rf = rf.predict_proba(X_test_unscaled)[:, 1]
    results.append(evaluate_model(y_test, y_pred_rf, y_prob_rf, "Random Forest", out_dir))
    
    print("--- 3. XGBoost ---")
    scale_pos_weight = sum(y_train == 0) / sum(y_train == 1)
    xgb_model = xgb.XGBClassifier(scale_pos_weight=scale_pos_weight, random_state=42, use_label_encoder=False, eval_metric='logloss', max_depth=5)
    xgb_model.fit(X_train_unscaled, y_train)
    y_pred_xgb = xgb_model.predict(X_test_unscaled)
    y_prob_xgb = xgb_model.predict_proba(X_test_unscaled)[:, 1]
    results.append(evaluate_model(y_test, y_pred_xgb, y_prob_xgb, "XGBoost", out_dir))
    
    print("--- 4. Artificial Neural Network ---")
    # Using exact architecture requested
    ann = Sequential([
        Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.AUC(name='prc', curve='PR')])
    
    # Class weights for ANN
    w0 = 1.0
    w1 = sum(y_train == 0) / sum(y_train == 1)
    class_weight = {0: w0, 1: w1}
    
    es = EarlyStopping(monitor='val_prc', mode='max', patience=15, restore_best_weights=True)
    
    ann.fit(X_train_scaled, y_train, validation_data=(X_val_scaled, y_val), epochs=100, batch_size=32, class_weight=class_weight, callbacks=[es], verbose=0)
    
    y_prob_ann = ann.predict(X_test_scaled).flatten()
    y_pred_ann = (y_prob_ann > 0.5).astype(int)
    results.append(evaluate_model(y_test, y_pred_ann, y_prob_ann, "Artificial Neural Network", out_dir))
    
    # Save best models
    with open('d:/ALK/models/secom/secom_xgboost.pkl', 'wb') as f:
        pickle.dump(xgb_model, f)
    ann.save('d:/ALK/models/secom/secom_ann.keras')
    
    # Comparison table
    df_results = pd.DataFrame(results)
    df_results.to_csv(f'{out_dir}/model_comparison.csv', index=False)
    print("\n=== MODEL COMPARISON ===")
    print(df_results.to_string(index=False))
    
    # Identify best model (prioritizing PR-AUC and Fail F1)
    best_model_name = df_results.sort_values(by=['PR-AUC', 'Fail F1'], ascending=False).iloc[0]['Model']
    print(f"\nSelected Model for metrics: {best_model_name}")
    best_model_name = "XGBoost" # Force XGBoost for SHAP tree explainer
    print(f"Forcing SHAP on: {best_model_name}")
    
    # --- SHAP EXPLAINABILITY ---
    print("Running SHAP Analysis...")
    if best_model_name == "XGBoost":
        explainer = shap.TreeExplainer(xgb_model)
        shap_values = explainer.shap_values(X_test_unscaled)
        
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_test_unscaled, feature_names=feature_cols, show=False)
        plt.tight_layout()
        plt.savefig(f'{out_dir}/shap_summary.png')
        plt.close()
        
        # Global feature importance (mean absolute SHAP)
        mean_shap = np.abs(shap_values).mean(axis=0)
        shap_df = pd.DataFrame({'Feature': feature_cols, 'Mean_Abs_SHAP': mean_shap})
        shap_df = shap_df.sort_values(by='Mean_Abs_SHAP', ascending=False)
        shap_df.head(20).to_csv(f'{out_dir}/top_process_risk_factors.csv', index=False)
        
        plt.figure(figsize=(10, 8))
        sns.barplot(x='Mean_Abs_SHAP', y='Feature', data=shap_df.head(20), palette='viridis')
        plt.title('Top 20 Process-Risk Candidate Factors (SHAP)')
        plt.xlabel('Mean |SHAP value| (Impact on model output)')
        plt.tight_layout()
        plt.savefig(f'{out_dir}/top_process_risk_factors.png')
        plt.close()
        
    elif best_model_name == "Random Forest":
        explainer = shap.TreeExplainer(rf)
        shap_values = explainer.shap_values(X_test_unscaled)
        # RF shap values is a list of arrays for classification, we want the positive class [1]
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
            
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_test_unscaled, feature_names=feature_cols, show=False)
        plt.tight_layout()
        plt.savefig(f'{out_dir}/shap_summary.png')
        plt.close()
        
        mean_shap = np.abs(shap_values).mean(axis=0)
        shap_df = pd.DataFrame({'Feature': feature_cols, 'Mean_Abs_SHAP': mean_shap})
        shap_df = shap_df.sort_values(by='Mean_Abs_SHAP', ascending=False)
        shap_df.head(20).to_csv(f'{out_dir}/top_process_risk_factors.csv', index=False)
        
        plt.figure(figsize=(10, 8))
        sns.barplot(x='Mean_Abs_SHAP', y='Feature', data=shap_df.head(20), palette='viridis')
        plt.title('Top 20 Process-Risk Candidate Factors (SHAP)')
        plt.tight_layout()
        plt.savefig(f'{out_dir}/top_process_risk_factors.png')
        plt.close()
    elif best_model_name == "Artificial Neural Network":
        # We need a background dataset for DeepExplainer
        background = X_train_scaled[np.random.choice(X_train_scaled.shape[0], 100, replace=False)]
        explainer = shap.DeepExplainer(ann, background)
        shap_values = explainer.shap_values(X_test_scaled)
        
        # shap_values for keras is a list of arrays or a single array
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
            
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_test_scaled, feature_names=feature_cols, show=False)
        plt.tight_layout()
        plt.savefig(f'{out_dir}/shap_summary.png')
        plt.close()
        
        mean_shap = np.abs(shap_values).mean(axis=0)
        shap_df = pd.DataFrame({'Feature': feature_cols, 'Mean_Abs_SHAP': mean_shap})
        shap_df = shap_df.sort_values(by='Mean_Abs_SHAP', ascending=False)
        shap_df.head(20).to_csv(f'{out_dir}/top_process_risk_factors.csv', index=False)
        
        plt.figure(figsize=(10, 8))
        sns.barplot(x='Mean_Abs_SHAP', y='Feature', data=shap_df.head(20), palette='viridis')
        plt.title('Top 20 Process-Risk Candidate Factors (SHAP)')
        plt.xlabel('Mean |SHAP value|')
        plt.tight_layout()
        plt.savefig(f'{out_dir}/top_process_risk_factors.png')
        plt.close()
        
        # Individual sample explanation for a False Negative
        fn_indices = np.where((y_test == 1) & (y_pred_ann == 0))[0]
        if len(fn_indices) > 0:
            idx = fn_indices[0]
            # Use waterfall plot or force plot. Waterfall requires an Explanation object
            exp = shap.Explanation(values=shap_values[idx], base_values=explainer.expected_value[0] if isinstance(explainer.expected_value, list) else explainer.expected_value, data=X_test_scaled[idx], feature_names=feature_cols)
            plt.figure(figsize=(10, 6))
            shap.waterfall_plot(exp, show=False)
            plt.tight_layout()
            plt.savefig(f'{out_dir}/shap_individual_fn.png')
            plt.close()
            
    else:
        print("SHAP not implemented for the selected model in this script version.")

    print("Process Complete.")

if __name__ == "__main__":
    main()
