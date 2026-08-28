import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def main():
    print("--- SECOM Dataset Analysis ---")
    out_dir = 'd:/ALK/results/phase11_secom'
    os.makedirs(out_dir, exist_ok=True)
    
    # 1. Load Data
    print("Loading SECOM data...")
    # secom.data is space-separated
    data = pd.read_csv('d:/ALK/secom_data/secom.data', sep=' ', header=None)
    
    # secom_labels.data contains label and timestamp separated by space
    # but the timestamp has quotes or is a string. Actually it's something like "-1 19/07/2008 11:55:00"
    labels_df = pd.read_csv('d:/ALK/secom_data/secom_labels.data', sep=' ', header=None, parse_dates=False)
    # The label file has multiple spaces, or a quote. Let's process it more carefully if needed.
    # Typically: label, date, time
    try:
        labels_df = pd.read_csv('d:/ALK/secom_data/secom_labels.data', delim_whitespace=True, header=None, parse_dates=[[1, 2]])
        labels_df.columns = ['Timestamp', 'Label']
    except Exception as e:
        # fallback
        with open('d:/ALK/secom_data/secom_labels.data', 'r') as f:
            lines = f.readlines()
        labels = [int(line.split(' ')[0]) for line in lines]
        timestamps = [' '.join(line.split(' ')[1:]).strip().strip('"') for line in lines]
        labels_df = pd.DataFrame({'Label': labels, 'Timestamp': pd.to_datetime(timestamps, format='%d/%m/%Y %H:%M:%S', errors='coerce')})
    
    df = data.copy()
    num_samples = df.shape[0]
    num_features = df.shape[1]
    
    # 2. Dataset Structure
    print(f"Dataset Shape: {num_samples} samples, {num_features} features")
    
    # 3. Label Analysis
    label_counts = labels_df['Label'].value_counts()
    pass_count = label_counts.get(-1, 0)
    fail_count = label_counts.get(1, 0)
    
    class_dist = pd.DataFrame({
        'Label': ['Pass (-1)', 'Fail (1)'],
        'Count': [pass_count, fail_count],
        'Percentage': [pass_count/num_samples*100, fail_count/num_samples*100]
    })
    class_dist.to_csv(f'{out_dir}/secom_class_distribution.csv', index=False)
    
    plt.figure(figsize=(8, 6))
    sns.barplot(x='Label', y='Count', data=class_dist)
    plt.title('SECOM Class Distribution (Pass vs Fail)')
    for i, v in enumerate(class_dist['Count']):
        plt.text(i, v + 10, str(v), ha='center')
    plt.tight_layout()
    plt.savefig(f'{out_dir}/secom_class_distribution.png')
    plt.close()
    
    # 4. Missing-Value Analysis
    missing_per_feature = df.isnull().sum()
    missing_pct_per_feature = (missing_per_feature / num_samples) * 100
    missing_df = pd.DataFrame({
        'Feature_Index': missing_per_feature.index,
        'Missing_Count': missing_per_feature.values,
        'Missing_Percentage': missing_pct_per_feature.values
    })
    missing_df.to_csv(f'{out_dir}/secom_missing_values.csv', index=False)
    
    plt.figure(figsize=(10, 6))
    plt.hist(missing_df['Missing_Percentage'], bins=50, color='skyblue', edgecolor='black')
    plt.title('Distribution of Missing Value Percentages across Features')
    plt.xlabel('Percentage of Missing Values (%)')
    plt.ylabel('Number of Features')
    plt.tight_layout()
    plt.savefig(f'{out_dir}/secom_missing_values.png')
    plt.close()
    
    # 5. Feature Quality Analysis
    constant_features = []
    near_constant_features = []
    high_missing_features = []
    
    for col in df.columns:
        if missing_pct_per_feature[col] > 50:
            high_missing_features.append(col)
        else:
            valid_data = df[col].dropna()
            if len(valid_data.unique()) <= 1:
                constant_features.append(col)
            elif valid_data.var() < 1e-6:
                near_constant_features.append(col)
                
    quality_summary = pd.DataFrame({
        'Metric': ['Total Features', 'Constant Features', 'Near-Constant (Var < 1e-6)', 'High Missing (>50%)'],
        'Count': [num_features, len(constant_features), len(near_constant_features), len(high_missing_features)]
    })
    quality_summary.to_csv(f'{out_dir}/secom_feature_quality.csv', index=False)
    
    # 6. Failure-Related Analysis & Feature Association
    print("Calculating Associations...")
    valid_features = [c for c in df.columns if c not in constant_features and c not in high_missing_features]
    
    associations = []
    
    y = labels_df['Label'].values
    
    for col in valid_features:
        x = df[col].values
        mask = ~np.isnan(x)
        x_valid = x[mask]
        y_valid = y[mask]
        
        if len(np.unique(y_valid)) > 1 and len(x_valid) > 10:
            try:
                # Point-biserial correlation
                pbc, pbc_pvalue = stats.pointbiserialr(x_valid, y_valid)
                
                # T-test between pass and fail
                pass_vals = x_valid[y_valid == -1]
                fail_vals = x_valid[y_valid == 1]
                
                if len(pass_vals) > 0 and len(fail_vals) > 0:
                    t_stat, t_pvalue = stats.ttest_ind(pass_vals, fail_vals, equal_var=False)
                    
                    associations.append({
                        'Feature': col,
                        'Correlation': pbc,
                        'Abs_Correlation': abs(pbc) if not np.isnan(pbc) else 0,
                        'Corr_P_Value': pbc_pvalue,
                        'T_Stat': t_stat,
                        'T_P_Value': t_pvalue,
                        'Pass_Mean': np.mean(pass_vals),
                        'Fail_Mean': np.mean(fail_vals),
                        'Missing_Pct': missing_pct_per_feature[col]
                    })
            except:
                pass
                
    assoc_df = pd.DataFrame(associations).sort_values(by='Abs_Correlation', ascending=False)
    assoc_df.to_csv(f'{out_dir}/secom_feature_association.csv', index=False)
    
    # Top 20 features plot
    top_features = assoc_df.head(20)
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Abs_Correlation', y=top_features['Feature'].astype(str), data=top_features, palette='rocket')
    plt.title('Top 20 Process Features Associated with Failure (Point-Biserial Correlation)')
    plt.xlabel('Absolute Correlation |r|')
    plt.ylabel('SECOM Feature Index')
    plt.tight_layout()
    plt.savefig(f'{out_dir}/secom_top_features.png')
    plt.close()
    
    # Plot distributions of top 4 features
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    for i in range(min(4, len(top_features))):
        feature_idx = top_features.iloc[i]['Feature']
        sns.kdeplot(data=df[labels_df['Label'] == -1][feature_idx], ax=axes[i], label='Pass', fill=True)
        sns.kdeplot(data=df[labels_df['Label'] == 1][feature_idx], ax=axes[i], label='Fail', fill=True)
        axes[i].set_title(f'Distribution of Feature {feature_idx}')
        axes[i].legend()
    plt.tight_layout()
    plt.savefig(f'{out_dir}/secom_distribution_analysis.png')
    plt.close()
    
    # Summary stats output
    data_quality_summary = pd.DataFrame({
        'Metric': ['Num Samples', 'Num Features', 'Pass Count', 'Fail Count'],
        'Value': [num_samples, num_features, pass_count, fail_count]
    })
    data_quality_summary.to_csv(f'{out_dir}/secom_data_quality_summary.csv', index=False)

    print("SECOM Analysis Complete.")

if __name__ == "__main__":
    main()
