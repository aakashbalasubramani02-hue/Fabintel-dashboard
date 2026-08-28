import os
import sys
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas.core.indexes

# Fix for older pandas pickles
sys.modules['pandas.indexes'] = pandas.core.indexes

def setup_dirs():
    dirs = [
        "d:/ALK/src",
        "d:/ALK/results/figures",
        "d:/ALK/results/metrics",
        "d:/ALK/data",
        "d:/ALK/models",
        "d:/ALK/app",
        "d:/ALK/notebooks"
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)

def main():
    setup_dirs()
    print("Loading dataset...")
    # Load dataset safely
    dataset_path = 'd:/ALK/LSWMD.pkl'
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset not found at {dataset_path}")
        return
    
    with open(dataset_path, 'rb') as f:
        df = pickle.load(f, encoding='latin1')
    
    print("\n--- DATASET INSPECTION ---")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print("\nData Types:")
    print(df.dtypes)
    print("\nMissing Values:")
    print(df.isnull().sum())
    
    # Process labels to clean strings from arrays
    df['failureNum'] = df.failureType
    df['trainTestNum'] = df.trianTestLabel
    
    # Clean the failureType column (it contains lists of strings or empty lists)
    df['failureType'] = df['failureType'].apply(lambda x: x[0][0] if len(x) > 0 and len(x[0]) > 0 else 'Unlabeled')
    df['trianTestLabel'] = df['trianTestLabel'].apply(lambda x: x[0][0] if len(x) > 0 and len(x[0]) > 0 else 'Unlabeled')
    
    print("\nLabel Distribution (failureType):")
    label_counts = df['failureType'].value_counts()
    print(label_counts)
    
    print("\nTrain/Test Split Labels:")
    print(df['trianTestLabel'].value_counts())
    
    print("\nLabeled vs Unlabeled:")
    unlabeled_count = (df['failureType'] == 'Unlabeled').sum()
    labeled_count = len(df) - unlabeled_count
    print(f"Labeled: {labeled_count}")
    print(f"Unlabeled: {unlabeled_count}")
    
    # Check wafer map dimensions (sample first 1000 for speed)
    print("\nWafer map dimensions (sample of 1000):")
    dim_sample = df['waferMap'].head(1000).apply(lambda x: x.shape)
    print(dim_sample.value_counts().head(5))
    
    print("\nGenerating Class Distribution Plot...")
    # Exclude unlabeled for the plot to see defect distribution clearly
    labeled_df = df[df['failureType'] != 'Unlabeled']
    plt.figure(figsize=(10, 6))
    labeled_df['failureType'].value_counts().plot(kind='bar')
    plt.title('Defect Class Distribution (Labeled Data)')
    plt.xlabel('Defect Class')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('d:/ALK/results/figures/class_distribution.png')
    plt.close()
    
    print("Generating Sample Wafer Maps Plot...")
    defect_classes = [c for c in labeled_df['failureType'].unique() if c != 'none']
    # add none at the end
    defect_classes.append('none')
    
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    axes = axes.ravel()
    
    for i, cls in enumerate(defect_classes):
        if i >= 9:
            break
        # Get one sample
        sample = labeled_df[labeled_df['failureType'] == cls].iloc[0]
        wafer_map = sample['waferMap']
        axes[i].imshow(wafer_map, cmap='viridis')
        axes[i].set_title(f"{cls}\nShape: {wafer_map.shape}")
        axes[i].axis('off')
        
    plt.tight_layout()
    plt.savefig('d:/ALK/results/figures/sample_wafers.png')
    plt.close()
    print("Exploration complete. Plots saved to d:/ALK/results/figures/")

if __name__ == "__main__":
    main()
