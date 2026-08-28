import os
import sys
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from sklearn.model_selection import train_test_split
import cv2
import pandas.core.indexes
sys.modules['pandas.indexes'] = pandas.core.indexes

def setup_dirs():
    os.makedirs('d:/ALK/data', exist_ok=True)
    os.makedirs('d:/ALK/models', exist_ok=True)
    os.makedirs('d:/ALK/results/figures', exist_ok=True)

def pad_wafer(wafer, target_size=(212, 212)):
    """Zero-pads a wafer map to the target size, placing it in the center."""
    h, w = wafer.shape
    pad_h = max(0, target_size[0] - h)
    pad_w = max(0, target_size[1] - w)
    
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left
    
    padded = np.pad(wafer, ((top, bottom), (left, right)), mode='constant', constant_values=0)
    return padded

def resize_wafer(wafer, target_size=(64, 64)):
    """Resizes a wafer map using nearest-neighbor interpolation."""
    # OpenCV expects (width, height)
    resized = cv2.resize(wafer.astype(np.uint8), (target_size[1], target_size[0]), interpolation=cv2.INTER_NEAREST)
    return resized

def main():
    setup_dirs()
    print("Loading original dataset...")
    dataset_path = 'd:/ALK/LSWMD.pkl'
    with open(dataset_path, 'rb') as f:
        df = pickle.load(f, encoding='latin1')
    
    # Clean labels
    df['failureType'] = df['failureType'].apply(lambda x: x[0][0] if len(x) > 0 and len(x[0]) > 0 else 'Unlabeled')
    df['trianTestLabel'] = df['trianTestLabel'].apply(lambda x: x[0][0] if len(x) > 0 and len(x[0]) > 0 else 'Unlabeled')
    
    # Filter labeled data
    labeled_df = df[df['failureType'] != 'Unlabeled'].copy()
    
    # Encode labels - use the exact original class names
    classes = ['Center', 'Donut', 'Edge-Loc', 'Edge-Ring', 'Loc', 'Near-full', 'Random', 'Scratch', 'none']
    label_map = {cls: i for i, cls in enumerate(classes)}
    with open('d:/ALK/models/label_map.json', 'w') as f:
        json.dump(label_map, f, indent=4)
    
    labeled_df['label'] = labeled_df['failureType'].map(label_map)
    
    # Split based on predefined trianTestLabel
    train_full = labeled_df[labeled_df['trianTestLabel'] == 'Training'].copy()
    test_df = labeled_df[labeled_df['trianTestLabel'] == 'Test'].copy()
    
    # Split train_full into train (80%) and val (20%) using stratified split
    print("Splitting Train into Train/Val (80/20)...")
    train_df, val_df = train_test_split(train_full, test_size=0.2, random_state=42, stratify=train_full['label'])
    
    print("\n--- CLASS COUNTS ---")
    def print_counts(name, d):
        print(f"\n{name} set ({len(d)} samples):")
        counts = d['failureType'].value_counts()
        print(counts)
        
    print_counts("Train", train_df)
    print_counts("Validation", val_df)
    print_counts("Test", test_df)
    
    # Save the splits to disk for fast loading during training
    print("\nSaving splits to disk (as pickle to preserve 2D arrays)...")
    train_df[['waferMap', 'label', 'failureType']].to_pickle('d:/ALK/data/train.pkl')
    val_df[['waferMap', 'label', 'failureType']].to_pickle('d:/ALK/data/val.pkl')
    test_df[['waferMap', 'label', 'failureType']].to_pickle('d:/ALK/data/test.pkl')
    
    print("\nGenerating sample processed plots...")
    # Generate a plot showing original vs padded vs resized for one sample of each class
    fig, axes = plt.subplots(9, 3, figsize=(10, 25))
    
    for i, cls in enumerate(classes):
        sample = train_df[train_df['failureType'] == cls].iloc[0]
        wafer = sample['waferMap']
        
        # Original
        axes[i, 0].imshow(wafer, cmap='viridis', interpolation='nearest')
        axes[i, 0].set_title(f"Original {cls}\n{wafer.shape}")
        axes[i, 0].axis('off')
        
        # Padded
        padded = pad_wafer(wafer, (212, 212))
        axes[i, 1].imshow(padded, cmap='viridis', interpolation='nearest')
        axes[i, 1].set_title(f"Padded\n{padded.shape}")
        axes[i, 1].axis('off')
        
        # Resized
        resized = resize_wafer(wafer, (64, 64))
        axes[i, 2].imshow(resized, cmap='viridis', interpolation='nearest')
        axes[i, 2].set_title(f"Resized (NN)\n{resized.shape}")
        axes[i, 2].axis('off')

    plt.tight_layout()
    plt.savefig('d:/ALK/results/figures/preprocessing_samples.png')
    plt.close()
    
    print("\nPreprocessing complete. Sample plot saved to d:/ALK/results/figures/preprocessing_samples.png")

if __name__ == "__main__":
    main()
