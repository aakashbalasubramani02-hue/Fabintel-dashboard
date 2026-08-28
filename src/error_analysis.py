import os
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import cv2

def get_label_map():
    return {
        'Center': 0, 'Donut': 1, 'Edge-Loc': 2, 'Edge-Ring': 3, 
        'Loc': 4, 'Random': 5, 'Scratch': 6, 'Near-full': 7, 'none': 8
    }

def resize_wafer(img, size=(64, 64)):
    return cv2.resize(img, size, interpolation=cv2.INTER_NEAREST)

def main():
    print("--- Starting Error Analysis ---")
    model_path = 'd:/ALK/models/Exp_D_ResNet.keras'
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return

    print("Loading model...")
    model = tf.keras.models.load_model(model_path)
    
    print("Loading data...")
    with open('d:/ALK/data/train.pkl', 'rb') as f:
        df_train = pickle.load(f)
        
    label_map = get_label_map()
    idx2label = {v: k for k, v in label_map.items()}
    
    _, df_val = train_test_split(df_train, test_size=0.2, random_state=42, stratify=df_train['label'])
    
    print("Preparing validation images for prediction...")
    X_val = np.array([resize_wafer(img) for img in df_val['waferMap'].values])
    X_val = np.expand_dims(X_val, axis=-1)
    y_val = df_val['label'].values
    
    print("Generating predictions...")
    probs = model.predict(X_val, batch_size=128)
    preds = np.argmax(probs, axis=1)
    confidences = np.max(probs, axis=1)
    
    # 1. Per-class error rate & Most confused pairs
    errors = (preds != y_val)
    print(f"\nOverall Validation Error Rate: {np.mean(errors):.2%}")
    
    conf_pairs = {}
    for true_lbl, pred_lbl in zip(y_val[errors], preds[errors]):
        pair = (idx2label[true_lbl], idx2label[pred_lbl])
        conf_pairs[pair] = conf_pairs.get(pair, 0) + 1
        
    sorted_pairs = sorted(conf_pairs.items(), key=lambda x: x[1], reverse=True)
    print("\n--- Most Confused Class Pairs (True -> Predicted) ---")
    for (t, p), count in sorted_pairs[:10]:
        print(f"{t} -> {p} : {count} times")
        
    # 2. Extract incorrect predictions
    df_errors = df_val[errors].copy()
    df_errors['predicted'] = preds[errors]
    df_errors['confidence'] = confidences[errors]
    
    # 3. Extract correct predictions
    df_correct = df_val[~errors].copy()
    df_correct['predicted'] = preds[~errors]
    df_correct['confidence'] = confidences[~errors]
    
    # Highest confidence incorrect
    highest_conf_err = df_errors.sort_values(by='confidence', ascending=False)
    print("\n--- Highest Confidence INCORRECT Predictions ---")
    for i, row in highest_conf_err.head(5).iterrows():
        print(f"True: {idx2label[row['label']]}, Pred: {idx2label[row['predicted']]}, Conf: {row['confidence']:.4f}")
        
    # Lowest confidence correct
    lowest_conf_corr = df_correct.sort_values(by='confidence', ascending=True)
    print("\n--- Lowest Confidence CORRECT Predictions ---")
    for i, row in lowest_conf_corr.head(5).iterrows():
        print(f"True: {idx2label[row['label']]}, Pred: {idx2label[row['predicted']]}, Conf: {row['confidence']:.4f}")

    # Plot representative misclassifications (Original Wafer)
    print("\nPlotting Highest Confidence Errors...")
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    axes = axes.flatten()
    
    for idx, (index, row) in enumerate(highest_conf_err.head(9).iterrows()):
        ax = axes[idx]
        img = row['waferMap']
        ax.imshow(img, cmap='inferno')
        true_lbl = idx2label[row['label']]
        pred_lbl = idx2label[row['predicted']]
        conf = row['confidence']
        ax.set_title(f"T: {true_lbl} | P: {pred_lbl}\nConf: {conf:.3f}", fontsize=10)
        ax.axis('off')
        
    plt.tight_layout()
    plt.savefig('d:/ALK/results/figures/high_conf_errors.png')
    plt.close()

    print("Error analysis complete. Results saved.")

if __name__ == "__main__":
    main()
