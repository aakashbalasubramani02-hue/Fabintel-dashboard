import os
import pickle
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score, 
    balanced_accuracy_score, 
    precision_recall_fscore_support, 
    confusion_matrix, 
    classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
import cv2

def get_label_map():
    return {
        'Center': 0, 'Donut': 1, 'Edge-Loc': 2, 'Edge-Ring': 3, 
        'Loc': 4, 'Near-full': 5, 'Random': 6, 'Scratch': 7, 'none': 8
    }

def resize_wafer(img, size=(128, 128)):
    return cv2.resize(img, size, interpolation=cv2.INTER_NEAREST)

def sparse_focal_loss(gamma=2.0, alpha=0.25):
    def focal_loss_fn(y_true, y_pred):
        y_true = tf.cast(y_true, tf.int32)
        y_true_one_hot = tf.one_hot(y_true, depth=tf.shape(y_pred)[1])
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        cross_entropy = -y_true_one_hot * tf.math.log(y_pred)
        weight = alpha * y_true_one_hot * tf.math.pow(1 - y_pred, gamma)
        loss = weight * cross_entropy
        return tf.reduce_sum(loss, axis=1)
    return focal_loss_fn

def main():
    print("--- FINAL LOCKED TEST EVALUATION (V2) ---")
    
    # V2 Validation results (H3_RotFlip)
    val_metrics = {
        "Accuracy": 0.9819,
        "Balanced Accuracy": 0.9188,
        "Macro F1": 0.9278,
        "Weighted F1": 0.9821,
        "Scratch F1": 0.8691
    }
    
    model_path = 'd:/ALK/models/H3_RotFlip.keras'
    custom_objects = {'focal_loss_fn': sparse_focal_loss()}
    model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
    
    print("Loading test data...")
    with open('d:/ALK/data/test.pkl', 'rb') as f:
        df_test = pickle.load(f)
        
    label_map = get_label_map()
    idx2label = {v: k for k, v in label_map.items()}
    class_names = [idx2label[i] for i in range(len(idx2label))]
    
    print("Preparing 128x128 test images...")
    X_test = np.array([resize_wafer(img) for img in df_test['waferMap'].values])
    X_test = np.expand_dims(X_test, axis=-1)
    y_test = df_test['label'].values
    
    print("Generating predictions on untouched test set...")
    probs = model.predict(X_test, batch_size=128)
    preds = np.argmax(probs, axis=1)
    
    print("Calculating overall metrics...")
    acc = accuracy_score(y_test, preds)
    bal_acc = balanced_accuracy_score(y_test, preds)
    
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(y_test, preds, average='macro')
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(y_test, preds, average='weighted')
    
    overall_metrics = {
        "Accuracy": acc,
        "Balanced Accuracy": bal_acc,
        "Macro Precision": macro_p,
        "Macro Recall": macro_r,
        "Macro F1": macro_f1,
        "Weighted Precision": weighted_p,
        "Weighted Recall": weighted_r,
        "Weighted F1": weighted_f1
    }
    
    print("Calculating per-class metrics...")
    report_dict = classification_report(y_test, preds, target_names=class_names, output_dict=True)
    df_report = pd.DataFrame(report_dict).transpose()
    
    # Save JSON and CSV
    os.makedirs('d:/ALK/results/phase10', exist_ok=True)
    with open('d:/ALK/results/phase10/v2_test_metrics.json', 'w') as f:
        json.dump(overall_metrics, f, indent=4)
        
    df_report.to_csv('d:/ALK/results/phase10/v2_test_classification_report.csv')
    
    print("Generating confusion matrices...")
    cm = confusion_matrix(y_test, preds)
    cm_norm = confusion_matrix(y_test, preds, normalize='true')
    
    # Raw CM
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Final V2 Test Confusion Matrix (Raw)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('d:/ALK/results/phase10/v2_test_confusion_matrix.png')
    plt.close()
    
    # Normalized CM
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_norm, annot=True, fmt='.3f', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Final V2 Test Confusion Matrix (Normalized)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('d:/ALK/results/phase10/v2_test_normalized_confusion_matrix.png')
    plt.close()
    
    # Per-class F1 Visualization
    per_class_f1 = [report_dict[cls]['f1-score'] for cls in class_names]
    plt.figure(figsize=(10, 6))
    sns.barplot(x=class_names, y=per_class_f1, palette='viridis')
    plt.title('Final Test Per-Class F1 Score')
    plt.ylabel('F1 Score')
    plt.xticks(rotation=45)
    plt.ylim(0, 1.05)
    for i, v in enumerate(per_class_f1):
        plt.text(i, v + 0.01, f"{v:.3f}", ha='center')
    plt.tight_layout()
    plt.savefig('d:/ALK/results/final_per_class_f1.png')
    plt.close()
    
    # V1 vs V2 Comparison
    v1_metrics = {
        "Accuracy": 0.9602,
        "Balanced Accuracy": 0.6663,
        "Macro F1": 0.6929,
        "Weighted F1": 0.9594,
        "Scratch F1": 0.7148,
        "Near-full F1": 0.4560,
        "Loc F1": 0.5340
    }
    
    v2_test_metrics = {
        "Accuracy": acc,
        "Balanced Accuracy": bal_acc,
        "Macro F1": macro_f1,
        "Weighted F1": weighted_f1,
        "Scratch F1": report_dict['Scratch']['f1-score'],
        "Near-full F1": report_dict['Near-full']['f1-score'],
        "Loc F1": report_dict['Loc']['f1-score']
    }
    
    comp_data = []
    for m in v1_metrics.keys():
        v1_val = v1_metrics[m]
        v2_val = v2_test_metrics[m]
        comp_data.append({
            "Metric": m,
            "V1 Test": f"{v1_val*100:.2f}%",
            "V2 Test": f"{v2_val*100:.2f}%",
            "Improvement": f"{(v2_val - v1_val)*100:.2f}%",
            "Improvement_Raw": v2_val - v1_val
        })
        
    df_v1_v2 = pd.DataFrame(comp_data)
    df_v1_v2.to_csv('d:/ALK/results/phase10/v1_vs_v2_comparison.csv', index=False)
    
    # Plot V1 vs V2
    plt.figure(figsize=(12, 6))
    x = np.arange(len(v1_metrics))
    width = 0.35
    v1_vals = [v1_metrics[m] for m in v1_metrics.keys()]
    v2_vals = [v2_test_metrics[m] for m in v1_metrics.keys()]
    plt.bar(x - width/2, v1_vals, width, label='V1 Test', color='lightcoral')
    plt.bar(x + width/2, v2_vals, width, label='V2 Test', color='forestgreen')
    plt.xticks(x, list(v1_metrics.keys()))
    plt.title('V1 vs V2 Final Test Set Performance')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('d:/ALK/results/phase10/v1_vs_v2_comparison.png')
    plt.close()
    
    # Generalization Analysis
    gen_data = []
    for m in ["Accuracy", "Balanced Accuracy", "Macro F1"]:
        val_score = val_metrics[m]
        test_score = v2_test_metrics[m]
        gap = val_score - test_score
        gen_data.append({
            "Metric": m,
            "V2 Validation": f"{val_score*100:.2f}%",
            "V2 Test": f"{test_score*100:.2f}%",
            "Generalization Gap": f"{gap*100:.2f}%"
        })
    df_gen = pd.DataFrame(gen_data)
    df_gen.to_csv('d:/ALK/results/phase10/v2_generalization_analysis.csv', index=False)
    
    print("\n=== V1 VS V2 FINAL TEST METRICS ===")
    print(df_v1_v2[['Metric', 'V1 Test', 'V2 Test', 'Improvement']].to_string(index=False))
    
    print("\nBest Performing Classes (F1):")
    cls_f1s = [(cls, report_dict[cls]['f1-score']) for cls in class_names]
    cls_f1s.sort(key=lambda x: x[1], reverse=True)
    for k, v in cls_f1s[:3]:
        print(f"  {k}: {v:.4f}")
        
    print("\nWorst Performing Classes (F1):")
    for k, v in cls_f1s[-3:]:
        print(f"  {k}: {v:.4f}")
        
    print("\nCompleted.")

if __name__ == "__main__":
    main()
