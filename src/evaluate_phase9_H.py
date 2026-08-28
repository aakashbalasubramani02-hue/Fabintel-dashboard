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
    # Needed for loading G5 model since it uses custom focal loss
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

def evaluate_model(model_path, X_val, y_val, class_names):
    custom_objects = {'focal_loss_fn': sparse_focal_loss()}
    model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
        
    probs = model.predict(X_val, batch_size=128)
    preds = np.argmax(probs, axis=1)
    
    acc = accuracy_score(y_val, preds)
    bal_acc = balanced_accuracy_score(y_val, preds)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(y_val, preds, average='macro')
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(y_val, preds, average='weighted')
    
    report = classification_report(y_val, preds, target_names=class_names, output_dict=True)
    
    cm = confusion_matrix(y_val, preds)
    return {
        'Accuracy': acc,
        'Balanced Accuracy': bal_acc,
        'Macro F1': macro_f1,
        'Weighted F1': weighted_f1,
        'Scratch F1': report['Scratch']['f1-score'],
        'Near-full F1': report['Near-full']['f1-score'],
        'Loc F1': report['Loc']['f1-score'],
        'Center F1': report['Center']['f1-score'],
        'Donut F1': report['Donut']['f1-score'],
        'none Precision': report['none']['precision'],
        'none Recall': report['none']['recall'],
        'report': report,
        'cm': cm
    }

def main():
    print("--- Phase 9 Experiment H Evaluation ---")
    
    experiments = [
        ('H1_RotOnly', 'Rotation Only'),
        ('H2_FlipOnly', 'Flip Only'),
        ('H3_RotFlip', 'Rotation + Flip (V1 Augmentation)'),
        ('H4_ReducedAug', 'Reduced Augmentation (prob=0.5)')
    ]
    
    print("Loading validation data...")
    with open('d:/ALK/data/val.pkl', 'rb') as f:
        df_val = pickle.load(f)
        
    label_map = get_label_map()
    idx2label = {v: k for k, v in label_map.items()}
    class_names = [idx2label[i] for i in range(len(idx2label))]
    
    print("Preparing 128x128 validation images...")
    X_val = np.array([resize_wafer(img) for img in df_val['waferMap'].values])
    X_val = np.expand_dims(X_val, axis=-1)
    y_val = df_val['label'].values
    
    results = []
    
    os.makedirs('d:/ALK/results/phase9', exist_ok=True)
    
    for exp_id, exp_name in experiments:
        model_path = f'd:/ALK/models/{exp_id}.keras'
        if not os.path.exists(model_path):
            print(f"Skipping {exp_id} - Model not found")
            continue
            
        print(f"\nEvaluating {exp_id}...")
        res = evaluate_model(model_path, X_val, y_val, class_names)
        
        # Save confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(res['cm'], annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.title(f'Confusion Matrix: {exp_id}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(f'd:/ALK/results/phase9/{exp_id}_cm.png')
        plt.close()
        
        results.append({
            'Experiment': exp_id,
            'Augmentation': exp_name,
            'Accuracy': res['Accuracy'],
            'Balanced Accuracy': res['Balanced Accuracy'],
            'Macro F1': res['Macro F1'],
            'Weighted F1': res['Weighted F1'],
            'Scratch F1': res['Scratch F1'],
            'Near-full F1': res['Near-full F1'],
            'Loc F1': res['Loc F1'],
            'Center F1': res['Center F1'],
            'Donut F1': res['Donut F1']
        })
        
    df_results = pd.DataFrame(results)
    df_results.to_csv('d:/ALK/results/phase9/experiment_h_comparison.csv', index=False)
    
    print("\n=== Experiment H Results ===")
    print(df_results.to_string(index=False))
    
    # Generate Comparison Plots
    # 1. Macro F1 & Balanced Accuracy
    plt.figure(figsize=(12, 6))
    x = np.arange(len(df_results))
    width = 0.35
    plt.bar(x - width/2, df_results['Macro F1'], width, label='Macro F1', color='skyblue')
    plt.bar(x + width/2, df_results['Balanced Accuracy'], width, label='Balanced Accuracy', color='salmon')
    plt.xticks(x, df_results['Experiment'])
    plt.title('Macro F1 and Balanced Accuracy by Augmentation Strategy')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('d:/ALK/results/phase9/H_MacroF1_BalAcc_Comparison.png')
    plt.close()
    
    # 2. Minority Class F1 comparison
    plt.figure(figsize=(14, 6))
    width = 0.15
    plt.bar(x - 2*width, df_results['Scratch F1'], width, label='Scratch')
    plt.bar(x - width, df_results['Near-full F1'], width, label='Near-full')
    plt.bar(x, df_results['Loc F1'], width, label='Loc')
    plt.bar(x + width, df_results['Center F1'], width, label='Center')
    plt.bar(x + 2*width, df_results['Donut F1'], width, label='Donut')
    plt.xticks(x, df_results['Experiment'])
    plt.title('Minority Class F1 Score Comparison')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('d:/ALK/results/phase9/H_Minority_F1_Comparison.png')
    plt.close()

    print("\nEvaluation Complete.")

if __name__ == "__main__":
    main()
