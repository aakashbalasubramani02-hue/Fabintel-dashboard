import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, balanced_accuracy_score, f1_score
import json
import os

def evaluate_model(model, val_gen, label_map, experiment_name):
    print("Generating predictions...")
    y_pred_probs = model.predict(val_gen)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    # We need the true labels. Since val_gen might shuffle, let's make sure it doesn't.
    # We already passed shuffle=False to val_gen in train.py.
    y_true = val_gen.labels
    
    # Calculate metrics
    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    weighted_f1 = f1_score(y_true, y_pred, average='weighted')
    
    # Per-class metrics
    classes = [k for k, v in sorted(label_map.items(), key=lambda item: item[1])]
    report = classification_report(y_true, y_pred, target_names=classes, output_dict=True)
    
    print(f"\nResults for {experiment_name}:")
    print(f"Accuracy: {acc:.4f}")
    print(f"Balanced Accuracy: {bal_acc:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")
    
    print("\nPer-class Metrics:")
    for cls in classes:
        print(f"{cls} - Precision: {report[cls]['precision']:.4f}, Recall: {report[cls]['recall']:.4f}, F1: {report[cls]['f1-score']:.4f}")
    
    # Save metrics
    metrics = {
        'accuracy': acc,
        'balanced_accuracy': bal_acc,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        'classification_report': report
    }
    with open(f'd:/ALK/results/metrics/{experiment_name}_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)
        
    # Confusion Matrix Plot
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(f'Confusion Matrix - {experiment_name}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(f'd:/ALK/results/figures/{experiment_name}_cm.png')
    plt.close()
    
    print(f"\nEvaluation complete. Metrics and Confusion Matrix saved to d:/ALK/results/")

if __name__ == "__main__":
    pass
