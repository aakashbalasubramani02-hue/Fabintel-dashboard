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

def resize_wafer(img, size=(128, 128)):
    return cv2.resize(img, size, interpolation=cv2.INTER_NEAREST)

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        inputs=[model.inputs], outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def main():
    print("--- Starting Phase 8: Explainable AI (Grad-CAM) ---")
    model_path = 'd:/ALK/models/Exp_F_128x128.keras'
    model = tf.keras.models.load_model(model_path)
    
    # Find the last conv layer dynamically
    # Look for the last layer that has 'conv' in the name
    last_conv_layer_name = None
    for layer in reversed(model.layers):
        if 'conv' in layer.name.lower():
            last_conv_layer_name = layer.name
            break
            
    if last_conv_layer_name is None:
        last_conv_layer_name = model.layers[-4].name # fallback
        
    print(f"Using {last_conv_layer_name} as the last convolutional layer for Grad-CAM.")

    print("Loading validation data...")
    with open('d:/ALK/data/train.pkl', 'rb') as f:
        df_train = pickle.load(f)
        
    label_map = get_label_map()
    idx2label = {v: k for k, v in label_map.items()}
    
    _, df_val = train_test_split(df_train, test_size=0.2, random_state=42, stratify=df_train['label'])
    
    # Prepare data
    print("Preparing 128x128 validation images...")
    X_val = np.array([resize_wafer(img) for img in df_val['waferMap'].values])
    X_val = np.expand_dims(X_val, axis=-1)
    y_val = df_val['label'].values
    
    # Generate predictions
    print("Generating predictions...")
    probs = model.predict(X_val, batch_size=128)
    preds = np.argmax(probs, axis=1)
    conf = np.max(probs, axis=1)
    
    # Add to dataframe for easy querying
    df_val = df_val.copy()
    df_val['pred'] = preds
    df_val['conf'] = conf
    df_val['is_correct'] = (df_val['label'] == df_val['pred'])
    
    # Find samples to visualize
    print("Selecting representative samples...")
    selected_indices = []
    
    # 1. Correct predictions across various classes
    for lbl in range(9):
        # get highest confidence correct for this class
        sub = df_val[(df_val['label'] == lbl) & (df_val['is_correct'] == True)]
        if len(sub) > 0:
            idx = sub.sort_values('conf', ascending=False).index[0]
            selected_indices.append(idx)
            
    # 2. High confidence incorrect predictions
    incorrect = df_val[df_val['is_correct'] == False].sort_values('conf', ascending=False)
    selected_indices.extend(incorrect.index[:7].tolist()) # Top 7 incorrect to make 16 total
    
    # Deduplicate just in case
    selected_indices = list(dict.fromkeys(selected_indices))
    
    # Generate and plot Grad-CAM
    print(f"Generating Grad-CAM for {len(selected_indices)} samples...")
    
    # Calculate grid size
    n_cols = 4
    n_rows = int(np.ceil(len(selected_indices) / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))
    axes = axes.flatten()
    
    for i, idx in enumerate(selected_indices):
        ax = axes[i]
        
        row = df_val.loc[idx]
        img_raw = row['waferMap']
        img_128 = resize_wafer(img_raw)
        img_array = np.expand_dims(img_128, axis=(0, -1)).astype(np.float32)
        
        true_lbl = idx2label[row['label']]
        pred_lbl = idx2label[row['pred']]
        confidence = row['conf']
        is_corr = row['is_correct']
        
        heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)
        
        # Resize heatmap to match raw image for superimposed display
        heatmap_resized = cv2.resize(heatmap, (img_raw.shape[1], img_raw.shape[0]))
        
        # Superimpose
        # Map raw image to RGB 0-255 for displaying
        # 0 = background (black), 1 = normal (gray), 2 = defect (white)
        img_rgb = np.zeros((*img_raw.shape, 3), dtype=np.uint8)
        img_rgb[img_raw == 1] = [128, 128, 128]
        img_rgb[img_raw == 2] = [255, 255, 255]
        
        # Colorize heatmap
        heatmap_col = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
        
        # Superimpose
        superimposed = cv2.addWeighted(img_rgb, 0.6, heatmap_col, 0.4, 0)
        
        ax.imshow(superimposed)
        title_color = 'green' if is_corr else 'red'
        ax.set_title(f"T: {true_lbl} | P: {pred_lbl}\nConf: {confidence:.2f}", color=title_color, fontsize=12)
        ax.axis('off')
        
    for i in range(len(selected_indices), len(axes)):
        axes[i].axis('off')
        
    plt.tight_layout()
    plt.savefig('d:/ALK/results/figures/gradcam_visualizations.png')
    plt.close()
    
    print("Grad-CAM visualization saved.")

if __name__ == "__main__":
    main()
