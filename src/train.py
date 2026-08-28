import os
import sys
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from data_loader import WaferDataGenerator
from models import build_baseline_cnn, build_resnet
from evaluate import evaluate_model
import argparse

def sparse_focal_loss(gamma=2.0, alpha=0.25):
    def focal_loss_fn(y_true, y_pred):
        y_true = tf.cast(y_true, tf.int32)
        # Convert sparse y_true to one-hot
        y_true_one_hot = tf.one_hot(y_true, depth=tf.shape(y_pred)[1])
        # Compute focal loss
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        cross_entropy = -y_true_one_hot * tf.math.log(y_pred)
        weight = alpha * y_true_one_hot * tf.math.pow(1 - y_pred, gamma)
        loss = weight * cross_entropy
        return tf.reduce_sum(loss, axis=1)
    return focal_loss_fn

def train_model(experiment_name="Exp_A_Baseline", use_class_weights=False, cap_weight=20.0, use_focal_loss=False, use_augmentation=False, aug_rotation=False, aug_flip=False, aug_prob=1.0, use_resnet=False, epochs=30, batch_size=128, img_size=64):
    print(f"\n--- Starting {experiment_name} ---")
    
    # Load label map
    with open('d:/ALK/models/label_map.json', 'r') as f:
        label_map = json.load(f)
    num_classes = len(label_map)
    
    # Data generators
    train_gen = WaferDataGenerator('d:/ALK/data/train.pkl', batch_size=batch_size, augment=use_augmentation, aug_rotation=aug_rotation, aug_flip=aug_flip, aug_prob=aug_prob, target_size=(img_size, img_size))
    val_gen = WaferDataGenerator('d:/ALK/data/val.pkl', batch_size=batch_size, shuffle=False, augment=False, target_size=(img_size, img_size))
    
    # Class weights calculation if requested
    class_weight_dict = None
    if use_class_weights:
        import pickle
        with open('d:/ALK/data/train.pkl', 'rb') as f:
            df_train = pickle.load(f)
        counts = df_train['label'].value_counts().to_dict()
        total = sum(counts.values())
        
        # Calculate standard inverse frequency weights
        raw_weights = {label: total / (num_classes * count) for label, count in counts.items()}
        
        # Cap the weights to prevent rare classes from causing unstable gradients
        MAX_WEIGHT = cap_weight
        MIN_WEIGHT = 0.1
        class_weight_dict = {label: max(MIN_WEIGHT, min(MAX_WEIGHT, w)) for label, w in raw_weights.items()}
        print(f"Using Capped Class Weights (Cap={cap_weight}):", class_weight_dict)
    
    if use_focal_loss:
        print("Using Focal Loss (gamma=2.0, alpha=0.25)...")
        loss_fn = sparse_focal_loss(gamma=2.0, alpha=0.25)
    else:
        loss_fn = 'sparse_categorical_crossentropy'
    
    if use_resnet:
        print("Building ResNet architecture...")
        model = build_resnet(input_shape=(img_size, img_size, 1), num_classes=num_classes)
    else:
        print("Building Baseline CNN architecture...")
        model = build_baseline_cnn(input_shape=(img_size, img_size, 1), num_classes=num_classes)
        
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=loss_fn,
        metrics=['accuracy']
    )
    
    model_path = f'd:/ALK/models/{experiment_name}.keras'
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1),
        ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True, verbose=1)
    ]
    
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=callbacks,
        class_weight=class_weight_dict
    )
    
    # Save training history
    with open(f'd:/ALK/results/metrics/{experiment_name}_history.json', 'w') as f:
        json.dump(history.history, f)
        
    print(f"\n--- Evaluating {experiment_name} on Validation Set ---")
    evaluate_model(model, val_gen, label_map, experiment_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str, default='Exp_A_Baseline')
    parser.add_argument('--weights', action='store_true', help="Use class weights")
    parser.add_argument('--cap_weight', type=float, default=20.0, help="Maximum cap for class weights")
    parser.add_argument('--focal_loss', action='store_true', help="Use focal loss instead of cross-entropy")
    parser.add_argument('--aug', action='store_true', help="Use augmentation")
    parser.add_argument('--aug_rotation', action='store_true', help="Use rotation augmentation")
    parser.add_argument('--aug_flip', action='store_true', help="Use flip augmentation")
    parser.add_argument('--aug_prob', type=float, default=1.0, help="Probability of applying augmentation")
    parser.add_argument('--resnet', action='store_true', help="Use ResNet model")
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--img_size', type=int, default=64)
    args = parser.parse_args()
    
    train_model(
        experiment_name=args.exp, 
        use_class_weights=args.weights,
        cap_weight=args.cap_weight,
        use_focal_loss=args.focal_loss,
        use_augmentation=args.aug, 
        aug_rotation=args.aug_rotation,
        aug_flip=args.aug_flip,
        aug_prob=args.aug_prob,
        use_resnet=args.resnet,
        epochs=args.epochs,
        img_size=args.img_size
    )
