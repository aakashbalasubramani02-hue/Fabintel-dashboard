import numpy as np
import pandas as pd
import tensorflow as tf
import cv2
import pickle

class WaferDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, pkl_path, batch_size=32, target_size=(64, 64), shuffle=True, augment=False, aug_rotation=False, aug_flip=False, aug_prob=1.0):
        """
        Custom data generator for Wafer Maps.
        """
        self.batch_size = batch_size
        self.target_size = target_size
        self.shuffle = shuffle
        self.augment = augment
        self.aug_rotation = aug_rotation
        self.aug_flip = aug_flip
        self.aug_prob = aug_prob
        
        # Load dataset
        with open(pkl_path, 'rb') as f:
            df = pickle.load(f)
            
        self.wafer_maps = df['waferMap'].values
        self.labels = df['label'].values
        self.indices = np.arange(len(self.wafer_maps))
        
        if self.shuffle:
            np.random.shuffle(self.indices)
            
    def __len__(self):
        return int(np.ceil(len(self.wafer_maps) / self.batch_size))
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
            
    def __getitem__(self, index):
        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        
        X = np.empty((len(batch_indices), *self.target_size, 1), dtype=np.float32)
        y = np.empty((len(batch_indices),), dtype=np.int32)
        
        for i, idx in enumerate(batch_indices):
            wafer = self.wafer_maps[idx]
            label = self.labels[idx]
            
            # Resize using nearest neighbor
            resized = cv2.resize(wafer.astype(np.uint8), (self.target_size[1], self.target_size[0]), interpolation=cv2.INTER_NEAREST)
            
            # Expand dims to match (H, W, 1)
            resized = np.expand_dims(resized, axis=-1)
            
            # Apply augmentations if specified
            if self.augment:
                # Basic spatial augmentations
                # Random rotation (90, 180, 270)
                if getattr(self, 'aug_rotation', True) and np.random.rand() < self.aug_prob:
                    k = np.random.randint(1, 4)
                    resized = np.rot90(resized, k=k)
                # Random flip
                if getattr(self, 'aug_flip', True) and np.random.rand() < self.aug_prob:
                    resized = np.flip(resized, axis=np.random.choice([0, 1]))
                    
            X[i,] = resized
            y[i] = label
            
        return X, y

if __name__ == "__main__":
    # Test generator
    gen = WaferDataGenerator('d:/ALK/data/val.pkl', batch_size=4)
    X, y = gen[0]
    print(f"Batch X shape: {X.shape}")
    print(f"Batch y shape: {y.shape}")
