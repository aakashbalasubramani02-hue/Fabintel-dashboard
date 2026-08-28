"""
FABINTEL — Wafer Defect Inference Backend
Loads the V2 ResNet-Light model and provides prediction + Grad-CAM.
"""
import os
import json
import numpy as np
import cv2
import tensorflow as tf

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'models', 'H3_RotFlip.keras')
LABEL_MAP_PATH = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'models', 'label_map.json')
DEMO_DATA_PATH = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data', 'val.pkl')

MODEL_PATH = os.path.normpath(MODEL_PATH)
LABEL_MAP_PATH = os.path.normpath(LABEL_MAP_PATH)
DEMO_DATA_PATH = os.path.normpath(DEMO_DATA_PATH)

INPUT_SIZE = (128, 128)

# ---------------------------------------------------------------------------
# Focal loss (needed for model loading)
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Model + label map (lazy-loaded, cached)
# ---------------------------------------------------------------------------
_model = None
_label_map = None
_idx2label = None
_last_conv_layer = None

def _load_model():
    global _model, _last_conv_layer
    if _model is None:
        _model = tf.keras.models.load_model(
            MODEL_PATH,
            custom_objects={'focal_loss_fn': sparse_focal_loss()}
        )
        # Find last conv layer dynamically
        for layer in reversed(_model.layers):
            if 'conv' in layer.name.lower():
                _last_conv_layer = layer.name
                break
    return _model

def _load_label_map():
    global _label_map, _idx2label
    if _label_map is None:
        with open(LABEL_MAP_PATH, 'r') as f:
            _label_map = json.load(f)
        _idx2label = {v: k for k, v in _label_map.items()}
    return _label_map, _idx2label

def get_class_names():
    """Return ordered list of class names."""
    lm, idx2l = _load_label_map()
    return [idx2l[i] for i in range(len(idx2l))]

# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
def resize_wafer(img, size=INPUT_SIZE):
    """Resize wafer map using nearest-neighbor interpolation."""
    return cv2.resize(img.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST)

def preprocess_wafer(img):
    """Full preprocessing: resize → expand dims → float32."""
    resized = resize_wafer(img, INPUT_SIZE)
    arr = np.expand_dims(resized, axis=(0, -1)).astype(np.float32)
    return resized, arr

# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
def predict_wafer(img):
    """
    Run inference on a raw 2D wafer map numpy array.
    Returns dict with class, confidence, probabilities, resized image.
    """
    model = _load_model()
    _, idx2label = _load_label_map()

    resized, arr = preprocess_wafer(img)
    probs = model.predict(arr, verbose=0)[0]
    pred_idx = int(np.argmax(probs))
    confidence = float(probs[pred_idx])

    return {
        'class': idx2label[pred_idx],
        'class_index': pred_idx,
        'confidence': confidence,
        'probabilities': {idx2label[i]: float(probs[i]) for i in range(len(probs))},
        'resized': resized,
        'input_array': arr,
    }

# ---------------------------------------------------------------------------
# Grad-CAM
# ---------------------------------------------------------------------------
def make_gradcam_heatmap(img_array, pred_index=None):
    """Generate Grad-CAM heatmap for a preprocessed image array."""
    model = _load_model()
    grad_model = tf.keras.models.Model(
        inputs=[model.inputs],
        outputs=[model.get_layer(_last_conv_layer).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_output = conv_output[0]
    heatmap = conv_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()

def generate_gradcam_images(raw_img, resized_img, input_array, pred_index=None):
    """
    Generate original RGB, Grad-CAM heatmap, and overlay images.
    Returns dict with 'original', 'heatmap', 'overlay' as uint8 RGB arrays.
    """
    heatmap = make_gradcam_heatmap(input_array, pred_index)

    # Convert raw wafer to RGB (0=black, 1=gray, 2=white)
    img_rgb = np.zeros((*raw_img.shape, 3), dtype=np.uint8)
    img_rgb[raw_img == 1] = [128, 128, 128]
    img_rgb[raw_img == 2] = [255, 255, 255]

    # Resize heatmap to raw image dimensions
    heatmap_resized = cv2.resize(heatmap, (raw_img.shape[1], raw_img.shape[0]))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    # Overlay
    overlay = cv2.addWeighted(img_rgb, 0.6, heatmap_colored, 0.4, 0)

    return {
        'original': img_rgb,
        'heatmap': heatmap_colored,
        'overlay': overlay,
    }

# ---------------------------------------------------------------------------
# Demo data
# ---------------------------------------------------------------------------
_demo_data = None

def load_demo_samples(n_per_class=3):
    """Load a small set of demo samples from the validation split."""
    global _demo_data
    if _demo_data is not None:
        return _demo_data

    import pickle
    with open(DEMO_DATA_PATH, 'rb') as f:
        df = pickle.load(f)

    _, idx2label = _load_label_map()
    samples = []
    for lbl_idx in range(9):
        subset = df[df['label'] == lbl_idx]
        if len(subset) > n_per_class:
            subset = subset.sample(n=n_per_class, random_state=42)
        for _, row in subset.iterrows():
            samples.append({
                'waferMap': row['waferMap'],
                'label': int(row['label']),
                'failureType': idx2label[int(row['label'])],
            })
    _demo_data = samples
    return _demo_data

def get_model_info():
    """Return metadata about the loaded wafer model."""
    return {
        'name': 'ResNet-Light V2',
        'architecture': 'Lightweight ResNet (GAP)',
        'input_resolution': '128 × 128',
        'loss': 'Focal Loss (γ=2.0, α=0.25)',
        'classes': 9,
        'test_accuracy': '95.03%',
        'test_macro_f1': '69.09%',
        'test_balanced_accuracy': '69.69%',
        'status': 'VALIDATED',
        'artifact': 'H3_RotFlip.keras',
    }
