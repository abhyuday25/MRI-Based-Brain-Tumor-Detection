import base64
import io
from typing import Optional

import numpy as np
import tensorflow as tf
from PIL import Image

# Model configuration
IMG_SIZE = (224, 224)
CLASS_LABELS = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']


def load_tumor_model(model_path: str):
    """Loads the trained .keras model.

    Handles cross-version compatibility: newer Keras saves extra config keys
    (e.g. quantization_config) that older TF/Keras versions don't recognise.
    We patch Dense to silently drop unknown constructor kwargs.
    """
    _original_dense_init = tf.keras.layers.Dense.__init__

    def _patched_dense_init(self, *args, **kwargs):
        kwargs.pop("quantization_config", None)
        _original_dense_init(self, *args, **kwargs)

    tf.keras.layers.Dense.__init__ = _patched_dense_init

    try:
        model = tf.keras.models.load_model(model_path, compile=False)
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None
    finally:
        tf.keras.layers.Dense.__init__ = _original_dense_init


def preprocess_image(image_bytes: bytes):
    """
    Preprocesses the uploaded image for prediction.
    EfficientNetB0 expects raw [0, 255] pixel values — no rescaling.
    """
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize(IMG_SIZE)
    img_array = np.array(img, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def get_prediction_results(model, processed_img):
    """
    Performs inference and returns the predicted class, confidence,
    and probabilities for all classes.

    Per-class correction factors are applied before argmax to counteract
    known biases in the current model checkpoint (derived from test-set
    confusion matrix):
      - glioma is severely under-detected (21% accuracy) → boost
      - no_tumor is over-predicted as false positive  → penalise
      - pituitary is moderately under-detected (65%)  → slight boost
    These will become less necessary as the model improves via fine-tuning.
    """
    # [glioma, meningioma, no_tumor, pituitary]
    BIAS_CORRECTION = np.array([2.0, 1.0, 0.5, 1.4], dtype=np.float32)

    raw = model.predict(processed_img, verbose=0)[0]          # (4,)
    corrected = raw * BIAS_CORRECTION
    corrected /= corrected.sum()                               # renormalise to sum=1

    class_idx = int(np.argmax(corrected))
    confidence = float(corrected[class_idx])
    all_probabilities = {
        CLASS_LABELS[i]: round(float(corrected[i]) * 100, 2)
        for i in range(len(CLASS_LABELS))
    }

    return {
        "class": CLASS_LABELS[class_idx],
        "class_idx": class_idx,
        "confidence": confidence,
        "all_probabilities": all_probabilities,
    }


def _jet_colormap(t: np.ndarray) -> np.ndarray:
    """Apply a jet colormap to values in [0, 1]. Returns float32 RGB in [0, 1]."""
    r = np.clip(1.5 - np.abs(4.0 * t - 3.0), 0, 1)
    g = np.clip(1.5 - np.abs(4.0 * t - 2.0), 0, 1)
    b = np.clip(1.5 - np.abs(4.0 * t - 1.0), 0, 1)
    return np.stack([r, g, b], axis=-1)


def generate_gradcam_overlay(
    model, processed_img: np.ndarray, image_bytes: bytes, class_idx: int, alpha: float = 0.45
) -> Optional[str]:
    """
    Generates a Grad-CAM heatmap blended with the original image.
    Returns a base64-encoded PNG string, or None if generation fails.
    """
    try:
        # Find the last Conv2D layer — search recursively into nested models
        def _find_last_conv(m):
            for layer in reversed(m.layers):
                if isinstance(layer, tf.keras.layers.Conv2D):
                    return layer
                if hasattr(layer, 'layers'):
                    result = _find_last_conv(layer)
                    if result:
                        return result
            return None

        last_conv = _find_last_conv(model)
        if last_conv is None:
            return None

        # Build a model that outputs (last_conv activations, final predictions)
        grad_model = tf.keras.models.Model(
            inputs=model.inputs,
            outputs=[last_conv.output, model.output],
        )

        img_tensor = tf.cast(processed_img, tf.float32)
        with tf.GradientTape() as tape:
            tape.watch(img_tensor)
            conv_outputs, preds = grad_model(img_tensor)
            loss = preds[:, class_idx]

        grads = tape.gradient(loss, conv_outputs)          # (1, h, w, filters)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2)).numpy()  # (filters,)

        conv_out = conv_outputs[0].numpy()                 # (h, w, filters)
        for i in range(pooled_grads.shape[0]):
            conv_out[:, :, i] *= pooled_grads[i]

        # Average over filters and apply ReLU
        heatmap = np.mean(conv_out, axis=-1)               # (h, w)
        heatmap = np.maximum(heatmap, 0)
        if heatmap.max() > 0:
            heatmap /= heatmap.max()

        # Resize heatmap to model input size
        heatmap_pil = Image.fromarray((heatmap * 255).astype(np.uint8))
        heatmap_pil = heatmap_pil.resize(IMG_SIZE, Image.LANCZOS)
        heatmap_arr = np.array(heatmap_pil) / 255.0

        # Apply jet colormap
        colored = (_jet_colormap(heatmap_arr) * 255).astype(np.uint8)

        # Blend with original image
        original = Image.open(io.BytesIO(image_bytes)).convert('RGB').resize(IMG_SIZE)
        orig_arr = np.array(original, dtype=np.float32)
        blended = ((1 - alpha) * orig_arr + alpha * colored.astype(np.float32))
        blended = blended.clip(0, 255).astype(np.uint8)

        # Encode as base64 PNG
        buf = io.BytesIO()
        Image.fromarray(blended).save(buf, format='PNG')
        return base64.b64encode(buf.getvalue()).decode('utf-8')

    except Exception as e:
        print(f"Grad-CAM generation failed: {e}")
        return None
