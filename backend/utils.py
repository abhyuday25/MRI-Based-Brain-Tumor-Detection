import tensorflow as tf
import numpy as np
from PIL import Image
import io

# Model configuration
IMG_SIZE = (150, 150)
CLASS_LABELS = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']

def load_tumor_model(model_path: str):
    """Loads the trained .keras model."""
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def preprocess_image(image_bytes: bytes):
    """
    Preprocesses the uploaded image for prediction.
    Steps:
    1. Convert bytes to PIL Image
    2. Convert to RGB
    3. Resize to target size
    4. Normalize pixel values [0, 1]
    5. Add batch dimension
    """
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize(IMG_SIZE)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def get_prediction_results(model, processed_img):
    """
    Performs inference and returns the class and confidence score.
    """
    predictions = model.predict(processed_img)
    class_idx = np.argmax(predictions[0])
    confidence = float(predictions[0][class_idx])
    
    return {
        "class": CLASS_LABELS[class_idx],
        "confidence": confidence
    }
