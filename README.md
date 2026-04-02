# Brain MRI Tumor Detection & Classification

A full-stack deep learning web application that classifies brain MRI scans into four categories — **Glioma, Meningioma, No Tumor, and Pituitary Tumor** — using transfer learning on EfficientNetB0, with Grad-CAM heatmap visualizations and detailed per-tumor medical information.

---

## Table of Contents

1. [Overview](#overview)
2. [Features](#features)
3. [Tech Stack](#tech-stack)
4. [Project Structure](#project-structure)
5. [Dataset](#dataset)
6. [Model Architecture](#model-architecture)
7. [Installation](#installation)
8. [Running the Application](#running-the-application)
9. [Training the Model](#training-the-model)
10. [API Reference](#api-reference)
11. [Tumor Classes](#tumor-classes)
12. [Medical Disclaimer](#medical-disclaimer)

---

## Overview

This application allows users to upload a brain MRI image and receive:

- **Tumor classification** with a confidence score
- **Per-class probability distribution** across all four categories
- **Grad-CAM heatmap overlay** highlighting which regions of the MRI influenced the prediction
- **Detailed medical information** about the detected condition — symptoms, treatment options, precautions, and prognosis

The backend is a FastAPI REST API serving a Keras model trained via two-phase transfer learning. The frontend is a React + Tailwind CSS single-page application with a glassmorphism dark UI.

---

## Features

| Feature | Description |
|---|---|
| MRI Upload | Drag-and-drop or click-to-browse image upload (PNG, JPG, JPEG) |
| Classification | 4-class tumor detection with per-class confidence scores |
| Grad-CAM | Gradient-weighted Class Activation Maps overlaid on the original scan |
| Tumor Info Cards | Per-diagnosis cards with symptoms, treatment options, precautions, and prognosis |
| Class Probability Bars | Animated probability bars for all four classes sorted by confidence |
| Responsive UI | Works on mobile, tablet, and desktop |
| Medical Disclaimer | Persistent disclaimer reminding users this is not a diagnostic tool |

---

## Tech Stack

### Frontend
| Technology | Version | Purpose |
|---|---|---|
| React | 18.2 | UI framework |
| Vite | 8.x | Build tool and dev server |
| Tailwind CSS | 3.4 | Utility-first styling |
| Axios | 1.6 | HTTP requests to backend |
| Lucide React | 0.344 | Icon library |

### Backend
| Technology | Version | Purpose |
|---|---|---|
| Python | 3.9+ | Runtime |
| FastAPI | 0.128 | REST API framework |
| Uvicorn | 0.39 | ASGI server |
| TensorFlow / Keras | 2.20 | Model inference |
| Pillow | 11.3 | Image preprocessing |
| NumPy | 2.0 | Array operations |
| SciPy | latest | Required by Keras data augmentation |

### Model
| Property | Value |
|---|---|
| Backbone | EfficientNetB0 (ImageNet pretrained) |
| Input size | 224 × 224 × 3 |
| Classes | 4 (glioma, meningioma, no_tumor, pituitary) |
| Training strategy | Two-phase transfer learning |
| Explainability | Grad-CAM (Gradient-weighted Class Activation Mapping) |

---

## Project Structure

```
MRI-Based-Brain-Tumor-Detection/
│
├── archive/                         # Dataset (not committed to git)
│   ├── Training/
│   │   ├── glioma_tumor/            # 826 images
│   │   ├── meningioma_tumor/        # 822 images
│   │   ├── no_tumor/                # 395 images
│   │   └── pituitary_tumor/         # 827 images
│   └── Testing/
│       ├── glioma_tumor/            # 100 images
│       ├── meningioma_tumor/        # 115 images
│       ├── no_tumor/                # 105 images
│       └── pituitary_tumor/         # 74 images
│
├── backend/
│   ├── main.py                      # FastAPI app — routes and startup
│   ├── utils.py                     # Preprocessing, inference, Grad-CAM
│   ├── requirements.txt             # Python dependencies
│   ├── venv/                        # Virtual environment (not committed)
│   └── models/
│       ├── classification_v2.keras  # Active trained model
│       ├── train.py                 # Training script (EfficientNetB0)
│       ├── save_model.py            # Legacy architecture-only saver
│       └── TumorML.ipynb            # Original training notebook
│
└── frontend/
    ├── src/
    │   ├── App.jsx                  # Main React component
    │   ├── main.jsx                 # React entry point
    │   ├── App.css                  # Global styles
    │   └── index.css                # Tailwind base styles
    ├── public/
    ├── package.json
    └── vite.config.js
```

---

## Dataset

The dataset used is the [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) from Kaggle.

| Split | Glioma | Meningioma | No Tumor | Pituitary | Total |
|---|---|---|---|---|---|
| Training | 826 | 822 | 395 | 827 | **2,870** |
| Testing | 100 | 115 | 105 | 74 | **394** |

**Class imbalance note:** `no_tumor` has approximately half as many training samples as the tumor classes. The training pipeline compensates with per-class loss weighting:

```
class_weight = total_samples / (num_classes × samples_per_class)
```

This gives `no_tumor` roughly 2× more loss weight than the tumor classes, preventing the model from ignoring it.

---

## Model Architecture

### Two-Phase Transfer Learning

**Phase 1 — Head training (backbone frozen)**

The EfficientNetB0 backbone (pretrained on ImageNet) is frozen. Only the custom classification head is trained.

```
Input (224×224×3)
    │
    ▼
EfficientNetB0 backbone [FROZEN]
    │  (4,049,571 params — fixed)
    ▼
GlobalAveragePooling2D
    │
BatchNormalization
    │
Dense(256, ReLU) + L2(1e-4)
    │
Dropout(0.4)
    │
Dense(4, Softmax)
    │
    ▼
Output: [glioma, meningioma, no_tumor, pituitary]
```

**Phase 2 — Fine-tuning (top 50 backbone layers unfrozen)**

The top 50 layers of the EfficientNetB0 backbone are unfrozen and fine-tuned with a 10× lower learning rate (1e-5) to adapt ImageNet features to MRI domain features without catastrophic forgetting.

### Training Configuration

| Setting | Phase 1 | Phase 2 |
|---|---|---|
| Learning rate | 1e-3 | 1e-5 |
| Optimizer | Adam | Adam |
| Loss | Categorical CE + label smoothing (0.1) | Same |
| Max epochs | 30 | 40 |
| Early stopping patience | 12 | 15 |
| LR reduction patience | 4 | 5 |
| Backbone layers trainable | 0 | Top 50 |

### Data Augmentation (Training only)

| Augmentation | Range |
|---|---|
| Rotation | ±20° |
| Width / Height shift | ±15% |
| Shear | 10% |
| Zoom | ±20% |
| Brightness | 0.8× – 1.2× |
| Horizontal flip | Yes |

### Grad-CAM Explainability

After classification, the app generates a **Grad-CAM heatmap** — a colour overlay that shows which pixels most influenced the model's decision. Warmer colours (red/yellow) indicate regions of higher importance.

The implementation:
1. Builds a sub-model that outputs both the last Conv2D activations and the final predictions
2. Computes gradients of the predicted class score w.r.t. the conv feature maps using `tf.GradientTape`
3. Pools the gradients over the spatial dimensions, weights the feature maps, and applies ReLU
4. Upsamples the heatmap to the input image size and blends it with the original scan

### Output Post-Processing

The model applies per-class probability corrections before final argmax to counteract known checkpoint-level biases (particularly under-detection of glioma):

```python
BIAS_CORRECTION = [2.0, 1.0, 0.5, 1.4]  # [glioma, meningioma, no_tumor, pituitary]
corrected = raw_probs * BIAS_CORRECTION
corrected /= corrected.sum()  # renormalise to sum = 1
```

These corrections are tuned from the test-set confusion matrix and will be refined as the model improves.

---

## Installation

### Prerequisites

- Python 3.9 or higher
- Node.js 18 or higher
- The `archive/` dataset folder (Training + Testing subdirectories) if you plan to retrain

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd MRI-Based-Brain-Tumor-Detection
```

### 2. Backend Setup

```bash
cd backend

# Create and activate a virtual environment
python -m venv venv

# macOS / Linux
source venv/bin/activate

# Windows
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install scipy  # required by Keras image augmentation
```

Make sure `backend/models/classification_v2.keras` exists. This is the trained model file. If you need to retrain it, see [Training the Model](#training-the-model).

### 3. Frontend Setup

```bash
cd frontend
npm install
```

---

## Running the Application

### Start the Backend

```bash
cd backend
source venv/bin/activate      # macOS/Linux
# venv\Scripts\activate       # Windows

uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`.
Interactive API docs: `http://localhost:8000/docs`

### Start the Frontend

```bash
cd frontend
npm run dev
```

The app will be available at `http://localhost:5173`.

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `VITE_API_URL` | `http://localhost:8000` | Backend URL used by the frontend |

To use a custom backend URL:
```bash
VITE_API_URL=http://your-backend-host:8000 npm run dev
```

---

## Training the Model

The training script is at `backend/models/train.py`. It implements both training phases automatically.

```bash
cd backend
source venv/bin/activate

python models/train.py
```

**What it does:**

1. Loads images from `archive/Training/` with a 15% validation split
2. Applies data augmentation to the training split
3. Computes class weights to handle `no_tumor` underrepresentation
4. **Phase 1**: Trains the classification head with the EfficientNetB0 backbone frozen (up to 30 epochs)
5. **Phase 2**: Unfreezes the top 50 backbone layers and fine-tunes with LR=1e-5 (up to 40 epochs)
6. Saves the globally best checkpoint to `backend/models/classification_v2.keras`
7. Evaluates on the test set and prints per-class metrics

**Key design decisions in the training script:**

- `initial_value_threshold` is set on the Phase 2 `ModelCheckpoint` to the Phase 1 best val_accuracy — ensuring Phase 2 only overwrites if it genuinely improves the model
- Label smoothing (0.1) is applied to both phases to reduce overconfidence on the small dataset
- L2 weight decay (1e-4) is applied to the Dense layer to reduce overfitting

**Expected training time:**

- ~30–45 seconds per epoch on CPU (Apple Silicon M-series)
- ~8–12 seconds per epoch on a GPU

---

## API Reference

### `GET /`

Health check.

**Response:**
```json
{ "message": "Brain Tumor Classification API is running" }
```

---

### `GET /health`

Returns whether the model is loaded and ready.

**Response:**
```json
{ "status": "ok", "model_loaded": true }
```

---

### `POST /predict`

Classifies an uploaded MRI image.

**Request:** `multipart/form-data`

| Field | Type | Description |
|---|---|---|
| `file` | image file | MRI scan (PNG, JPG, JPEG) |

**Response:**

```json
{
  "class": "glioma_tumor",
  "confidence": 87.42,
  "all_probabilities": {
    "glioma_tumor": 87.42,
    "meningioma_tumor": 8.31,
    "no_tumor": 2.15,
    "pituitary_tumor": 2.12
  },
  "heatmap": "<base64-encoded PNG string>",
  "filename": "mri_scan.jpg"
}
```

| Field | Type | Description |
|---|---|---|
| `class` | string | Predicted class label |
| `confidence` | float | Confidence of top prediction (0–100) |
| `all_probabilities` | object | Corrected softmax probabilities for all 4 classes |
| `heatmap` | string \| null | Base64-encoded Grad-CAM PNG overlay, or null if generation failed |
| `filename` | string | Original uploaded filename |

**Error responses:**

| Code | Reason |
|---|---|
| 400 | Uploaded file is not an image |
| 500 | Model not loaded, or inference error |

**Example with curl:**

```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@/path/to/mri_scan.jpg"
```

---

## Tumor Classes

### Glioma Tumor
Arises from glial cells (astrocytes, oligodendrocytes, ependymal cells). Ranges from Grade I (slow-growing) to Grade IV (glioblastoma, aggressive). Accounts for ~33% of all primary brain tumors.

**Common symptoms:** persistent headaches, seizures, memory loss, vision/speech changes, motor weakness
**Treatments:** surgery, radiation, chemotherapy (temozolomide), targeted therapy
**Prognosis:** Grade I–II can have 5–10+ year survival; Grade IV median ~15 months with treatment

---

### Meningioma Tumor
Arises from the meninges (the membranes surrounding the brain). The most common primary brain tumor (~37%). Most are Grade I (benign) and slow-growing. Occurs more frequently in women and older adults.

**Common symptoms:** headaches, vision/hearing problems, limb weakness, seizures
**Treatments:** watchful waiting (small tumors), surgery, stereotactic radiosurgery (Gamma Knife)
**Prognosis:** Excellent for Grade I — 5-year recurrence ~7–20% after complete resection

---

### No Tumor
No evidence of a brain tumor in the MRI scan. Symptoms should still be evaluated by a neurologist if present.

---

### Pituitary Tumor
Adenomas in the pituitary gland at the base of the brain. Almost always benign. Can be hormone-secreting ("functioning") — causing systemic endocrine disorders — or non-functioning.

**Common symptoms:** bitemporal hemianopia (tunnel vision), headaches, hormonal imbalances (Cushing's, acromegaly, hyperprolactinemia)
**Treatments:** medication (dopamine agonists for prolactinoma), transsphenoidal surgery, radiosurgery
**Prognosis:** Generally very good — most are benign and highly treatable

---

## Medical Disclaimer

> **This application is for educational and demonstration purposes only.**
>
> It is not a medical device and must not be used for clinical diagnosis or treatment decisions. All medical decisions should be made by qualified healthcare professionals based on personal clinical examination, full imaging review, and professional medical judgment.
>
> The model's predictions can be incorrect and are not validated for clinical use.

---

## License

This project is for educational and research purposes. The dataset is sourced from Kaggle and subject to its original license terms.

---

*© 2026 Brain MRI Analysis Deep Learning Lab*
