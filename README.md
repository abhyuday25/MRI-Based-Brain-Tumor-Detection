# Brain MRI Tumor Detection & Classification

A production-ready web application for MRI-based brain tumor classification using deep learning. Users can upload MRI images to classify scans into one of four categories: **Glioma, Meningioma, No Tumor, or Pituitary Tumor**.

![App Screenshot](https://images.unsplash.com/photo-1559757148-5c350d0d3c56?auto=format&fit=crop&q=80&w=1000)

## Features

-   **Deep Learning Backend**: FastAPI server running a trained Keras model.
-   **Modern React UI**: Built with Vite and Tailwind CSS for speed and premium aesthetics.
-   **Drag & Drop Upload**: Easy and intuitive image selection.
-   **Real-time Classification**: Instant results with confidence scores.
-   **Responsive Design**: Works on mobile, tablet, and desktop.
-   **Glassmorphism Theme**: Dark medical-grade user interface.

## Tech Stack

-   **Frontend**: React.js, Tailwind CSS, Lucide Icons, Axios.
-   **Backend**: FastAPI (Python), TensorFlow/Keras, Pillow, Uvicorn.
-   **Model**: CNN Architecture (Input size: 150x150).

## Installation & Setup

### Prerequisites

-   Python 3.10+
-   Node.js 18+
-   `classification.keras` model file (trained)

### 1. Backend Setup

```bash
cd backend
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

pip install -r requirements.txt
```

**Note**: Ensure your `classification.keras` file is placed inside the `backend/model/` directory.

### 2. Frontend Setup

```bash
cd frontend
npm install
```

## Running the Application

### Start the Backend

```bash
cd backend
uvicorn main:app --reload
```
The server will run on `http://localhost:8000`.

### Start the Frontend

```bash
cd frontend
npm run dev
```
The application will be accessible at `http://localhost:3000`.

## API Usage

### `POST /predict`

**Input**: Multi-part form data with an `image` file.
**Output**: 

```json
{
  "class": "pituitary_tumor",
  "confidence": 99.5,
  "filename": "scan_01.jpg"
}
```

## Disclaimer

**Educational Purposes Only**: This application is a demonstration of deep learning and is not intended for diagnostic use. All medical decisions should be made by qualified healthcare professionals based on personal consultation.

---
© 2026 Brain MRI Analysis Deep Learning Lab