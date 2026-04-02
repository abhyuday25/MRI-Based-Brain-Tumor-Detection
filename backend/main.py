import os

import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from utils import (
    generate_gradcam_overlay,
    get_prediction_results,
    load_tumor_model,
    preprocess_image,
)

app = FastAPI(title="Brain Tumor Classification API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model on startup
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "classification_v2.keras")
model = None


@app.on_event("startup")
async def startup_event():
    global model
    if os.path.exists(MODEL_PATH):
        model = load_tumor_model(MODEL_PATH)
        if model:
            print("Model loaded successfully")
        else:
            print("Model file found but failed to load")
    else:
        print(f"Warning: Model file not found at {MODEL_PATH}")


@app.get("/")
async def root():
    return {"message": "Brain Tumor Classification API is running"}


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict")
async def predict_tumor(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded on server. Please check the model path.")

    try:
        contents = await file.read()
        processed_img = preprocess_image(contents)
        result = get_prediction_results(model, processed_img)
        heatmap = generate_gradcam_overlay(model, processed_img, contents, result["class_idx"])

        return {
            "class": result["class"],
            "confidence": round(result["confidence"] * 100, 2),
            "all_probabilities": result["all_probabilities"],
            "heatmap": heatmap,
            "filename": file.filename,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
