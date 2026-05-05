from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional
from pathlib import Path
from datetime import datetime
import shutil
import pickle

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
UPLOADS_DIR = BASE_DIR / "uploads"
MODEL_PATH = BASE_DIR / "model.p"
PICKLE_PATH = BASE_DIR / "data.pickle"

DATA_DIR.mkdir(exist_ok=True)
UPLOADS_DIR.mkdir(exist_ok=True)

app = FastAPI(title="MyTransApp - Sign Language Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/uploads", StaticFiles(directory=str(UPLOADS_DIR)), name="uploads")

model_bundle = None

def load_model():
    global model_bundle
    if MODEL_PATH.exists():
        with open(MODEL_PATH, "rb") as f:
            model_bundle = pickle.load(f)

load_model()

class LandmarksRequest(BaseModel):
    landmarks: List[float]

def get_sign_labels():
    """Read labels from DATA_DIR (used for training) and UPLOADS_DIR (contributed)."""
    labels = set()

    for folder in DATA_DIR.iterdir():
        if folder.is_dir():
            labels.add(folder.name)

    for folder in UPLOADS_DIR.iterdir():
        if folder.is_dir():
            labels.add(folder.name)

    return sorted(labels)

@app.get("/api/signs")
async def get_dictionary():
    try:
        return {"signs": get_sign_labels()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/signs/contribute")
async def contribute(
    label: str = Form(...),
    image: UploadFile = File(...),          # ← now accepts image, not video
    uploadedAt: Optional[str] = Form(None)
):
    """
    Save a contributed JPEG image into uploads/<LABEL>/.
    The file is also copied into data/<LABEL>/ so it can be used for retraining.
    """
    try:
        clean_label = label.strip().upper().replace(" ", "_")

        # Save under uploads/ for review
        upload_label_dir = UPLOADS_DIR / clean_label
        upload_label_dir.mkdir(parents=True, exist_ok=True)

        # Also save under data/ so retraining can pick it up immediately
        data_label_dir = DATA_DIR / clean_label
        data_label_dir.mkdir(parents=True, exist_ok=True)

        # Build unique filename based on existing count
        existing = len(list(data_label_dir.glob("*.jpg")))
        file_name = f"{existing}.jpg"

        upload_path = upload_label_dir / file_name
        data_path = data_label_dir / file_name

        contents = await image.read()

        with open(upload_path, "wb") as f:
            f.write(contents)

        with open(data_path, "wb") as f:
            f.write(contents)

        return {
            "status": "success",
            "label": clean_label,
            "uploadedAt": uploadedAt,
            "image_url": f"/uploads/{clean_label}/{file_name}",
            "total_images": existing + 1,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict")
async def predict(request: LandmarksRequest):
    try:
        if model_bundle is None:
            raise HTTPException(status_code=500, detail="Model file not loaded")

        model = (
            model_bundle["model"]
            if isinstance(model_bundle, dict) and "model" in model_bundle
            else model_bundle
        )

        prediction = model.predict([request.landmarks])[0]

        confidence = None
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba([request.landmarks])[0]
            confidence = float(max(probs))

        return {
            "res": prediction,
            "prediction": prediction,
            "confidence": confidence,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/")
async def root():
    return {
        "message": "Backend is running",
        "data_dir": str(DATA_DIR),
        "model_loaded": model_bundle is not None,
        "sign_count": len(get_sign_labels()),
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)