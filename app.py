from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends, Security, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.security import APIKeyHeader
from fastapi.responses import FileResponse
from pydantic import BaseModel

from typing import List, Optional
from pathlib import Path
from datetime import datetime
import shutil
import pickle
import os

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

MODEL_PATH = BASE_DIR / "model.p"

DATA_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

app = FastAPI(title="MyTransApp - Sign Language Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/uploads", StaticFiles(directory=str(DATA_DIR)), name="uploads")

# ==================== MODEL LOADING ====================
model_bundle = None

def load_model():
    global model_bundle
    if MODEL_PATH.exists():
        with open(MODEL_PATH, "rb") as f:
            model_bundle = pickle.load(f)
        print(f"✅ Model loaded from {MODEL_PATH}")
    else:
        print("⚠️ model.p not found — prediction endpoint will be unavailable")

load_model()

# ==================== API KEY AUTH ====================
API_KEY = os.getenv("CONTRIBUTE_API_KEY", "")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def verify_api_key(key: Optional[str] = Security(api_key_header)):
    if API_KEY and key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return key

# ==================== SCHEMAS ====================
class LandmarksRequest(BaseModel):
    landmarks: List[float]

# ==================== HELPERS ====================
def get_sign_labels() -> List[str]:
    labels = set()
    for base in (DATA_DIR, DATA_DIR):
        try:
            for entry in base.iterdir():
                if entry.is_dir():
                    labels.add(entry.name)
        except PermissionError:
            pass
    return sorted(labels)

def count_images_in_dir(directory: Path) -> int:
    extensions = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")
    total = 0
    for ext in extensions:
        total += len(list(directory.glob(ext)))
    return total

# ==================== ROUTES ====================

@app.get("/")
async def root():
    return {
        "message": "Backend is running",
        "data_dir": str(DATA_DIR),
        "model_loaded": model_bundle is not None,
        "sign_count": len(get_sign_labels()),
    }

@app.get("/api/signs")
async def get_dictionary():
    try:
        return {"signs": get_sign_labels()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── GET /api/signs/:label/images ─────────────────────────────────────────────
@app.get("/api/signs/{label}/images")
async def get_sign_images(label: str, request: Request):
    clean_label = label.strip().upper().replace(" ", "_")

    # Check uploads/ first, then data/
    folder = DATA_DIR / clean_label
    if not folder.is_dir():
        folder = DATA_DIR / clean_label

    if not folder.is_dir():
        return {"images": []}

    allowed_ext = {".jpg", ".jpeg", ".png", ".webp", ".gif"}
    files = sorted([
        f.name for f in folder.iterdir()
        if f.suffix.lower() in allowed_ext
    ])

    # Build absolute URLs using the request's base URL (works with ngrok)
    base_url = str(request.base_url).rstrip("/")
    images = [f"{base_url}/uploads/{clean_label}/{f}" for f in files]

    return {"images": images}


@app.post("/api/signs/contribute")
async def contribute(
    label: str = Form(...),
    image: UploadFile = File(...),
    uploadedAt: Optional[str] = Form(None),
    _: Optional[str] = Depends(verify_api_key),
):
    try:
        clean_label = label.strip().upper().replace(" ", "_")

        upload_label_dir = DATA_DIR / clean_label
        upload_label_dir.mkdir(parents=True, exist_ok=True)

        data_label_dir = DATA_DIR / clean_label
        data_label_dir.mkdir(parents=True, exist_ok=True)

        existing = count_images_in_dir(data_label_dir)
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
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict")
async def predict(request: LandmarksRequest):
    try:
        if model_bundle is None:
            raise HTTPException(status_code=503, detail="Model not loaded. Please train and upload model.p")

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


@app.post("/reload-model")
async def reload_model(_: Optional[str] = Depends(verify_api_key)):
    load_model()
    return {
        "status": "ok",
        "model_loaded": model_bundle is not None,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)