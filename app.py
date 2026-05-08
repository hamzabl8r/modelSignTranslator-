from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends, Security, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
from typing import List, Optional
from pathlib import Path
from datetime import datetime
import shutil
import pickle
import os
import uuid
import numpy as np


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_PATH = BASE_DIR / "model.p"

DATA_DIR.mkdir(exist_ok=True)

app = FastAPI(title="MyTransApp - Sign Language Backend")


# ==================== CORS ====================

ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*")

if ALLOWED_ORIGINS == "*":
    allow_origins = ["*"]
else:
    allow_origins = [origin.strip() for origin in ALLOWED_ORIGINS.split(",") if origin.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=False if allow_origins == ["*"] else True,
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
        model_bundle = None
        print("⚠️ model.p not found — prediction endpoint unavailable")


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

def clean_label_name(label: str) -> str:
    return label.strip().upper().replace(" ", "_")


def get_sign_labels() -> List[str]:
    labels = []

    for entry in DATA_DIR.iterdir():
        if entry.is_dir():
            labels.append(entry.name)

    return sorted(labels)


def count_images_in_dir(directory: Path) -> int:
    extensions = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")
    total = 0

    for ext in extensions:
        total += len(list(directory.glob(ext)))

    return total


def get_public_file_url(request: Request, file_path: Path) -> str:
    relative_path = file_path.relative_to(DATA_DIR)
    return str(request.base_url).rstrip("/") + f"/uploads/{relative_path.as_posix()}"


def extract_model_and_labels(bundle):
    """
    Supports multiple model.p formats:
    1. {"model": model, "labels": labels}
    2. {"model": model, "classes": classes}
    3. {"model": model, "label_encoder": encoder}
    4. direct sklearn model object
    """
    model = None
    labels = None
    label_encoder = None

    if isinstance(bundle, dict):
        model = bundle.get("model") or bundle.get("classifier") or bundle.get("clf")
        labels = bundle.get("labels") or bundle.get("classes") or bundle.get("labels_dict")
        label_encoder = bundle.get("label_encoder") or bundle.get("encoder")
    else:
        model = bundle

    return model, labels, label_encoder


def convert_prediction_to_label(prediction, labels=None, label_encoder=None):
    value = prediction

    if isinstance(prediction, (list, tuple, np.ndarray)):
        value = prediction[0]

    if label_encoder is not None:
        try:
            decoded = label_encoder.inverse_transform([value])
            return str(decoded[0])
        except Exception:
            pass

    if labels is not None:
        if isinstance(labels, dict):
            return str(labels.get(value, labels.get(str(value), value)))

        if isinstance(labels, (list, tuple)):
            try:
                index = int(value)
                if 0 <= index < len(labels):
                    return str(labels[index])
            except Exception:
                pass

    return str(value)


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
        signs = []

        for label in get_sign_labels():
            label_dir = DATA_DIR / label

            signs.append({
                "label": label,
                "image_count": count_images_in_dir(label_dir),
            })

        return {"signs": signs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/signs/{label}/images")
async def get_sign_images(label: str, request: Request):
    try:
        clean_label = clean_label_name(label)
        sign_dir = DATA_DIR / clean_label

        if not sign_dir.exists() or not sign_dir.is_dir():
            raise HTTPException(status_code=404, detail="Sign not found")

        extensions = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")
        image_files = []

        for ext in extensions:
            image_files.extend(sign_dir.glob(ext))

        image_files = sorted(image_files)

        images = [
            {
                "filename": image.name,
                "url": get_public_file_url(request, image),
            }
            for image in image_files
        ]

        return {
            "label": clean_label,
            "count": len(images),
            "images": images,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/signs/contribute")
async def contribute(
    label: str = Form(...),
    image: UploadFile = File(...),
    uploadedAt: Optional[str] = Form(None),
    _: Optional[str] = Depends(verify_api_key),
):
    try:
        clean_label = clean_label_name(label)

        if not clean_label:
            raise HTTPException(status_code=400, detail="Label is required")

        if not image.content_type or not image.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Uploaded file must be an image")

        label_dir = DATA_DIR / clean_label
        label_dir.mkdir(parents=True, exist_ok=True)

        original_ext = Path(image.filename or "").suffix.lower()

        if original_ext not in [".jpg", ".jpeg", ".png"]:
            original_ext = ".jpg"

        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        unique_id = uuid.uuid4().hex[:8]
        filename = f"{timestamp}_{unique_id}{original_ext}"

        destination = label_dir / filename

        with destination.open("wb") as buffer:
            shutil.copyfileobj(image.file, buffer)

        return {
            "status": "ok",
            "message": "Image uploaded successfully",
            "label": clean_label,
            "filename": filename,
            "uploadedAt": uploadedAt,
            "path": str(destination),
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict")
async def predict(request: LandmarksRequest):
    try:
        if model_bundle is None:
            raise HTTPException(
                status_code=503,
                detail="Model not loaded. Please train and upload model.p",
            )

        if not request.landmarks:
            raise HTTPException(status_code=400, detail="No landmarks provided")

        if len(request.landmarks) != 84:
            raise HTTPException(
                status_code=400,
                detail=f"Expected 84 landmark values, got {len(request.landmarks)}",
            )

        model, labels, label_encoder = extract_model_and_labels(model_bundle)

        if model is None:
            raise HTTPException(status_code=500, detail="Invalid model bundle")

        data = np.asarray(request.landmarks, dtype=np.float32).reshape(1, -1)

        prediction = model.predict(data)
        label = convert_prediction_to_label(prediction, labels, label_encoder)

        probability = None

        if hasattr(model, "predict_proba"):
            try:
                probabilities = model.predict_proba(data)[0]
                probability = float(np.max(probabilities))
            except Exception:
                probability = None

        return {
            "res": label,
            "prediction": label,
            "confidence": probability,
        }

    except HTTPException:
        raise
    except Exception as e:
        print("❌ Prediction error:", str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/reload-model")
async def reload_model(_: Optional[str] = Depends(verify_api_key)):
    load_model()

    return {
        "status": "ok",
        "model_loaded": model_bundle is not None,
    }


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))

<<<<<<< HEAD
    uvicorn.run(app, host="0.0.0.0", port=port)
=======
    uvicorn.run(app, host="0.0.0.0", port=port)
>>>>>>> 7b05ad02be46e96d9abfc7f55ad67e5baa1150d5
