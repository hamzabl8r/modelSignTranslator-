from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import pickle
import numpy as np

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict to your frontend domain in production
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

# Load model once at startup
try:
    model_dict = pickle.load(open('./model.p', 'rb'))
    model = model_dict['model']
except FileNotFoundError:
    raise RuntimeError("model.p not found. Please ensure the model file exists.")


class LandmarkData(BaseModel):
    landmarks: List[float]


@app.get("/")
async def root():
    return {"status": "MediSign AI Server is running"}


@app.post("/predict")
async def predict(data: dict):
    try:
        landmarks = data.get('landmarks')
        if not landmarks:
            return {"res": "..."}

        input_data = np.array(landmarks).reshape(1, -1)
        
        received_count = input_data.shape[1]
        
        if received_count == 42:
            input_data = np.pad(input_data, ((0, 0), (0, 42)), mode='constant')
        
        prediction = model.predict(input_data)
        return {"res": str(prediction[0])}
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        return {"res": "error", "details": str(e)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)