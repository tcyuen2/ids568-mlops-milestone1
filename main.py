"""
FastAPI Iris Classifier API

This module implements a production-ready prediction service for the Iris dataset.
It demonstrates key MLOps concepts:
- Pydantic request/response validation
- Deterministic model loading at startup (not per-request)
- Proper HTTP status codes (200, 422, 500)
- Health endpoints for orchestration

Lifecycle Position: Model Artifact → API Service → Consumer
This service sits between the trained model artifact and downstream consumers,
providing a standardized REST interface for predictions.
"""
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Load model (only once at startup)
model = joblib.load("model.pkl")

app = FastAPI(
    title="Iris Classifier API",
    version="1.0.0"
)

# 1. Request model
class PredictionRequest(BaseModel):
    sepal_length: float = Field(..., gt=0, description="Sepal length in cm")
    sepal_width: float = Field(..., gt=0, description="Sepal width in cm")
    petal_length: float = Field(..., gt=0, description="Petal length in cm")
    petal_width: float = Field(..., gt=0, description="Petal width in cm")

# 2. Response model (ADD THIS!)
class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    model_version: str

# 3. Endpoint (must come AFTER the classes)
@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    features = np.array([[
        request.sepal_length,
        request.sepal_width,
        request.petal_length,
        request.petal_width
    ]])
    
    prediction_idx = model.predict(features)[0]
    confidence = model.predict_proba(features)[0][prediction_idx]
    
    target_names = ["setosa", "versicolor", "virginica"]
    
    return PredictionResponse(
        prediction=target_names[prediction_idx],
        confidence=float(confidence),
        model_version="1.0.0"
    )

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }

