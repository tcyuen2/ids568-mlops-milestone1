# Milestone 1: Web & Serverless Model Serving

A production-ready ML prediction service demonstrating MLOps best practices for deploying machine learning models as web services.

## Table of Contents

1. [Overview](#overview)
2. [ML Lifecycle Position](#ml-lifecycle-position)
3. [Setup Instructions](#setup-instructions)
4. [API Usage Examples](#api-usage-examples)
5. [Deployment URLs](#deployment-urls)
6. [Deployment Guide](#deployment-guide)
7. [Comparative Analysis](#comparative-analysis)
8. [Model-API Interaction](#model-api-interaction)

---

## Overview

This project implements an Iris flower classification service deployed in two patterns:

1. **FastAPI on Cloud Run**: Containerized web service with automatic HTTPS and scaling
2. **Google Cloud Function**: Serverless function with event-driven scaling

Both implementations serve the same prediction logic but demonstrate different trade-offs in latency, state management, and operational characteristics.

### Project Structure

```
milestone1/
├── main.py                    # FastAPI application
├── model.pkl                  # Trained scikit-learn model
├── requirements.txt           # Pinned dependencies
├── Dockerfile                 # Cloud Run container config
├── train_model.py             # Model training script
├── benchmark.py               # Latency comparison tool
├── cloud_function/            # Cloud Function code
│   ├── main.py                # Function entry point
│   ├── requirements.txt       # Function dependencies
│   └── model.pkl              # Model artifact (copy)
├── tests/                     # Test suite
│   └── test_api.py
└── README.md                  # This file
```

---

## ML Lifecycle Position

This deployment sits in the **Model Serving** stage of the ML lifecycle:

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│    Data     │ →  │  Training   │ →  │   Model     │ →  │  THIS API   │ →  │  Consumer   │
│  Pipeline   │    │  Pipeline   │    │  Artifact   │    │  SERVICE    │    │ Application │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
                                            ↓
                                      model.pkl
                                            ↓
                         ┌──────────────────────────────────────┐
                         │    Prediction API (FastAPI/GCF)     │
                         │  ┌─────────────────────────────────┐ │
                         │  │  Load model at startup          │ │
                         │  │  Validate input (Pydantic)      │ │
                         │  │  Run inference                   │ │
                         │  │  Return structured response     │ │
                         │  └─────────────────────────────────┘ │
                         └──────────────────────────────────────┘
```

### Monitoring Touchpoints

In a production system, monitoring would be added at:

| Stage | Monitoring | Tool Examples |
|-------|------------|---------------|
| Input Validation | Request count, validation errors | Prometheus, Cloud Monitoring |
| Model Inference | Latency, prediction distribution | OpenTelemetry, Datadog |
| Response | Error rates, status codes | Cloud Logging, Sentry |
| Infrastructure | CPU, memory, cold starts | Cloud Run Metrics |

---

## Setup Instructions

### Prerequisites

- Python 3.11+
- Google Cloud SDK (`gcloud`)
- Docker (for Cloud Run deployment)
- A GCP project with billing enabled

### Local Development

1. **Clone and setup environment:**

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

2. **Train the model (or use existing model.pkl):**

```bash
python train_model.py
```

3. **Run the FastAPI server locally:**

```bash
uvicorn main:app --reload --port 8000
```

4. **Access the API:**
   - Swagger UI: http://localhost:8000/docs
   - ReDoc: http://localhost:8000/redoc
   - Health check: http://localhost:8000/health

### Running Tests

```bash
pip install pytest httpx
pytest tests/test_api.py -v
```

---

## API Usage Examples

### Health Check

```bash
curl http://localhost:8000/health
```

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_version": "1.0.0"
}
```

### Make a Prediction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "sepal_length": 5.1,
    "sepal_width": 3.5,
    "petal_length": 1.4,
    "petal_width": 0.2
  }'
```

**Response:**
```json
{
  "prediction": "setosa",
  "prediction_id": 0,
  "confidence": 1.0,
  "model_version": "1.0.0"
}
```

### Validation Error Example

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"sepal_length": "five", "sepal_width": 3.5}'
```

**Response (422 Unprocessable Entity):**
```json
{
  "detail": [
    {
      "loc": ["body", "sepal_length"],
      "msg": "Input should be a valid number",
      "type": "float_parsing"
    }
  ]
}
```

### Get Model Info

```bash
curl http://localhost:8000/model/info
```

**Response:**
```json
{
  "model_type": "RandomForestClassifier",
  "model_version": "1.0.0",
  "feature_names": ["sepal_length", "sepal_width", "petal_length", "petal_width"],
  "target_names": ["setosa", "versicolor", "virginica"],
  "n_estimators": 100
}
```

---

## Deployment URLs

| Service | URL | Status |
|---------|-----|--------|
| Cloud Run | `https://iris-classifier-XXXXX.run.app` | Replace with your URL |
| Cloud Function | `https://us-central1-PROJECT.cloudfunctions.net/iris-predict` | Replace with your URL |

**Test Cloud Run:**
```bash
curl -X POST https://YOUR-CLOUD-RUN-URL/predict \
  -H "Content-Type: application/json" \
  -d '{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}'
```

**Test Cloud Function:**
```bash
curl -X POST https://YOUR-CLOUD-FUNCTION-URL \
  -H "Content-Type: application/json" \
  -d '{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}'
```

---

## Deployment Guide

### Cloud Run Deployment

1. **Enable required APIs:**

```bash
gcloud services enable artifactregistry.googleapis.com run.googleapis.com
```

2. **Create Artifact Registry repository:**

```bash
gcloud artifacts repositories create ml-models \
  --repository-format=docker \
  --location=us-central1 \
  --description="ML model container images"
```

3. **Configure Docker authentication:**

```bash
gcloud auth configure-docker us-central1-docker.pkg.dev
```

4. **Build and tag the image:**

```bash
# Set your project ID
export PROJECT_ID=$(gcloud config get-value project)

# Build the image
docker build -t iris-classifier:v1 .

# Tag for Artifact Registry
docker tag iris-classifier:v1 \
  us-central1-docker.pkg.dev/$PROJECT_ID/ml-models/iris-classifier:v1
```

5. **Push to Artifact Registry:**

```bash
docker push us-central1-docker.pkg.dev/$PROJECT_ID/ml-models/iris-classifier:v1
```

6. **Deploy to Cloud Run:**

```bash
gcloud run deploy iris-classifier \
  --image us-central1-docker.pkg.dev/$PROJECT_ID/ml-models/iris-classifier:v1 \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --port 8080 \
  --memory 512Mi \
  --min-instances 0 \
  --max-instances 10
```

7. **Verify deployment:**

```bash
# Get the service URL
CLOUD_RUN_URL=$(gcloud run services describe iris-classifier \
  --region us-central1 --format 'value(status.url)')

# Test the endpoint
curl -X POST $CLOUD_RUN_URL/predict \
  -H "Content-Type: application/json" \
  -d '{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}'
```

### Cloud Function Deployment

1. **Navigate to cloud_function directory:**

```bash
cd cloud_function
```

2. **Deploy the function:**

```bash
gcloud functions deploy iris-predict \
  --gen2 \
  --runtime python311 \
  --region us-central1 \
  --source . \
  --entry-point predict \
  --trigger-http \
  --allow-unauthenticated \
  --memory 512MB
```

3. **Get the function URL and test:**

```bash
FUNCTION_URL=$(gcloud functions describe iris-predict \
  --region us-central1 --format 'value(serviceConfig.uri)')

curl -X POST $FUNCTION_URL \
  -H "Content-Type: application/json" \
  -d '{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}'
```

---

## Comparative Analysis

### Cloud Run vs Cloud Functions: Architecture Comparison

| Aspect | Cloud Run (Container) | Cloud Functions (Serverless) |
|--------|----------------------|------------------------------|
| **Deployment Unit** | Docker image | Source code |
| **State Management** | Container state persists during instance lifetime | Global variables only (per-instance) |
| **Concurrency** | 80+ requests per instance | 1 request per instance (Gen1) or configurable (Gen2) |
| **Package Limit** | No limit (within container size) | 500MB |
| **Cold Start** | Longer (container startup) | Shorter (runtime initialization) |
| **Warm Latency** | Generally faster | Varies by complexity |
| **Reproducibility** | Full control (Dockerfile) | Managed runtime (less control) |

### Lifecycle Differences

**Cloud Run (Stateful Container):**
```
Request → Load Balancer → Container Instance
                              ↓
                         [Model in memory]
                              ↓
                         Process request
                              ↓
                         Return response
```
- Container persists between requests
- Model loaded once at startup
- State maintained during instance lifetime
- Predictable warm latency

**Cloud Functions (Stateless Function):**
```
Request → Cloud Functions Runtime → Function Instance
                                         ↓
                                    [Check global model cache]
                                         ↓
                                    Load if not cached
                                         ↓
                                    Process request
                                         ↓
                                    Return response
```
- Instance may be terminated between requests
- Model cached in global variable
- No guaranteed state persistence
- Variable latency based on instance state

### Artifact Loading Strategies

| Pattern | Cloud Run | Cloud Functions |
|---------|-----------|-----------------|
| **When** | At container startup (lifespan) | First request or global scope |
| **Caching** | Automatic (process memory) | Global variable (manual) |
| **Cold Start Impact** | Part of container init | Adds to first request latency |
| **Determinism** | High (same container) | Medium (instance recycling) |

**Cloud Run Loading:**
```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    model = joblib.load("model.pkl")  # Loaded once
    yield
```

**Cloud Functions Loading:**
```python
model = None  # Global cache

def load_model():
    global model
    if model is None:
        model = joblib.load("model.pkl")
    return model
```

### Latency Characteristics

#### Cold Start Components

| Phase | Cloud Run | Cloud Functions |
|-------|-----------|-----------------|
| Instance provisioning | 200-500ms | 100-300ms |
| Container/runtime start | 500-2000ms | 200-500ms |
| Dependency loading | 100-500ms | 100-300ms |
| Model loading | 500-5000ms | 500-5000ms |
| **Total Cold Start** | **1.3-8s** | **0.9-6s** |

#### Warm Instance Performance

| Metric | Cloud Run | Cloud Functions |
|--------|-----------|-----------------|
| P50 Latency | ~20-50ms | ~30-80ms |
| P95 Latency | ~50-100ms | ~80-200ms |
| P99 Latency | ~100-200ms | ~150-400ms |

*Note: Actual values depend on model size, network conditions, and load.*

### Cold Start Mitigation Strategies

| Strategy | Cloud Run | Cloud Functions |
|----------|-----------|-----------------|
| Minimum instances | `--min-instances 1` | Min instances (Gen2) |
| Smaller base image | `python:3.11-slim` | N/A (managed) |
| Fewer dependencies | Minimal requirements | Minimal requirements |
| Smaller model | Quantization | Quantization |
| Warm-up requests | Cloud Scheduler | Cloud Scheduler |

### Reproducibility Comparison

| Factor | Cloud Run | Cloud Functions |
|--------|-----------|-----------------|
| **Environment Control** | Full (Dockerfile) | Partial (managed runtime) |
| **Dependency Pinning** | requirements.txt | requirements.txt |
| **Python Version** | Exact (base image) | Major.minor only |
| **System Libraries** | Full control | Platform managed |
| **Build Reproducibility** | High | Medium |

**Recommendation:** For maximum reproducibility, Cloud Run with a fully pinned Dockerfile provides the most control. Cloud Functions are suitable for simpler use cases where rapid iteration is more important than environment control.

### When to Choose Each Pattern

**Choose Cloud Run when:**
- Latency SLAs require <100ms response time
- Model artifact is large (>100MB)
- Need custom system dependencies
- Full environment reproducibility is critical
- High concurrency is expected

**Choose Cloud Functions when:**
- Traffic is sporadic (hours between requests)
- Cost optimization is the priority
- Model is small with Python-only dependencies
- Rapid prototyping is needed
- Simpler operational model is preferred

---

## Model-API Interaction

### Data Flow

```
1. Client sends JSON request
   ↓
2. FastAPI/Cloud Function receives request
   ↓
3. Pydantic validates input schema
   ↓
4. Features extracted and converted to NumPy array
   ↓
5. Model.predict() called on cached model
   ↓
6. NumPy types converted to Python types
   ↓
7. Response model validated and serialized
   ↓
8. JSON response returned to client
```

### Request Validation (Pydantic)

The API uses Pydantic models to validate incoming requests:

```python
class PredictionRequest(BaseModel):
    sepal_length: float = Field(..., gt=0)  # Must be positive
    sepal_width: float = Field(..., gt=0)
    petal_length: float = Field(..., gt=0)
    petal_width: float = Field(..., gt=0)
```

**Validation Rules:**
- All fields are required (`...`)
- All values must be greater than 0 (`gt=0`)
- All values must be floats
- Invalid input returns HTTP 422 with detailed error messages

### Model Artifact Management

The model artifact (`model.pkl`) is:

1. **Trained** with fixed random seeds for reproducibility
2. **Serialized** using joblib for efficient storage
3. **Loaded once** at service startup (not per-request)
4. **Verified** to produce deterministic predictions

```python
# Training with reproducibility
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Verification after loading
loaded_model = joblib.load("model.pkl")
assert model.predict(X) == loaded_model.predict(X)  # Deterministic
```

### Response Serialization

NumPy types are not JSON-serializable by default. Pydantic handles this automatically:

```python
class PredictionResponse(BaseModel):
    prediction: str
    confidence: float  # Converts np.float64 → float automatically
    model_version: str
```

---

## Running the Benchmark

After deploying both services, compare latency:

```bash
python benchmark.py \
  --cloud-run-url https://YOUR-CLOUD-RUN-URL \
  --cloud-function-url https://YOUR-CLOUD-FUNCTION-URL \
  --num-requests 20 \
  --output benchmark_results.json
```

This will measure cold start and warm latency for both deployments and generate a comparison report.

---

## License

This project is part of the MLOps course curriculum.
