# Milestone 1: Web & Serverless Model Serving

## Project Outline

This project takes a trained machine learning model (Iris flower classifier) and makes it available as a web API, and is deployed in two ways
1. Cloud Run - using Docker
2. Cloud Functions - using serverless deployment

## Project Structure
```
milestone1/
├── main.py              # FastAPI app with /predict and /health endpoints
├── model.pkl            # Trained model file
├── train_model.py       # Script to train the model
├── requirements.txt     # Python packages needed
├── Dockerfile           # Instructions to build the container
└── cloud_function/      # Code for the serverless function
    ├── main.py
    ├── requirements.txt
    └── model.pkl
```

## Setup Instructions

### Run Locally
```bash
# Install packages
pip install -r requirements.txt

# Train the model (creates model.pkl)
python train_model.py

# Start the API
uvicorn main:app --reload --port 8000

# Open in browser
# http://localhost:8000/docs
```

# URLs

Cloud Run:  https://iris-classifier-324763324859.us-central1.run.app |
Cloud Function:  https://us-central1-project-06e66f92-e1ae-4362-a27.cloudfunctions.net/iris-predict |

# Testing the Cloud Run

Health check (browser):
```
https://iris-classifier-324763324859.us-central1.run.app/health
```

Swagger UI :
```
https://iris-classifier-324763324859.us-central1.run.app/docs
```

# Testing the Cloud Function

https://us-central1-project-06e66f92-e1ae-4362-a27.cloudfunctions.net/iris-predict


## API Usage

### Request

Send a POST to `/predict` with:
```json
{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}
```

### Response
```json
{
  "prediction": "setosa",
  "confidence": 1.0,
  "model_version": "1.0.0"
}
```

## ML Lifecycle
```
Training Data -> Train Model-> model.pkl -> API -> Users/Apps
```

This API sis in the serving stage. It loads the trained model and exposes it via HTTP endpoints so other applications can get predictions.

## Model-API Interaction

1. Model loads once when the app starts (not every request)
2. User sends flower measurements as JSON
3. Pydantic validates the input
4. Model makes a prediction
5. Response is sent back as JSON

## Cloud Run vs Cloud Functions Comparison

Cloud Run and Cloud Functions are pretty different in how they work. With Cloud Run, you deploy a Docker container, which gives you more control since you write the Dockerfile manually. Cloud Functions is simpler, where you just upload your Python code and Google does the rest.

For cold starts, Cloud Functions is actually faster since it only needs to start the Python runtime. But once things are warmed up, Cloud Run handles requests faster. Cloud Run is better for apps which need low latency, while Cloud Functions is good for prototyping.

### Cold Start Behavior

A cold start happens when the service has been sitting idle and needs to wake up for the first request. For Cloud Run, this takes about 1-3 seconds because the whole docker container needs to start up. Cloud Functions are faster at around 1 second because it's starting the Python runtime. Once everything is ready, both respond quickly in about 50-100ms.

### When to Use Each

I'd use Cloud Run if I needed really fast response times or had a bigger model, since there is more control over the environment with the Dockerfile. Cloud Functions makes more sense when traffic is unpredictable or for prototyping something quickly.