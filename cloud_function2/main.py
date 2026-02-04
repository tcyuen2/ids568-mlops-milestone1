import functions_framework
import joblib
import numpy as np

model = None

def load_model():
    global model
    if model is None:
        model = joblib.load("model.pkl")
    return model

@functions_framework.http
def predict(request):
    # Load model (cached after first call)
    clf = load_model()
    
    # Parse JSON request
    request_json = request.get_json(silent=True)
    
    if not request_json:
        return {"error": "No JSON provided"}, 400
    
    # Extract features
    try:
        features = np.array([[
            request_json["sepal_length"],
            request_json["sepal_width"],
            request_json["petal_length"],
            request_json["petal_width"]
        ]])
    except KeyError as e:
        return {"error": f"Missing field: {e}"}, 422
    
    # Get prediction
    prediction_idx = clf.predict(features)[0]
    confidence = clf.predict_proba(features)[0][prediction_idx]
    
    target_names = ["setosa", "versicolor", "virginica"]
    
    return {
        "prediction": target_names[prediction_idx],
        "confidence": float(confidence),
        "model_version": "1.0.0"
    }