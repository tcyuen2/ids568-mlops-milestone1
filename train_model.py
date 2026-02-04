"""
Train and save a scikit-learn Iris classifier model.
This script creates the model.pkl artifact for the prediction service.
"""
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load the Iris dataset
iris = load_iris()
X = iris.data 
y =iris.target

# Train a RandomForest classifier with fixed random_state=42 for reproducability
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)


# save the model
joblib.dump(model, "model.pkl")
print("Model saved to model.pkl")

