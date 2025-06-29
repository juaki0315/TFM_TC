import joblib
import numpy as np

def load_model(path):
    return joblib.load(path)

def predict(model, features):
    features = np.array(features).reshape(1, -1)
    return model.predict(features)[0]
