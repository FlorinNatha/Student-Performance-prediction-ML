from fastapi import FastAPI
import pickle
import numpy as np

app = FastAPI()

model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

@app.get("/")
def home():
    return {"message": "API running"}

@app.post("/predict")
def predict(data: dict):
    features = np.array(list(data.values())).reshape(1, -1)
    features = scaler.transform(features)

    prediction = model.predict(features)
    return {"prediction": int(prediction[0])}