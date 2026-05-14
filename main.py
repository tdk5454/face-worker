import os, asyncio, base64, json, pickle
import numpy as np
from fastapi import FastAPI, File, UploadFile
from deepface import DeepFace
from sklearn.preprocessing import normalize

app = FastAPI()

# Modeli yükle
with open("model.pkl", "rb") as f:
    md = pickle.load(f)
clf, le, MNAME = md["classifier"], md["label_encoder"], md["model_name"]

@app.post("/predict")
async def predict_api(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        with open("/tmp/input.jpg", "wb") as f:
            f.write(contents)
        
        rep = DeepFace.represent(img_path="/tmp/input.jpg", model_name=MNAME, enforce_detection=False)
        emb = normalize([rep[0]["embedding"]])
        probs = clf.predict_proba(emb)[0]
        idx = int(np.argmax(probs))
        
        return {
            "face": True, 
            "person": le.classes_[idx], 
            "score": round(float(probs[idx]), 3)
        }
    except Exception as e:
        return {"face": False, "error": str(e)}

@app.get("/")
async def root():
    return {"status": "running"}
