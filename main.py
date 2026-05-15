import os, base64, json, pickle
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from deepface import DeepFace
from sklearn.preprocessing import normalize

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

with open("model.pkl", "rb") as f:
    md = pickle.load(f)

clf   = md["classifier"]
le    = md["label_encoder"]
MNAME = md["model_name"]

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        with open("/tmp/input.jpg", "wb") as f:
            f.write(contents)

        rep   = DeepFace.represent(
            img_path          = "/tmp/input.jpg",
            model_name        = MNAME,
            enforce_detection = False,
            detector_backend  = "opencv"
        )
        emb   = normalize([rep[0]["embedding"]])
        probs = clf.predict_proba(emb)[0]
        idx   = int(np.argmax(probs))

        return {
            "face":   True,
            "person": le.classes_[idx],
            "score":  round(float(probs[idx]), 3)
        }
    except Exception as e:
        return {"face": False, "person": "Başka biri", "score": 0.0, "error": str(e)}

@app.get("/health")
async def health():
    return {"status": "ok"}
