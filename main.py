import os
import asyncio
import base64
import json
import pickle
import numpy as np
from fastapi import FastAPI
from deepface import DeepFace
from sklearn.preprocessing import normalize
from redis.asyncio import from_url

app = FastAPI()

# Redis bağlantısı
redis_url = os.environ.get("UPSTASH_REDIS_URL")
redis = from_url(redis_url)

# Model yükleme
with open("model.pkl", "rb") as f:
    md = pickle.load(f)

clf   = md["classifier"]
le    = md["label_encoder"]
MNAME = md["model_name"]

TMP = "/tmp/face_input.jpg"

def predict(img_bytes: bytes) -> dict:
    with open(TMP, "wb") as f:
        f.write(img_bytes)
    try:
        rep = DeepFace.represent(
            img_path=TMP,
            model_name=MNAME,
            enforce_detection=False,
            detector_backend="opencv"
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
        return {"face": False, "person": "Bilinmiyor", "score": 0.0, "error": str(e)}

async def worker_loop():
    print("Worker döngüsü başladı, kuyruk dinleniyor...", flush=True)
    while True:
        try:
            item = await redis.brpop("face_queue", timeout=10)
            if not item:
                continue
            
            _, raw = item
            job = json.loads(raw)
            job_id = job["job_id"]
            img_bytes = base64.b64decode(job["image_b64"])

            print(f"[{job_id}] İşlem başlıyor...", flush=True)
            result = predict(img_bytes)
            
            # Sonucu 5 dakika (300sn) sakla
            await redis.setex(f"result:{job_id}", 300, json.dumps(result))
            print(f"[{job_id}] Bitti -> {result['person']}", flush=True)
            
        except Exception as e:
            print(f"Worker Hatası: {str(e)}", flush=True)
            await asyncio.sleep(2)

@app.on_event("startup")
async def startup_event():
    print("Railway sistemi kalkıyor...", flush=True)
    asyncio.create_task(worker_loop())

@app.get("/")
async def root():
    return {"status": "active", "info": "Face Worker is running"}
