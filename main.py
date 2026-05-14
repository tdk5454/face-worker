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

# Redis bağlantısı - Hata vermemesi için kontrol ekledik
REDIS_URL = os.environ.get("UPSTASH_REDIS_URL")
if not REDIS_URL:
    print("HATA: UPSTASH_REDIS_URL bulunamadı!", flush=True)

redis = from_url(REDIS_URL) if REDIS_URL else None

# Model yükleme
try:
    with open("model.pkl", "rb") as f:
        md = pickle.load(f)
    clf   = md["classifier"]
    le    = md["label_encoder"]
    MNAME = md["model_name"]
    print("Model başarıyla yüklendi.", flush=True)
except Exception as e:
    print(f"Model yükleme hatası: {e}", flush=True)

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
            "face": True,
            "person": le.classes_[idx],
            "score": round(float(probs[idx]), 3)
        }
    except Exception as e:
        return {"face": False, "person": "Bilinmiyor", "error": str(e)}

async def worker_loop():
    print("Worker döngüsü aktif, kuyruk bekleniyor...", flush=True)
    while True:
        if not redis:
            await asyncio.sleep(5)
            continue
        try:
            item = await redis.brpop("face_queue", timeout=5)
            if item:
                _, raw = item
                job = json.loads(raw)
                job_id = job["job_id"]
                print(f"[{job_id}] işleniyor...", flush=True)
                
                img_bytes = base64.b64decode(job["image_b64"])
                result = predict(img_bytes)
                
                await redis.setex(f"result:{job_id}", 300, json.dumps(result))
                print(f"[{job_id}] tamamlandı.", flush=True)
        except Exception as e:
            print(f"Döngü hatası: {e}", flush=True)
            await asyncio.sleep(1)

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(worker_loop())

@app.get("/")
async def root():
    return {"status": "online", "redis_connected": redis is not None}

@app.get("/test")
async def test():
    # Railway'den linke tıkladığında bunu görmelisin
    return {"message": "API ve Worker çalışıyor!"}
