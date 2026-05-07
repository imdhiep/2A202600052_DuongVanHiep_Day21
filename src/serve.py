from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from google.cloud import storage
import joblib
import os
import numpy as np

app = FastAPI()

# Thong tin cau hinh tu bien moi truong
GCS_BUCKET = os.environ.get("GCS_BUCKET", "vinuni-mlops-lab-2026")
MODEL_KEY = "models/latest/model.pkl"
SCALER_KEY = "models/latest/scaler.pkl"

MODEL_PATH = "model.pkl"
SCALER_PATH = "scaler.pkl"

def download_from_gcs():
    """
    Tai model va scaler tu Google Cloud Storage ve thu muc local.
    """
    try:
        client = storage.Client()
        bucket = client.bucket(GCS_BUCKET)
        
        # Tai file model
        blob_model = bucket.blob(MODEL_KEY)
        blob_model.download_to_filename(MODEL_PATH)
        
        # Tai file scaler
        blob_scaler = bucket.blob(SCALER_KEY)
        blob_scaler.download_to_filename(SCALER_PATH)
        
        print(f"Thanh cong: Da tai model va scaler tu bucket {GCS_BUCKET}")
    except Exception as e:
        print(f"Canh bao: Khong the tai file tu GCS (co the dang chay local): {e}")

# Tu dong tai file khi khoi dong neu chua co
if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
    download_from_gcs()

# Load model va scaler vao bo nho
try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
except Exception as e:
    print(f"Loi: Khong the load model/scaler: {e}")
    model = None
    scaler = None

class PredictRequest(BaseModel):
    features: list[float]

@app.get("/health")
def health():
    """Endpoint kiem tra trang thai server."""
    return {"status": "ok"}

@app.post("/predict")
def predict(req: PredictRequest):
    """Endpoint du doan chat luong ruou."""
    if model is None or scaler is None:
        raise HTTPException(status_code=500, detail="Model hoac Scaler chua duoc tai len server.")
    
    if len(req.features) != 12:
        raise HTTPException(
            status_code=400, 
            detail=f"Yeu cau 12 dac trung, nhung nhan duoc {len(req.features)}"
        )

    try:
        # 1. Chuyen sang array va chuan hoa
        data = np.array(req.features).reshape(1, -1)
        data_scaled = scaler.transform(data)
        
        # 2. Du doan
        prediction = int(model.predict(data_scaled)[0])
        
        # 3. Map nhan ket qua
        labels = {0: "thap", 1: "trung_binh", 2: "cao"}
        
        return {
            "prediction": prediction,
            "label": labels.get(prediction, "unknown")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Loi trong qua trinh du doan: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
