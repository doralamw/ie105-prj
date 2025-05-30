# main.py
from fastapi import FastAPI, File, UploadFile, HTTPException
from starlette.responses import FileResponse
import tempfile
import shutil, os, joblib, numpy as np
from feature_extractor import (
    extract_static,  # Chỉ dùng static!
    usecols_static
)
from sklearn.preprocessing import StandardScaler

app = FastAPI(title="APK Malware Detector")
rf     = joblib.load("models/rf_model.joblib")
# scaler = joblib.load("models/scaler.joblib")  # nếu có

@app.get("/", include_in_schema=False)
async def root():
    return FileResponse(os.path.join(os.path.dirname(__file__), "index.html"))

@app.post("/predict/")
async def predict(apk: UploadFile = File(...)):
    # 1. Tạo file tạm trong thư mục temp của hệ thống
    tempdir = tempfile.gettempdir()
    tmp_path = os.path.join(tempdir, apk.filename)
    with open(tmp_path, "wb") as buf:
        shutil.copyfileobj(apk.file, buf)

    try:
        # 2. Extract static features
        xs = extract_static(tmp_path)
        X = xs.reshape(1, -1)
        # Nếu bạn có scaler, dùng thêm bước này:
        # if scaler: X = scaler.transform(X)
        pred = rf.predict(X)[0]
        return {"filename": apk.filename, "prediction": int(pred)}
    except Exception as e:
        raise HTTPException(500, detail=str(e))
    finally:
        # 3. Xóa file tạm
        try:
            os.remove(tmp_path)
        except OSError:
            pass
