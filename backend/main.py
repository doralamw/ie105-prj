# main.py
from fastapi import FastAPI, File, UploadFile, HTTPException
from starlette.responses import FileResponse
import tempfile
import shutil, os, joblib, numpy as np
from feature_extractor import (
    extract_static, run_dynamic_sandbox,
    parse_syscalls, parse_binders,
    usecols_static, usecols_sys, usecols_binder
)
from sklearn.preprocessing import StandardScaler

app = FastAPI(title="APK Malware Detector")
rf     = joblib.load("models/rf_model.joblib")
#scaler = joblib.load("models/scaler.joblib")  # nếu có
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
        # 2. Extract static
        xs = extract_static(tmp_path)
        # 3. Extract dynamic
        logdir = run_dynamic_sandbox(tmp_path, timeout=30)
        x_sys    = parse_syscalls(logdir)
        x_bind   = parse_binders(logdir)
        shutil.rmtree(logdir)
        # 4. Kết hợp, scale, predict
        X = np.hstack([xs, x_sys, x_bind]).reshape(1, -1)
       # if scaler: X = scaler.transform(X)
        pred = rf.predict(X)[0]
        return {"filename": apk.filename, "prediction": int(pred)}
    except Exception as e:
        raise HTTPException(500, detail=str(e))
    finally:
        # 5. Xóa file tạm
        try:
            os.remove(tmp_path)
        except OSError:
            pass