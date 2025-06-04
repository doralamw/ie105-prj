import traceback
from fastapi import FastAPI, File, UploadFile, HTTPException
from starlette.responses import FileResponse
import uuid
import shutil, os, joblib, numpy as np
from fastapi.middleware.cors import CORSMiddleware
import requests

MOBSF_URL = "http://localhost:8000/api/v1"
MOBSF_API_KEY = "97cd0c93af457f925b669a370d182b12adb98519d06c001fe730992d4364b4f7"

from feature_extractor import extract_from_mobsf_report

with open("final_columns.txt", encoding="utf-8") as f:
    final_columns = [line.strip() for line in f if line.strip()]

rf = joblib.load("models/rf_model.joblib")
print("=== Đã load xong model rf ===")

app = FastAPI(title="APK Malware Detector")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", include_in_schema=False)
async def root():
    return FileResponse(os.path.join(os.path.dirname(__file__), "index.html"))

@app.post("/predict/")
async def predict(apk: UploadFile = File(...)):
    # CHỈ ĐỊNH THƯ MỤC TẠM NGẮN, ĐƯỜNG DẪN ĐƠN GIẢN
    tempdir = "E:\\Temp"
    os.makedirs(tempdir, exist_ok=True)
    # SỬ DỤNG CHỈ UUID, KHÔNG LẤY TÊN GỐC NGƯỜI DÙNG GỬI LÊN
    tmp_filename = f"{uuid.uuid4().hex}.apk"
    tmp_path = os.path.join(tempdir, tmp_filename)

    # Xóa ký tự xuống dòng và các ký tự lạ (phòng lỗi)
    tmp_path = tmp_path.replace('\n', '').replace('\r', '').replace(' ', '')

    print(f"Tạo file tạm: {repr(tmp_path)}")  # dùng repr để kiểm tra ký tự ẩn

    with open(tmp_path, "wb") as buf:
        shutil.copyfileobj(apk.file, buf)

    try:
        with open(tmp_path, "rb") as f:
            files = {'file': (os.path.basename(tmp_path), f)}
            headers = {"Authorization": MOBSF_API_KEY}
            upload_resp = requests.post(f"{MOBSF_URL}/upload", headers=headers, files=files)
            upload_resp.raise_for_status()
            scan_hash = upload_resp.json()["hash"]
            print("[*] Uploaded, scan hash:", scan_hash)

        scan_resp = requests.post(f"{MOBSF_URL}/scan", headers=headers, data={"hash": scan_hash})
        scan_resp.raise_for_status()
        print("[*] Scan triggered:", scan_resp.status_code)

        report_resp = requests.post(f"{MOBSF_URL}/report_json", headers=headers, data={"hash": scan_hash})
        report_resp.raise_for_status()
        report = report_resp.json()
        print("[*] Report downloaded")

        xs = extract_from_mobsf_report(report)
        print("Số feature ≠ 0:", (xs != 0).sum())
        print("Feature vector (first 10):", xs[:10])
        X = xs.reshape(1, -1)

        pred = rf.predict(X)[0]
        print("Prediction trả về từ model:", pred)

        return {"filename": apk.filename, "prediction": int(pred)}
    except Exception as e:
        tb = traceback.format_exc()
        print("=== LỖI PHÂN TÍCH FILE ===")
        print(tb)
        raise HTTPException(500, detail=f"{str(e)}\nTraceback:\n{tb}")
    finally:
        try:
            if os.path.exists(tmp_path):
                print(f"File tạm: {tmp_path}, size: {os.path.getsize(tmp_path)} bytes")
                os.remove(tmp_path)
        except Exception as e:
            print("Lỗi khi xóa file tạm:", str(e))
