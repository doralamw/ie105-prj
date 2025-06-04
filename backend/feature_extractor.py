import os
import numpy as np
import re
import pandas as pd
from sklearn.impute import SimpleImputer
from androguard.misc import AnalyzeAPK
from sklearn.feature_selection import VarianceThreshold, SelectKBest, chi2


with open("final_columns.txt") as f:
    final_columns = [line.strip() for line in f if line.strip()]

idx_stat  = {c:i for i, c in enumerate(final_columns)}
print("read successfully")
def extract_from_mobsf_report(report_json):
    x = np.zeros(len(final_columns), dtype=float)

    # Permissions
    for perm in report_json.get("permissions", []):
        key = f"permission_{perm}"
        if key in idx_stat:
            x[idx_stat[key]] = 1

    # API Calls (MobSF trả về là list tên api hoặc dict tuỳ phiên bản)
    api_calls = report_json.get("api_calls", [])
    # Nếu là list các dict thì lấy tên api từ từng dict
    if api_calls and isinstance(api_calls[0], dict):
        for api in api_calls:
            name = api.get("name") or api.get("api")
            if name:
                key = f"api_call_{name}"
                if key in idx_stat:
                    x[idx_stat[key]] += 1
    else:
        # Trường hợp là list string
        for call in api_calls:
            key = f"api_call_{call}"
            if key in idx_stat:
                x[idx_stat[key]] += 1

    # Activities
    for act in report_json.get("activities", []):
        key = f"activity_{act}"
        if key in idx_stat:
            x[idx_stat[key]] += 1

    # Services
    for svc in report_json.get("services", []):
        key = f"service_{svc}"
        if key in idx_stat:
            x[idx_stat[key]] += 1

    # Providers
    for prov in report_json.get("providers", []):
        key = f"provider_{prov}"
        if key in idx_stat:
            x[idx_stat[key]] += 1

    # Receivers
    for rcv in report_json.get("receivers", []):
        key = f"receiver_{rcv}"
        if key in idx_stat:
            x[idx_stat[key]] += 1

    # Intents (nếu có, hoặc bạn bổ sung theo nhu cầu)
    for intent in report_json.get("intents", []):
        key = f"intent_{intent}"
        if key in idx_stat:
            x[idx_stat[key]] += 1

    # Package
    pkg = report_json.get("package")
    if pkg:
        key = f"package_{pkg}"
        if key in idx_stat:
            x[idx_stat[key]] = 1

    # Có thể bổ sung các trường khác tùy final_columns.txt của bạn
    print("Số feature ≠ 0:", (x != 0).sum())
    print(x.nonzero())
    return x