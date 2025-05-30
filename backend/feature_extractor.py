import os
import numpy as np
import re
import pandas as pd
from androguard.misc import AnalyzeAPK

# Đọc tên các cột đặc trưng tĩnh
cols = pd.read_csv('feature_vectors_static.csv', nrows=0).columns.tolist()
patterns = {
    'F1_permissions':    r'permission',
    'F2_api_calls':      r'api_call|call_',
    'F3_intents':        r'android\.intent\.action',
    'F4_components':     r'activity|provider',
    'F5_packages':       r'package',
    'F6_services':       r'service(_|$)',
    'F7_receivers':      r'receiver'
}

usecols = []
for pat in patterns.values():
    usecols += [c for c in cols if re.search(pat, c, flags=re.IGNORECASE)]

seen = set()
usecols_static = [c for c in usecols if not (c in seen or seen.add(c))]  # list tên feature static (F1–F7)
idx_stat  = {c:i for i, c in enumerate(usecols_static)}

def extract_static(apk_path):
    # AnalyzeAPK trả về tuple (APK, DalvikVMFormat, Analysis)
    a, d, dx = AnalyzeAPK(apk_path)
    x = np.zeros(len(usecols_static), dtype=float)

    # F1: permissions
    for perm in a.get_permissions():
        key = f"permission_{perm}"
        if key in idx_stat:
            x[idx_stat[key]] = 1

    # F3: intent filters
    for comp in ("activity", "service", "receiver"):
        try:
            intent_dict = a.get_intent_filters(comp)
        except TypeError:
            continue

        for intentf in intent_dict.values():
            for action in intentf.get_actions():
                key = f"intent_{action}"
                if key in idx_stat:
                    x[idx_stat[key]] += 1.0

    # F4–F7: components
    for act in a.get_activities():
        key = f"activity_{act}"
        if key in idx_stat: x[idx_stat[key]] += 1
    for prov in a.get_providers():
        key = f"provider_{prov}"
        if key in idx_stat: x[idx_stat[key]] += 1
    for svc in a.get_services():
        key = f"service_{svc}"
        if key in idx_stat: x[idx_stat[key]] += 1
    for rcv in a.get_receivers():
        key = f"receiver_{rcv}"
        if key in idx_stat: x[idx_stat[key]] += 1

    # F5: package
    pkg = a.get_package()
    key = f"package_{pkg}"
    if key in idx_stat: x[idx_stat[key]] = 1

    # F2: API calls from DEX
    for dex in d:
        for method in dex.get_methods():
            cls  = method.get_class_name().strip(';')
            name = method.get_name()
            key  = f"api_call_{cls}_{name}"
            if key in idx_stat:
                x[idx_stat[key]] += 1.0

    return x

# ========================
# XÓA toàn bộ code bên dưới vì liên quan dynamic:
# - run_dynamic_sandbox
# - parse_syscalls
# - parse_binders
# ========================
