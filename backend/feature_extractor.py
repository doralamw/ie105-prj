import os
import numpy as np
import re
import pandas as pd
from sklearn.impute import SimpleImputer
from androguard.misc import AnalyzeAPK
from sklearn.feature_selection import VarianceThreshold, SelectKBest, chi2


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
df_static = pd.read_csv(
    'feature_vectors_static.csv',
    usecols=usecols,
    index_col=0,
)
df_static = df_static.apply(pd.to_numeric, errors='coerce').astype('float32')
df_sys_binder = pd.read_csv('feature_vectors_syscallsbinders_frequency_5_Cat.csv')
y = df_sys_binder['Class'].values

X = df_static.values

#Xử lý NaN
imp = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0)
X_imputed = imp.fit_transform(X)

#VarianceThreshold 
vt_selector = VarianceThreshold(threshold=0.0001)
X_vt = vt_selector.fit_transform(X_imputed)
names_vt = [f for f, keep in zip(usecols, vt_selector.get_support()) if keep]

#SelectKBest
kbest_selector = SelectKBest(chi2, k=2000)
X_kbest = kbest_selector.fit_transform(X_vt, y)
names_final = [f for f, keep in zip(names_vt, kbest_selector.get_support()) if keep]


idx_stat  = {c:i for i, c in enumerate(names_final)}

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
