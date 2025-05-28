import os, glob, subprocess
import numpy as np
import re
import pandas as pd
from androguard.misc import AnalyzeAPK
from collections import Counter
from pathlib import Path
import tempfile
import time


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


# 1) Các danh sách tên cột bạn đã train
usecols_static = [c for c in usecols if not (c in seen or seen.add(c))]    # list tên feature static (F1–F7)
usecols_sys    = pd.read_csv('feature_vectors_syscalls_frequency_5_Cat.csv', nrows =0).columns.tolist()    # list tên syscall (139)
usecols_binder= pd.read_csv('feature_vectors_syscallsbinders_frequency_5_Cat.csv', nrows=0).columns.tolist()    # list tên binder-call (470)

# Chuyển thành map name→index
idx_stat  = {c:i for i,c in enumerate(usecols_static)}
idx_sys   = {c:i for i,c in enumerate(usecols_sys)}
idx_bind  = {c:i for i,c in enumerate(usecols_binder)}

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
            # nếu API khác, skip
            continue

        for intentf in intent_dict.values():
            # mỗi intentf là một IntentFilter object
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
            # Ví dụ key format của bạn là 'api_call_<class>_<method>'
            cls  = method.get_class_name().strip(';')  # có thể cần strip dấu
            name = method.get_name()
            key  = f"api_call_{cls}_{name}"
            if key in idx_stat:
                x[idx_stat[key]] += 1.0

    return x
def run_dynamic_sandbox(apk_path: str, timeout: int = 30) -> str:
    """
    Spin up a fresh apk-sandbox container to:
      1. Mount the APK read-only
      2. Mount a host directory for logs
      3. Wait for emulator to come up
      4. Install the APK
      5. Run strace on the app process for `timeout` seconds
      6. Kill strace and exit container
    Returns the host path where strace logs were written.
    """

    # 1) Prepare a unique log directory on the host
    job = Path(apk_path).stem
    host_logdir = os.path.join(tempfile.gettempdir(), "logs", job)
    os.makedirs(host_logdir, exist_ok=True)

    # 2) Convert Windows-style path to POSIX style for -v mount
    host_apk = Path(apk_path).as_posix()

    # 3) Build the docker run command
    cmd = [
    "docker", "exec", "project-final-sandbox-1",  # hoặc tên container sandbox
    "bash", "-c",
    (
        "adb wait-for-device && "
        "adb install -r /data/app.apk && "
        "PID=$(adb shell pidof com.example.yourapp) && "
        "strace -ff -tt -o /data/logs/strace.log -p $PID & "
        f"sleep {timeout} && pkill strace"
    )
]

    # 4) Execute and raise on failure
    subprocess.run(cmd, check=True)

    # 5) Return the host log directory
    return host_logdir
def parse_syscalls(logdir):
    cnt = Counter()
    for f in glob.glob(f"{logdir}/strace.log.*"):
        for line in open(f):
            name = line.split('(')[0]
            if name in idx_sys: cnt[name] += 1
    return np.array([cnt[c] for c in usecols_sys], dtype=float)

def extract_binder_name(line):
    # Ví dụ parse "[   42.123456] binder: transact ... code:..."
    parts = line.split()
    # tách phần tên binder ở chỗ nhất định
    # tự hiệu chỉnh cho đúng log format
    if "binder" in parts:
        return parts[parts.index("binder:")+1]
    return None

def parse_binders(logdir):
    cnt = Counter()
    for f in glob.glob(f"{logdir}/binder.log*"):
        for line in open(f):
            name = extract_binder_name(line)
            if name and name in idx_bind: cnt[name] += 1
        return np.zeros(len(usecols_binder), dtype=float)
