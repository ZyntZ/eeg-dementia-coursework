import json, numpy as np, mne
import pandas as pd
from pathlib import Path
from scipy.signal import resample
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore")

# Параметры как в LEAD
SAMPLE_RATE_LIST = [200, 100, 50]
SEG_LEN = 400
OVERLAP = 200
STEP = SEG_LEN - OVERLAP

COMMON_CHANNELS = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Fz', 'Cz', 'Pz']

def resample_time_series(data, original_fs, target_fs):
    T, C = data.shape
    new_len = int(T * target_fs / original_fs)
    out = np.zeros((new_len, C), dtype=np.float32)
    for i in range(C):
        out[:, i] = resample(data[:, i], new_len)
    return out

def build_dataset(ds_name, raw_root, out_root):
    # raw_root: data/raw/ds004504 или ds006036
    # ожидаем структуру raw_root/sub-*/eeg/*.set
    # метки из participants.tsv
    p = pd.read_csv(Path(raw_root)/"participants.tsv", sep="\t")
    p.columns = p.columns.str.replace("﻿","").str.strip()
    label_map = {"C":0, "A":1, "F":2}  # Control AD FTD
    sub_info = {}
    for _, row in p.iterrows():
        pid = int(row["participant_id"].split("-")[1])
        sub_info[row["participant_id"]] = (label_map[row["Group"]], pid)

    # Считаем сегменты PASS1
    subject_counts = defaultdict(lambda: defaultdict(int))
    N_total = 0
    deriv = Path(raw_root)
    for sub in sorted([d.name for d in deriv.glob("sub-*")]):
        sub_path = deriv / sub / "eeg"
        if not sub_path.exists():
            continue
        for f in sub_path.glob("*.set"):
            raw = mne.io.read_raw_eeglab(f, preload=False, verbose=False)
            T_raw = raw.n_times
            orig_fs = raw.info["sfreq"]
            for fs in SAMPLE_RATE_LIST:
                T_res = int(T_raw * fs / orig_fs)
                n_seg = 0 if T_res < SEG_LEN else 1 + (T_res - SEG_LEN)//STEP
                subject_counts[sub][fs] += n_seg
                N_total += n_seg

    print(f"{ds_name}: N_total={N_total}, SEG_LEN={SEG_LEN}, STEP={STEP}")

    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    X_path = out_root/"X.dat"
    y_path = out_root/"y.dat"
    meta_path = out_root/"meta.json"

    X_mm = np.memmap(X_path, dtype="float32", mode="w+", shape=(N_total, SEG_LEN, len(COMMON_CHANNELS)))
    y_mm = np.memmap(y_path, dtype="float32", mode="w+", shape=(N_total, 3))

    cur = 0
    total_hours = 0
    for sub in sorted([d.name for d in deriv.glob("sub-*")]):
        if sub not in sub_info:
            continue
        label, pid = sub_info[sub]
        sub_path = deriv / sub / "eeg"
        for f in sub_path.glob("*.set"):
            raw = mne.io.read_raw_eeglab(f, preload=True, verbose=False)
            # порядок каналов как в COMMON_CHANNELS
            raw.reorder_channels(COMMON_CHANNELS)
            # фильтры как в LEAD: notch 50Hz + bandpass 0.5-45
            raw.notch_filter(50, verbose=False)
            raw.filter(0.5,45, verbose=False)
            data_raw = raw.get_data().T.astype(np.float32)
            total_hours += data_raw.shape[0] / raw.info["sfreq"] / 3600
            for fs in SAMPLE_RATE_LIST:
                data = resample_time_series(data_raw, raw.info["sfreq"], fs)
                starts = np.arange(0, data.shape[0]-SEG_LEN+1, STEP, dtype=int)
                for s in starts:
                    X_mm[cur] = data[s:s+SEG_LEN]
                    y_mm[cur,0]=float(label)
                    y_mm[cur,1]=float(pid)
                    y_mm[cur,2]=float(fs)
                    cur+=1

    print(f"DONE {ds_name} cur={cur}/{N_total} hours={total_hours:.2f}")
    del X_mm, y_mm
    meta={
        "N": int(N_total),
        "T": SEG_LEN,
        "C": len(COMMON_CHANNELS),
        "SAMPLE_RATE_LIST": SAMPLE_RATE_LIST,
        "MONTAGE": "standard_1020",
        "CHANNELS": COMMON_CHANNELS,
        "LABELS": {"disease_id": {"0":"Control","1":"AD","2":"FTD"}, "subject_id":"1..88", "sampling_frequency":"int"}
    }
    with open(meta_path,"w") as f:
        json.dump(meta,f,indent=2)
    print(f"Saved {meta_path}")
    return N_total

if __name__=="__main__":
    import sys
    # Пример: python src/08_lead_preprocessing.py data/raw/ds004504 Processed/L400/ADFTD-RS
    if len(sys.argv)==3:
        build_dataset(sys.argv[1], sys.argv[1], sys.argv[2])
    else:
        print("Usage: python src/08_lead_preprocessing.py <raw_root> <out_root>")
