"""
04_extract_features.py — реальные спектральные признаки
- ds004504: весь файл = EC (глаза закрыты)
- ds006036: вырезаем только интервалы open eyes (ГО) по events.tsv, игнорируем фотостимуляцию как отдельный конфounder
Выход: data/features_spectral.csv с относительными мощностями
"""
import mne, numpy as np, pandas as pd
from pathlib import Path
from tqdm import tqdm
import json

BANDS = {"delta": (0.5,4), "theta": (4,8), "alpha": (8,13), "beta": (13,30), "alpha_low": (8,10), "alpha_high": (10,13)}
SFREQ=500

def psd_features(raw, fmin=0.5, fmax=45):
    # Welch, 4s окна, 50% overlap
    psd = raw.compute_psd(method='welch', fmin=fmin, fmax=fmax, n_fft=int(4*SFREQ), n_overlap=int(2*SFREQ), verbose=False)
    psds, freqs = psd.get_data(), psd.freqs
    # psds: (n_channels, n_freqs)
    total = psds[:, (freqs>=0.5)&(freqs<=30)].sum(axis=1, keepdims=True) + 1e-12
    feats = {}
    for name,(lo,hi) in BANDS.items():
        mask=(freqs>=lo)&(freqs<=hi)
        rel = psds[:, mask].sum(axis=1) / total.squeeze()  # per channel
        feats[f"rel_{name}_mean"] = rel.mean()
        feats[f"rel_{name}_occipital"] = rel[[7,8,13,14]].mean()  # P3,P4,T5,T6 / O1,O2 approx
        feats[f"rel_{name}_frontal"] = rel[[0,1,2,3,10,11]].mean()
    # дополнительно
    feats["alpha_theta_ratio"] = feats["rel_alpha_mean"] / (feats["rel_theta_mean"]+1e-9)
    feats["theta_beta_ratio"] = feats["rel_theta_mean"] / (feats["rel_beta_mean"]+1e-9)
    # spectral entropy
    p_norm = psds.mean(axis=0)
    p_norm = p_norm[(freqs>=0.5)&(freqs<=30)]
    p_norm/=p_norm.sum()
    feats["spectral_entropy"] = -(p_norm*np.log(p_norm+1e-12)).sum()
    # peak alpha freq
    alpha_mask=(freqs>=8)&(freqs<=13)
    feats["peak_alpha"] = freqs[alpha_mask][np.argmax(psds.mean(axis=0)[alpha_mask])]
    return feats

def get_eo_intervals(sub):
    # возвращает список (onset, duration) для open eyes
    ev_path=Path(f"data/raw/ds006036/{sub}/eeg/{sub}_task-photomark_events.tsv")
    df=pd.read_csv(ev_path, sep="\t")
    if "value" not in df.columns:
        return []
    opens=df[df["value"].str.contains("open eyes", case=False, na=False)][["onset"]].values.flatten()
    closes=df[df["value"].str.contains("closed eyes", case=False, na=False)][["onset"]].values.flatten()
    intervals=[]
    for ot in opens:
        nxt=closes[closes>ot]
        if len(nxt)>0:
            intervals.append((ot, nxt[0]-ot))
    return intervals

def process_one(sub, condition):
    if condition=="EC":
        f=Path(f"data/raw/ds004504/{sub}/eeg/{sub}_task-eyesclosed_eeg.set")
        raw=mne.io.read_raw_eeglab(f, preload=True, verbose=False)
        # фильтрация как в описании датасета: 0.5-45 уже отфильтровано, но делаем для консистентности 0.5-45
        raw.filter(0.5,45, verbose=False)
        feats=psd_features(raw)
        feats.update({"participant_id":sub, "condition":condition, "duration": raw.times[-1]})
        return feats
    else: # EO from ds006036
        f=Path(f"data/raw/ds006036/{sub}/eeg/{sub}_task-photomark_eeg.set")
        raw=mne.io.read_raw_eeglab(f, preload=True, verbose=False)
        raw.filter(0.5,45, verbose=False)
        intervals=get_eo_intervals(sub)
        if len(intervals)==0:
            return None
        # вырезаем и конкатенируем только EO интервалы
        raws=[]
        for onset, dur in intervals:
            if dur < 1.0: continue
            raws.append(raw.copy().crop(tmin=onset, tmax=onset+dur, include_tmax=False))
        if len(raws)==0:
            return None
        raw_eo=mne.concatenate_raws(raws, verbose=False)
        feats=psd_features(raw_eo)
        feats.update({"participant_id":sub, "condition":condition, "duration": raw_eo.times[-1], "n_intervals": len(intervals)})
        return feats

# Основной цикл
subs = sorted([p.name for p in Path("data/raw/ds004504").glob("sub-*")])
print(f"Участников: {len(subs)}")
rows=[]
for sub in tqdm(subs):
    for cond in ["EC","EO"]:
        try:
            r=process_one(sub, cond)
            if r: rows.append(r)
            else: print(f"skip {sub} {cond} — нет интервалов")
        except Exception as e:
            print(f"ERR {sub} {cond}: {e}")

df=pd.DataFrame(rows)
# мерджим диагноз
p=pd.read_csv("data/raw/ds004504/participants.tsv", sep="\t")
grp=p[["participant_id","Group","Age","MMSE","Gender"]].rename(columns={"Group":"diagnosis"})
grp["diagnosis"]=grp["diagnosis"].map({"A":"AD","F":"FTD","C":"HC"})
df=df.merge(grp, on="participant_id", how="left")
df.to_csv("data/features_spectral.csv", index=False)
print(df.head())
print(df.groupby(["condition","diagnosis"]).size())
print("Сохранено data/features_spectral.csv, строк:", len(df))
