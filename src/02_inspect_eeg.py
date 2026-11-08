import mne
import pandas as pd
from pathlib import Path
import numpy as np

# Проверим 1 файл из каждого датасета
for ds, task in [("ds004504","eyesclosed"),("ds006036","photomark")]:
    for sub in ["sub-001","sub-011","sub-037"]:
        f = Path(f"data/raw/{ds}/{sub}/eeg/{sub}_task-{task}_eeg.set")
        print(f"\n=== {ds} {sub} ===")
        if not f.exists():
            print("нет файла", f)
            continue
        raw = mne.io.read_raw_eeglab(f, preload=False, verbose=False)
        print(f"info: {raw.info['sfreq']} Hz, {len(raw.ch_names)} ch, {raw.n_times/raw.info['sfreq']:.1f} sec")
        print("ch_names:", raw.ch_names)
        print("ch_types:", set(raw.get_channel_types()))
        # события из .set vs events.tsv
        ev_tsv = Path(f"data/raw/{ds}/{sub}/eeg/{sub}_task-{task}_events.tsv")
        if ev_tsv.exists():
            df=pd.read_csv(ev_tsv, sep="\t")
            print(f"events.tsv: {len(df)} строк, колонки {df.columns.tolist()}")
            print(df.head(3).to_string())
        # аннотации внутри .set
        try:
            ann = raw.annotations
            print(f"annotations in .set: {len(ann)}")
            if len(ann)>0:
                print(ann[:3])
        except Exception as e:
            print(e)
        break

# Проверим все длительности быстро без загрузки данных
print("\n--- Сводка длительностей ---")
import json, glob
for ds, task in [("ds004504","eyesclosed"),("ds006036","photomark")]:
    durations=[]
    for sub in sorted([p.name for p in Path(f"data/raw/{ds}").glob("sub-*")]):
        j=Path(f"data/raw/{ds}/{sub}/eeg/{sub}_task-{task}_eeg.json")
        if j.exists():
            d=json.loads(j.read_text())["RecordingDuration"]
            durations.append(d)
    print(f"{ds}: n={len(durations)}, mean={np.mean(durations):.1f}s, min={np.min(durations):.1f}, max={np.max(durations):.1f}")
