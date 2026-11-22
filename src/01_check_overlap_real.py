import pandas as pd
from pathlib import Path

# 1. Участники
p1 = Path("data/raw/ds004504/participants.tsv")
p2 = Path("data/raw/ds006036/participants.tsv")
df1 = pd.read_csv(p1, sep="\t")
df2 = pd.read_csv(p2, sep="\t")
# чистим BOM и пробелы в именах колонок ds006036
df2.columns = df2.columns.str.replace("﻿","").str.strip().str.replace(" ","")
# приводим к единому формату
print("ds004504:", df1.shape, df1["Group"].value_counts().to_dict())
print("ds006036:", df2.shape, df2["Group"].value_counts().to_dict())
print("\nКолонки ds004504:", df1.columns.tolist())
print("Колонки ds006036:", df2.columns.tolist())
# проверка ID
ids1 = set(df1["participant_id"])
ids2 = set(df2["participant_id"])
print(f"\nПересечение ID: {len(ids1 & ids2)}/88, только в ds004504: {ids1-ids2}, только в ds006036: {ids2-ids1}")
# проверка диагнозов/возраста
merged = pd.merge(df1, df2, on="participant_id", suffixes=("_ec","_eo"))
# Group и Age должны совпадать
mismatch = merged[(merged["Group_ec"]!=merged["Group_eo"]) | (merged["Age_ec"]!=merged["Age_eo"])]
print(f"Несовпадений Group/Age: {len(mismatch)}")
if len(mismatch)>0:
    print(mismatch.head())

# 2. Разметка ГО в ds006036
missing = []
has_open = []
for sub in sorted(ids1):
    ev_path = Path(f"data/raw/ds006036/{sub}/eeg/{sub}_task-photomark_events.tsv")
    if not ev_path.exists():
        missing.append(sub)
        continue
    ev = pd.read_csv(ev_path, sep="\t")
    # ищем open eyes (case insensitive)
    has = ev["value"].str.contains("open eyes", case=False, na=False).any() if "value" in ev.columns else False
    # также проверяем другой столбец - в некоторых версиях value может быть trial_type
    if not has:
        # попробуем все столбцы
        for c in ev.columns:
            if ev[c].astype(str).str.contains("open eyes", case=False, na=False).any():
                has=True; break
    if has:
        has_open.append(sub)
    else:
        missing.append(sub)

print(f"\nРазметка open eyes есть у {len(has_open)}/88, отсутствует у: {missing}")
# сохраняем
merged.to_csv("reports/overlap_real.csv", index=False)
print("Сохранено reports/overlap_real.csv")

# 3. Длительность записей (по .set размеру и json)
import json
for ds, tag in [("ds004504","eyesclosed"), ("ds006036","photomark")]:
    print(f"\n--- {ds} ---")
    for sub in ["sub-001","sub-002","sub-088"]:
        j = Path(f"data/raw/{ds}/{sub}/eeg/{sub}_task-{tag}_eeg.json")
        if j.exists():
            print(sub, json.loads(j.read_text()))
            break
