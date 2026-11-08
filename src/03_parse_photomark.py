import pandas as pd
from pathlib import Path

# Проверим что именно размечено как open/closed eyes
# ds006036 — ключевой: нужно выделить интервалы ГО для чистого сравнения с ГЗ
def analyze_photomark(sub="sub-001"):
    df=pd.read_csv(f"data/raw/ds006036/{sub}/eeg/{sub}_task-photomark_events.tsv", sep="\t")
    print(f"{sub}: всего {len(df)} событий")
    # уникальные value
    print(df["value"].value_counts().head(20))
    # открытые/закрытые
    o = df[df["value"].str.contains("open eyes", case=False, na=False)]
    c = df[df["value"].str.contains("closed eyes", case=False, na=False)]
    print(f"\nopen eyes: {len(o)}, closed eyes: {len(c)}")
    print(o.head(10).to_string())
    print(c.head(10).to_string())
    # фотостимуляция маркеры
    photo = df[df["value"].str.contains("PHOTO", na=False)]
    print("\nPHOTO маркеры:")
    print(photo.head(10).to_string())
    # интервалы: считаем сколько времени ГО
    # value содержит onset, duration=0, так что нужно парсить интервалы между open и closed
    events = df.sort_values("onset")
    open_times = events[events["value"].str.contains("open eyes", case=False, na=False)]["onset"].values
    closed_times = events[events["value"].str.contains("closed eyes", case=False, na=False)]["onset"].values
    print(f"\nopen_times: {open_times[:5]}")
    print(f"closed_times: {closed_times[:5]}")
    # проверим структуру: обычно open -> ... -> closed
    # посчитаем суммарную длительность ГО
    import numpy as np
    total_open=0
    for ot in open_times:
        # найти следующее closed после open
        nxt = closed_times[closed_times>ot]
        if len(nxt)>0:
            total_open += nxt[0]-ot
    print(f"Суммарно GO: {total_open:.1f}s из {df['onset'].max():.1f}s")

for sub in ["sub-001","sub-002","sub-011","sub-037"]:
    print("\n"+"="*60)
    analyze_photomark(sub)

# Проверим sub-011 — почему нет разметки
print("\n"+"="*60)
print("sub-011 подробно:")
df=pd.read_csv(f"data/raw/ds006036/sub-011/eeg/sub-011_task-photomark_events.tsv", sep="\t")
print(df["value"].unique())
