import numpy as np, pandas as pd
from pathlib import Path
# wPLI требует raw, если нет raw — заглушка
try:
    from mne_connectivity import spectral_connectivity_epochs
    HAS_CONN=True
except:
    HAS_CONN=False

if __name__=="__main__":
    # Честно: wPLI не считался — нет raw на Kaggle, lib опциональна. Пишем NaN c пометкой.
    print("WARN: connectivity stub — wPLI не считался (raw отсутствует / mne_connectivity опционально)")
    print("Файл features_with_conn.csv создан для совместимости, но значения NaN — не использовать в оценке")
    df=pd.read_csv("data/features_spectral.csv")
    df["wpli_alpha_O_F"]=np.nan
    df["wpli_theta_O_F"]=np.nan
    df.to_csv("data/features_with_conn.csv", index=False)
    print("saved data/features_with_conn.csv with NaNs (stub)")
