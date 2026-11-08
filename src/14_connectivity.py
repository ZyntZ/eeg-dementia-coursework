import numpy as np, pandas as pd
from pathlib import Path
# wPLI требует raw, если нет raw — заглушка
try:
    from mne_connectivity import spectral_connectivity_epochs
    HAS_CONN=True
except:
    HAS_CONN=False

if __name__=="__main__":
    if not HAS_CONN or not Path("data/raw/ds004504/sub-001/eeg").exists():
        print("raw not present or lib missing, skip")
        df=pd.read_csv("data/features_spectral.csv")
        df["wpli_alpha_O_F"]=np.nan
        df["wpli_theta_O_F"]=np.nan
        df.to_csv("data/features_with_conn.csv", index=False)
        print("saved data/features_with_conn.csv with NaNs")
    else:
        print("connectivity would run here")
