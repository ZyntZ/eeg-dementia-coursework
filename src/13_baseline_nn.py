import pandas as pd, numpy as np, random, os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
import torch, torch.nn as nn
# Фиксация сида для воспроизводимости
SEED=42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False

df=pd.read_csv("data/features_spectral.csv")
feats=["rel_delta_mean","rel_theta_mean","rel_alpha_mean","rel_beta_mean","rel_alpha_occipital","alpha_theta_ratio","spectral_entropy","peak_alpha"]
subjects=df["participant_id"].unique()
subj_dx=df.drop_duplicates("participant_id").set_index("participant_id")["diagnosis"].to_dict()
y_subj=np.array([subj_dx[s] for s in subjects])
label_map={"AD":0,"FTD":1,"HC":2}

class MLP(nn.Module):
    def __init__(self, d_in=8, d_h=32):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(d_in,d_h), nn.ReLU(), nn.Dropout(0.3), nn.Linear(d_h,d_h), nn.ReLU(), nn.Linear(d_h,3))
    def forward(self,x): return self.net(x)

skf=StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
res=[]
for train_idx,test_idx in skf.split(subjects, y_subj):
    train_subs=subjects[train_idx]; test_subs=subjects[test_idx]
    train_mask=df["participant_id"].isin(train_subs) & (df["condition"]=="EC")
    X_train=df.loc[train_mask, feats].values; y_train=np.array([label_map[d] for d in df.loc[train_mask,"diagnosis"]])
    scaler=StandardScaler().fit(X_train)
    X_train_s=scaler.transform(X_train)
    # torch
    Xtr=torch.tensor(X_train_s, dtype=torch.float32); ytr=torch.tensor(y_train, dtype=torch.long)
    model=MLP(d_in=len(feats))
    opt=torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    crit=nn.CrossEntropyLoss()
    model.train()
    for epoch in range(80):
        opt.zero_grad(); loss=crit(model(Xtr), ytr); loss.backward(); opt.step()
    model.eval()
    for name, cond, subs in [("EC_new","EC",test_subs),("EO_new","EO",test_subs),("EO_familiar","EO",train_subs)]:
        mask=df["participant_id"].isin(subs) & (df["condition"]==cond)
        if mask.sum()==0: continue
        X_test=scaler.transform(df.loc[mask, feats].values)
        y_test=np.array([label_map[d] for d in df.loc[mask,"diagnosis"]])
        with torch.no_grad():
            pred=model(torch.tensor(X_test, dtype=torch.float32)).argmax(1).numpy()
        ba=balanced_accuracy_score(y_test, pred)
        res.append((name, ba))
        print(f"MLP {name} BA={ba:.3f} n={len(y_test)}")

import pandas as pd
r=pd.DataFrame(res, columns=["test","BA"])
print(r.groupby("test")["BA"].mean())
r.to_csv("reports/metrics_mlp.csv", index=False)
