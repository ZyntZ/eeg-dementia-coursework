import pandas as pd, numpy as np, torch, torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score, f1_score, confusion_matrix
from pathlib import Path
from tqdm import tqdm

df=pd.read_csv("data/features_spectral.csv")
feats=["rel_delta_mean","rel_theta_mean","rel_alpha_mean","rel_beta_mean","rel_alpha_occipital","alpha_theta_ratio","spectral_entropy","peak_alpha"]
subjects=df["participant_id"].unique()
subj_dx=df.drop_duplicates("participant_id").set_index("participant_id")["diagnosis"].to_dict()
label_map={"AD":0,"FTD":1,"HC":2}
inv_label={v:k for k,v in label_map.items()}

class MLP(nn.Module):
    def __init__(self, d_in=8, d_h=32):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(d_in,d_h), nn.ReLU(), nn.Dropout(0.3), nn.Linear(d_h,d_h), nn.ReLU(), nn.Linear(d_h,3))
    def forward(self,x): return self.net(x)

results=[]
cms_ec=np.zeros((3,3)); cms_eo=np.zeros((3,3))
for test_sub in tqdm(subjects, desc="MLP LOSO"):
    train_subs=[s for s in subjects if s!=test_sub]
    train_mask=df["participant_id"].isin(train_subs) & (df["condition"]=="EC")
    X_train=df.loc[train_mask, feats].values
    y_train=np.array([label_map[d] for d in df.loc[train_mask,"diagnosis"]])
    scaler=StandardScaler().fit(X_train)
    X_train_s=scaler.transform(X_train)
    Xtr=torch.tensor(X_train_s, dtype=torch.float32); ytr=torch.tensor(y_train, dtype=torch.long)
    model=MLP(d_in=len(feats))
    opt=torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    crit=nn.CrossEntropyLoss()
    model.train()
    for epoch in range(40):
        opt.zero_grad(); loss=crit(model(Xtr), ytr); loss.backward(); opt.step()
    model.eval()
    for cond, cm_acc in [("EC",cms_ec),("EO",cms_eo)]:
        mask=(df["participant_id"]==test_sub) & (df["condition"]==cond)
        if mask.sum()==0: continue
        X_test=scaler.transform(df.loc[mask, feats].values)
        y_test=np.array([label_map[d] for d in df.loc[mask,"diagnosis"]])
        with torch.no_grad():
            pred=model(torch.tensor(X_test, dtype=torch.float32)).argmax(1).numpy()
        cm_acc[y_test[0], pred[0]]+=1
        results.append({"test_sub":test_sub,"condition":cond,"y_true":inv_label[y_test[0]],"y_pred":inv_label[pred[0]]})

res=pd.DataFrame(results)
print(f"MLP LOSO total {len(res)}")
for cond in ["EC","EO"]:
    sub=res[res.condition==cond]
    ba=balanced_accuracy_score(sub["y_true"], sub["y_pred"])
    f1=f1_score(sub["y_true"], sub["y_pred"], average="macro", zero_division=0)
    print(f"{cond} BA={ba:.3f} F1={f1:.3f} n={len(sub)}")
    print(confusion_matrix(sub["y_true"], sub["y_pred"], labels=["AD","FTD","HC"]))
res.to_csv("reports/loso_mlp_preds.csv", index=False)
np.save("reports/loso_mlp_cm_ec.npy", cms_ec)
np.save("reports/loso_mlp_cm_eo.npy", cms_eo)
print("saved reports/loso_mlp_*")
