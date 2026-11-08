import pandas as pd, numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, confusion_matrix, classification_report
from pathlib import Path
from tqdm import tqdm

df=pd.read_csv("data/features_spectral.csv")
feats=["rel_delta_mean","rel_theta_mean","rel_alpha_mean","rel_beta_mean","rel_alpha_occipital","alpha_theta_ratio","spectral_entropy","peak_alpha"]
subjects=df["participant_id"].unique()
subj_dx=df.drop_duplicates("participant_id").set_index("participant_id")["diagnosis"].to_dict()
label_map={"AD":0,"FTD":1,"HC":2}
inv_label={v:k for k,v in label_map.items()}

# LOSO: 88 итераций, train на 87 EC, test на 1 EC + 1 EO того же человека
results=[]
cms_ec=np.zeros((3,3)); cms_eo=np.zeros((3,3))
for test_sub in tqdm(subjects, desc="LOSO"):
    train_subs=[s for s in subjects if s!=test_sub]
    train_mask=df["participant_id"].isin(train_subs) & (df["condition"]=="EC")
    X_train=df.loc[train_mask, feats].values
    y_train=np.array([label_map[d] for d in df.loc[train_mask,"diagnosis"]])
    scaler=StandardScaler().fit(X_train)
    clf=LogisticRegression(multi_class="multinomial", solver="lbfgs", C=1.0, max_iter=1000, class_weight="balanced")
    clf.fit(scaler.transform(X_train), y_train)
    for cond, cm_acc in [("EC",cms_ec),("EO",cms_eo)]:
        mask=df["participant_id"]==test_sub
        # для EO у sub-011 нет записи -> пропуск
        mask=mask & (df["condition"]==cond)
        if mask.sum()==0: continue
        X_test=scaler.transform(df.loc[mask, feats].values)
        y_test=np.array([label_map[d] for d in df.loc[mask,"diagnosis"]])
        y_pred=clf.predict(X_test)
        cm_acc[y_test[0], y_pred[0]]+=1
        results.append({"test_sub":test_sub,"condition":cond,"y_true":inv_label[y_test[0]],"y_pred":inv_label[y_pred[0]],"diagnosis":subj_dx[test_sub]})

res=pd.DataFrame(results)
print(f"LOSO total {len(res)} preds")
# Общие метрики
for cond in ["EC","EO"]:
    sub=res[res.condition==cond]
    if len(sub)==0: continue
    y_true=sub["y_true"]; y_pred=sub["y_pred"]
    ba=balanced_accuracy_score(y_true, y_pred)
    f1=f1_score(y_true, y_pred, average="macro", zero_division=0)
    print(f"\n{cond} BA={ba:.3f} F1={f1:.3f} n={len(sub)}")
    print(classification_report(y_true, y_pred, digits=3, zero_division=0))
    cm=confusion_matrix(y_true, y_pred, labels=["AD","FTD","HC"])
    print(cm)

# Дополнительно: HC vs пациенты хорошо делятся? (бинарная)
from sklearn.metrics import balanced_accuracy_score as bas
for cond in ["EC","EO"]:
    sub=res[res.condition==cond].copy()
    sub["y_true_bin"]=sub["y_true"].apply(lambda x: "HC" if x=="HC" else "PAT")
    sub["y_pred_bin"]=sub["y_pred"].apply(lambda x: "HC" if x=="HC" else "PAT")
    ba=bas(sub["y_true_bin"], sub["y_pred_bin"])
    print(f"{cond} PAT vs HC BA={ba:.3f}")

res.to_csv("reports/loso_preds.csv", index=False)
np.save("reports/loso_cm_ec.npy", cms_ec)
np.save("reports/loso_cm_eo.npy", cms_eo)
print("saved reports/loso_preds.csv and cms")
