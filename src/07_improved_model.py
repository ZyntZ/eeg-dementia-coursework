import pandas as pd, numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

df=pd.read_csv("data/features_spectral.csv")
# Попробуем разные наборы признаков
sets={
 "all_mean": ["rel_delta_mean","rel_theta_mean","rel_alpha_mean","rel_beta_mean","alpha_theta_ratio","spectral_entropy","peak_alpha"],
 "alpha_focus": ["rel_alpha_mean","rel_alpha_occipital","rel_alpha_frontal","alpha_theta_ratio","peak_alpha","spectral_entropy"],
 "no_delta": ["rel_theta_mean","rel_alpha_mean","rel_beta_mean","rel_theta_occipital","rel_alpha_occipital","alpha_theta_ratio","theta_beta_ratio","spectral_entropy"],
 "ratio_only": ["alpha_theta_ratio","theta_beta_ratio","spectral_entropy","peak_alpha","rel_alpha_occipital"],
 "occipital": ["rel_alpha_occipital","rel_theta_occipital","rel_delta_occipital","rel_beta_occipital","alpha_theta_ratio"]
}
subjects = df["participant_id"].unique()
subj_dx = df.drop_duplicates("participant_id").set_index("participant_id")["diagnosis"].to_dict()
y_subj = np.array([subj_dx[s] for s in subjects])

for name, feats in sets.items():
    print(f"\n=== {name} {feats} ===")
    skf=StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    res=[]
    for fold,(train_idx,test_idx) in enumerate(skf.split(subjects, y_subj),1):
        train_subs=subjects[train_idx]; test_subs=subjects[test_idx]
        train_mask=df["participant_id"].isin(train_subs) & (df["condition"]=="EC")
        X_train=df.loc[train_mask, feats].values; y_train=df.loc[train_mask,"diagnosis"].values
        scaler=StandardScaler().fit(X_train)
        clf=LogisticRegression(multi_class="multinomial", solver="lbfgs", C=1.0, max_iter=1000, class_weight="balanced")
        clf.fit(scaler.transform(X_train), y_train)
        for test_name, cond, subs in [("EC_new","EC",test_subs),("EO_new","EO",test_subs)]:
            mask=df["participant_id"].isin(subs) & (df["condition"]==cond)
            if mask.sum()==0: continue
            X_test=df.loc[mask, feats].values; y_test=df.loc[mask,"diagnosis"].values
            pred=clf.predict(scaler.transform(X_test))
            res.append((test_name, balanced_accuracy_score(y_test,pred)))
    import pandas as pd
    r=pd.DataFrame(res, columns=["test","BA"])
    print(r.groupby("test")["BA"].mean().round(3).to_dict())

# Попробуем бинарную лучше
print("\n=== Бинарная AD vs HC разные наборы ===")
for name, feats in sets.items():
    skf=StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    b=[]
    for train_idx,test_idx in skf.split(subjects, y_subj):
        train_subs=subjects[train_idx]; test_subs=subjects[test_idx]
        train_mask=df["participant_id"].isin(train_subs) & (df["condition"]=="EC") & (df["diagnosis"].isin(["AD","HC"]))
        X_train=df.loc[train_mask, feats].values; y_train=df.loc[train_mask,"diagnosis"].values
        if len(np.unique(y_train))<2: continue
        scaler=StandardScaler().fit(X_train)
        clf=LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced")
        clf.fit(scaler.transform(X_train), y_train)
        for test_name, cond in [("EC_new","EC"),("EO_new","EO")]:
            mask=df["participant_id"].isin(test_subs) & (df["condition"]==cond) & (df["diagnosis"].isin(["AD","HC"]))
            if mask.sum()<3: continue
            X_test=df.loc[mask, feats].values; y_test=df.loc[mask,"diagnosis"].values
            pred=clf.predict(scaler.transform(X_test))
            b.append((test_name, balanced_accuracy_score(y_test,pred)))
    r=pd.DataFrame(b, columns=["test","BA"])
    if not r.empty:
        print(name, r.groupby("test")["BA"].mean().round(3).to_dict())

# Проверим гипотезу руководителя напрямую на признаках без МЛ: HC падает альфа при EO, пациенты — нет
print("\n=== Проверка гипотезы руководителя (прямо по альфа) ===")
for dx in ["HC","AD","FTD"]:
    m=pd.merge(df[df.condition=="EC"][["participant_id","rel_alpha_occipital"]].rename(columns={"rel_alpha_occipital":"ec"}),
               df[df.condition=="EO"][["participant_id","rel_alpha_occipital"]].rename(columns={"rel_alpha_occipital":"eo"}),
               on="participant_id")
    # фильтр по диагнозу
    m["diagnosis"]=m["participant_id"].map(subj_dx)
    sub=m[m.diagnosis==dx]
    from scipy.stats import ttest_rel
    print(dx, f"n={len(sub)} EC {sub.ec.mean():.4f} EO {sub.eo.mean():.4f} diff {sub.eo.mean()-sub.ec.mean():.4f} p={ttest_rel(sub.ec, sub.eo).pvalue:.4f}")

