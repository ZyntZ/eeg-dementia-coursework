import pandas as pd, numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import balanced_accuracy_score, f1_score, confusion_matrix, classification_report

df=pd.read_csv("data/features_spectral.csv")
print(df.shape)
print(df.groupby(["condition","diagnosis"]).size())

# фичи — используем те же что в плане: относительные мощности + соотношения + энтропия + пик альфа
feature_cols = [c for c in df.columns if c.startswith("rel_") or c in ["alpha_theta_ratio","theta_beta_ratio","spectral_entropy","peak_alpha"]]
print("features:", feature_cols)
# уберем frontal/occipital дубли? оставим mean + ratio + entropy + peak
selected = ["rel_delta_mean","rel_theta_mean","rel_alpha_mean","rel_beta_mean",
            "rel_alpha_occipital","rel_theta_occipital",
            "alpha_theta_ratio","theta_beta_ratio","spectral_entropy","peak_alpha"]
print("selected:", selected)

# подготовим матрицу для EC и EO отдельно, но строки уже агрегированы по субъекту
# для протокола: обучение только на EC обучающих, тест на 3 условиях
subjects = df["participant_id"].unique()
# маппинг диагноз
subj_dx = df.drop_duplicates("participant_id").set_index("participant_id")["diagnosis"].to_dict()
y_subj = np.array([subj_dx[s] for s in subjects])

# 5-fold stratified group (группы = субъекты)
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

results=[]
cms=[]
for fold, (train_idx, test_idx) in enumerate(skf.split(subjects, y_subj), 1):
    train_subs = subjects[train_idx]
    test_subs = subjects[test_idx]
    # train: EC только обучающих
    train_mask = df["participant_id"].isin(train_subs) & (df["condition"]=="EC")
    X_train = df.loc[train_mask, selected].values
    y_train = df.loc[train_mask, "diagnosis"].values
    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    clf = LogisticRegression(multi_class="multinomial", solver="lbfgs", C=1.0, max_iter=1000, class_weight="balanced")
    clf.fit(X_train_s, y_train)

    # тесты
    for name, cond, subs in [
        ("EC_new","EC", test_subs),
        ("EO_new","EO", test_subs),
        ("EO_familiar","EO", train_subs),
    ]:
        mask = df["participant_id"].isin(subs) & (df["condition"]==cond)
        # для EO_familiar исключаем sub-011 у которого нет EO
        if mask.sum()==0:
            continue
        X_test = df.loc[mask, selected].values
        y_test = df.loc[mask, "diagnosis"].values
        y_pred = clf.predict(scaler.transform(X_test))
        ba = balanced_accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
        # per class
        print(f"fold {fold} {name} n={len(y_test)} BA={ba:.3f} F1={f1:.3f}")
        print(classification_report(y_test, y_pred, zero_division=0, digits=2))
        cms.append({"fold":fold,"test":name,"cm": confusion_matrix(y_test, y_pred, labels=["AD","FTD","HC"])})
        results.append({"fold":fold,"test":name,"n":len(y_test),"BA":ba,"F1_macro":f1,
                        "y_true": list(y_test), "y_pred": list(y_pred)})

res=pd.DataFrame([{"fold":r["fold"],"test":r["test"],"n":r["n"],"BA":r["BA"],"F1_macro":r["F1_macro"]} for r in results])
print("\n=== Среднее по фолдам ===")
print(res.groupby("test")[["BA","F1_macro"]].mean().round(3))
print(res.groupby("test")[["BA","F1_macro"]].std().round(3))

# Парное сравнение EC_new vs EO_new на одних и тех же новых людях
# Для каждого фолда считаем BA отдельно и делаем парный тест
from scipy.stats import ttest_rel, wilcoxon
ec_ba = res[res.test=="EC_new"].set_index("fold")["BA"]
eo_ba = res[res.test=="EO_new"].set_index("fold")["BA"]
print(f"\nПарное: EC_new mean {ec_ba.mean():.3f} vs EO_new mean {eo_ba.mean():.3f}")
print("ttest_rel p:", ttest_rel(ec_ba, eo_ba).pvalue)
# Также сравним familiar vs new для EO
fam_ba = res[res.test=="EO_familiar"].set_index("fold")["BA"]
print(f"EO_familiar mean {fam_ba.mean():.3f} vs EO_new mean {eo_ba.mean():.3f} diff {fam_ba.mean()-eo_ba.mean():.3f}")

# Сохраним
res.to_csv("reports/metrics_real.csv", index=False)
print("saved reports/metrics_real.csv")

# Конфузия усредненная
import numpy as np
for test in ["EC_new","EO_new","EO_familiar"]:
    cms_t = [c["cm"] for c in cms if c["test"]==test]
    avg = np.mean(cms_t, axis=0)
    print(f"\n{test} avg CM (AD,FTD,HC):\n", np.round(avg,1))

# Дополнительно: бинарная задача AD vs HC (как в фокусе курсовой)
print("\n=== Бинарная AD vs HC (переобучаем) ===")
for fold, (train_idx, test_idx) in enumerate(skf.split(subjects, y_subj), 1):
    # фильтруем только AD/HC в train/test
    train_subs = subjects[train_idx]
    test_subs = subjects[test_idx]
    train_mask = df["participant_id"].isin(train_subs) & (df["condition"]=="EC") & (df["diagnosis"].isin(["AD","HC"]))
    if train_mask.sum()==0: continue
    X_train = df.loc[train_mask, selected].values
    y_train = df.loc[train_mask, "diagnosis"].values
    scaler = StandardScaler().fit(X_train)
    clf = LogisticRegression(solver="lbfgs", C=1.0, max_iter=1000, class_weight="balanced")
    clf.fit(scaler.transform(X_train), y_train)
    for name, cond, subs in [("EC_new","EC",test_subs),("EO_new","EO",test_subs)]:
        mask = df["participant_id"].isin(subs) & (df["condition"]==cond) & (df["diagnosis"].isin(["AD","HC"]))
        if mask.sum()<3: continue
        X_test=df.loc[mask, selected].values; y_test=df.loc[mask,"diagnosis"].values
        y_pred=clf.predict(scaler.transform(X_test))
        ba=balanced_accuracy_score(y_test, y_pred)
        print(f"fold {fold} {name} binary BA={ba:.3f} n={len(y_test)}")

