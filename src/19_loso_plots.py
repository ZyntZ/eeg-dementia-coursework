import pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
from sklearn.metrics import confusion_matrix

# LOSO heatmaps EC vs EO для logreg
for name, path in [("logreg","reports/loso_preds.csv"),("mlp","reports/loso_mlp_preds.csv")]:
    df=pd.read_csv(path)
    for cond in ["EC","EO"]:
        sub=df[df.condition==cond]
        cm=confusion_matrix(sub["y_true"], sub["y_pred"], labels=["AD","FTD","HC"])
        cm_norm=cm.astype(float)/cm.sum(axis=1, keepdims=True)
        plt.figure(figsize=(4,3))
        sns.heatmap(cm_norm, annot=True, fmt=".2f", xticklabels=["AD","FTD","HC"], yticklabels=["AD","FTD","HC"], cmap="Blues")
        plt.title(f"LOSO {name} {cond} BA { (cm_norm.diagonal().mean()):.3f}")
        plt.xlabel("pred"); plt.ylabel("true")
        plt.tight_layout()
        plt.savefig(f"figures/loso_{name}_{cond}_cm.png", dpi=150)
        plt.close()
        print(f"saved loso_{name}_{cond}_cm.png")

# Сравнение протоколов bar
comp=pd.read_csv("reports/compare_protocols.csv")
plt.figure(figsize=(9,4))
# фильтр только EC/EO без familiar для чистоты
filt=comp[comp.condition.isin(["EC","EC_new","EO","EO_new"])]
# Сведем EC_new->EC, EO_new->EO
filt["cond_simple"]=filt["condition"].apply(lambda x: "EC" if "EC" in x else "EO")
pivot=filt.pivot(index="protocol", columns="cond_simple", values="BA")
pivot=pivot.reindex(["5fold_logreg","5fold_mlp","loso_logreg","loso_mlp"])
pivot.plot(kind="bar", figsize=(8,4))
plt.ylabel("BA"); plt.title("Protocol compare EC vs EO")
plt.ylim(0,0.6); plt.xticks(rotation=15); plt.tight_layout()
plt.savefig("figures/compare_protocols_bar.png", dpi=150)
print("saved compare_protocols_bar.png")

# Per-subject correctness
loso=pd.read_csv("reports/loso_preds.csv")
loso["correct"]=(loso.y_true==loso.y_pred).astype(int)
per_sub=loso.groupby("test_sub")["correct"].mean().reset_index()
per_sub.to_csv("reports/loso_per_subject.csv", index=False)
plt.figure(figsize=(8,3))
plt.hist(per_sub["correct"], bins=np.arange(0,1.1,0.2))
plt.xlabel("per-subject accuracy (2 conditions)"); plt.ylabel("n subjects")
plt.tight_layout(); plt.savefig("figures/loso_per_subject_hist.png", dpi=150)
print("saved per_subject")
