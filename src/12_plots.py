import pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns
from pathlib import Path
import matplotlib
matplotlib.use("Agg")

df=pd.read_csv("data/features_spectral.csv")
rep=pd.read_csv("reports/metrics_real.csv")

# 1. Confusion avg уже есть, делаем heatmap
import ast
# Пересчитаем матрицы из 05_train_eval_real.py логика, но сделаем быстро по rep
# Сделаем bar для BA
plt.figure(figsize=(7,4))
order=["EC_new","EO_new","EO_familiar"]
vals=[rep[rep.test==t]["BA"].mean() for t in order]
err=[rep[rep.test==t]["BA"].std() for t in order]
plt.bar(order, vals, yerr=err, capsize=5)
plt.ylabel("Balanced Accuracy")
plt.title("3-class LogisticRegression, train EC only")
plt.ylim(0,1)
plt.tight_layout()
plt.savefig("figures/bar_ba_3class.png", dpi=150)
print("saved bar_ba_3class.png")

# 2. Feature distributions EO vs EC по диагнозам
for feat, fname in [("rel_alpha_occipital","feat_alpha_occipital"),("alpha_theta_ratio","feat_alpha_theta")]:
    fig, axes=plt.subplots(1,2, figsize=(10,4), sharey=True)
    sns.boxplot(data=df[df.condition=="EC"], x="diagnosis", y=feat, order=["AD","FTD","HC"], ax=axes[0])
    axes[0].set_title(f"EC {feat}")
    sns.boxplot(data=df[df.condition=="EO"], x="diagnosis", y=feat, order=["AD","FTD","HC"], ax=axes[1])
    axes[1].set_title(f"EO {feat}")
    plt.tight_layout()
    plt.savefig(f"figures/{fname}_ec_eo.png", dpi=150)
    plt.close()

# 3. PSD пример уже есть, делаем scatter EC vs EO paired
paired=pd.merge(df[df.condition=="EC"][["participant_id","rel_alpha_occipital","diagnosis"]].rename(columns={"rel_alpha_occipital":"ec"}),
                df[df.condition=="EO"][["participant_id","rel_alpha_occipital"]].rename(columns={"rel_alpha_occipital":"eo"}), on="participant_id", how="inner")
plt.figure(figsize=(6,6))
for dx, color in [("HC","green"),("AD","red"),("FTD","orange")]:
    sub=paired[paired.diagnosis==dx]
    plt.scatter(sub.ec, sub.eo, label=dx, alpha=0.7)
plt.plot([0,0.2],[0,0.2],"k--")
plt.xlabel("EC rel_alpha_occipital"); plt.ylabel("EO rel_alpha_occipital")
plt.legend(); plt.tight_layout()
plt.savefig("figures/scatter_ec_eo_alpha.png", dpi=150)
print("saved scatter")
