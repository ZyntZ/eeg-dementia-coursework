import pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns
from pathlib import Path
df=pd.read_csv("data/features_spectral.csv")
# Покажем распределения
# 1. Alpha occipital HC vs AD EC
import scipy.stats as st
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
sns.boxplot(data=df[df.condition=="EC"], x="diagnosis", y="rel_alpha_occipital", order=["AD","FTD","HC"])
plt.title("EC: rel alpha occipital")
plt.subplot(1,2,2)
sns.boxplot(data=df[df.condition=="EO"], x="diagnosis", y="rel_alpha_occipital", order=["AD","FTD","HC"])
plt.title("EO: rel alpha occipital")
plt.tight_layout()
plt.savefig("figures/box_alpha_occipital.png", dpi=150)
print("saved box_alpha")

# 2. Alpha/theta ratio
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
sns.boxplot(data=df[df.condition=="EC"], x="diagnosis", y="alpha_theta_ratio", order=["AD","FTD","HC"])
plt.title("EC: alpha/theta ratio")
plt.subplot(1,2,2)
sns.boxplot(data=df[df.condition=="EO"], x="diagnosis", y="alpha_theta_ratio", order=["AD","FTD","HC"])
plt.title("EO: alpha/theta ratio")
plt.tight_layout()
plt.savefig("figures/box_alpha_theta.png", dpi=150)

# 3. Сравним EC vs EO внутри HC и AD
for dx in ["HC","AD","FTD"]:
    sub=df[df.diagnosis==dx]
    ec=sub[sub.condition=="EC"]["rel_alpha_occipital"]
    eo=sub[sub.condition=="EO"]["rel_alpha_occipital"]
    # парный тест только для тех у кого есть оба условия
    merged=pd.merge(sub[sub.condition=="EC"][["participant_id","rel_alpha_occipital"]].rename(columns={"rel_alpha_occipital":"ec"}),
                    sub[sub.condition=="EO"][["participant_id","rel_alpha_occipital"]].rename(columns={"rel_alpha_occipital":"eo"}),
                    on="participant_id")
    from scipy.stats import ttest_rel
    if len(merged)>5:
        print(f"{dx}: EC alpha_occ {ec.mean():.4f} EO {eo.mean():.4f} paired n={len(merged)} p={ttest_rel(merged.ec, merged.eo).pvalue:.4f} diff={merged.eo.mean()-merged.ec.mean():.4f}")
    else:
        print(dx, "no paired")

# 4. Сохраним таблицу средних
tab=df.groupby(["diagnosis","condition"])[["rel_alpha_mean","rel_alpha_occipital","alpha_theta_ratio","spectral_entropy","peak_alpha"]].mean().round(4)
print(tab)
tab.to_csv("reports/feature_means.csv")
