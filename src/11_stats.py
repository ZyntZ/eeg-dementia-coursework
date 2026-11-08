import pandas as pd, numpy as np
from pathlib import Path
from sklearn.metrics import balanced_accuracy_score, f1_score
from scipy.stats import ttest_rel, wilcoxon, bootstrap

df=pd.read_csv("data/features_spectral.csv")
rep=pd.read_csv("reports/metrics_real.csv")

# 1. Bootstrap CI по людям для каждого теста
def bootstrap_subject(y_true, y_pred, n_boot=2000, seed=42):
    rng=np.random.default_rng(seed)
    scores=[]
    y_true=np.array(y_true); y_pred=np.array(y_pred)
    for _ in range(n_boot):
        idx=rng.integers(0,len(y_true),len(y_true))
        scores.append(balanced_accuracy_score(y_true[idx], y_pred[idx]))
    lo,hi=np.percentile(scores,[2.5,97.5])
    return lo,hi

# Уже есть в metrics_real.csv, пересчитаем парные p для EC_new vs EO_new
ec=rep[rep.test=="EC_new"].set_index("fold")["BA"]
eo=rep[rep.test=="EO_new"].set_index("fold")["BA"]
fam=rep[rep.test=="EO_familiar"].set_index("fold")["BA"]
print(f"EC_new mean {ec.mean():.3f} std {ec.std():.3f}")
print(f"EO_new mean {eo.mean():.3f} std {eo.std():.3f}")
print(f"EO_familiar mean {fam.mean():.3f} std {fam.std():.3f}")
print(f"EC vs EO ttest p={ttest_rel(ec,eo).pvalue:.4f} wilcoxon p={wilcoxon(ec,eo).pvalue:.4f}")
print(f"EO_fam vs EO_new diff {fam.mean()-eo.mean():.3f} ttest p={ttest_rel(fam,eo).pvalue:.4f}")

# 2. Прямые признаки: парный тест EC vs EO внутри диагноза
subj_dx=df.drop_duplicates("participant_id").set_index("participant_id")["diagnosis"].to_dict()
for feat in ["rel_alpha_occipital","alpha_theta_ratio","rel_theta_mean"]:
    print(f"\n== {feat} ==")
    for dx in ["HC","AD","FTD"]:
        m=pd.merge(df[df.condition=="EC"][["participant_id",feat]].rename(columns={feat:"ec"}),
                   df[df.condition=="EO"][["participant_id",feat]].rename(columns={feat:"eo"}), on="participant_id")
        m["diagnosis"]=m["participant_id"].map(subj_dx)
        sub=m[m.diagnosis==dx]
        if len(sub)<5: continue
        diff=sub.eo-sub.ec
        print(f"{dx} n={len(sub)} EC {sub.ec.mean():.4f} EO {sub.eo.mean():.4f} diff {diff.mean():.4f} t p={ttest_rel(sub.ec,sub.eo).pvalue:.4f}")

# Сохраняем сводку
out=pd.DataFrame({"test":["EC_new","EO_new","EO_familiar"],"BA_mean":[ec.mean(),eo.mean(),fam.mean()],"BA_std":[ec.std(),eo.std(),fam.std()]})
out.to_csv("reports/stats_summary.csv", index=False)
print("\nSaved reports/stats_summary.csv")
