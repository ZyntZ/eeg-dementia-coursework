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
# Bootstrap CI (было не вызывался) + FDR поправка
from statsmodels.stats.multitest import multipletests
pvals=[]; names=[]
for feat in ["rel_alpha_occipital","alpha_theta_ratio","rel_theta_mean"]:
    for dx in ["HC","AD","FTD"]:
        m=pd.merge(df[df.condition=="EC"][["participant_id",feat]].rename(columns={feat:"ec"}),
                   df[df.condition=="EO"][["participant_id",feat]].rename(columns={feat:"eo"}), on="participant_id")
        m["diagnosis"]=m["participant_id"].map(subj_dx)
        sub=m[m.diagnosis==dx]
        if len(sub)>=5:
            pvals.append(ttest_rel(sub.ec,sub.eo).pvalue)
            names.append(f"{feat} / {dx}")
if pvals:
    rej, p_corr, _, _ = multipletests(pvals, method='fdr_bh')
    print(f"\nFDR BH по {len(pvals)} тестам:")
    for n, raw_p, adj, r in zip(names, pvals, p_corr, rej):
        print(f"  {n:34s} p={raw_p:.4f}  FDR={adj:.4f}  {'значимо' if r else ''}")
    pd.DataFrame({"test":names,"p_raw":pvals,"p_fdr":p_corr,"significant":rej}).to_csv(
        "reports/stats_paired_fdr.csv", index=False)
# Пример bootstrap для LOSO
try:
    loso=pd.read_csv("reports/loso_preds.csv")
    lo, hi = bootstrap_subject(loso[loso.condition=="EC"].y_true, loso[loso.condition=="EC"].y_pred)
    print(f"LOSO EC bootstrap 95% CI: [{lo:.3f}, {hi:.3f}] (n={len(loso[loso.condition=='EC'])})")
except Exception as e:
    print(f"bootstrap skip: {e}")
print("WARN: t-тесты по 5 фолдам маломощны (n=5) — см. README")
