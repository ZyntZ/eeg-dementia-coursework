import pandas as pd
import numpy as np
from sklearn.metrics import balanced_accuracy_score
from scipy.stats import ttest_rel, wilcoxon

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

# 2. Прямые признаки: парный тест EC_match vs EO внутри диагноза.
#    Именно EC_match, см. комментарий ниже у блока с поправкой FDR.
subj_dx=df.drop_duplicates("participant_id").set_index("participant_id")["diagnosis"].to_dict()
for feat in ["rel_alpha_occipital","alpha_theta_ratio","rel_theta_mean"]:
    print(f"\n== {feat} ==")
    for dx in ["HC","AD","FTD"]:
        m=pd.merge(df[df.condition=="EC_match"][["participant_id",feat]].rename(columns={feat:"ec"}),
                   df[df.condition=="EO"][["participant_id",feat]].rename(columns={feat:"eo"}), on="participant_id")
        m["diagnosis"]=m["participant_id"].map(subj_dx)
        sub=m[m.diagnosis==dx].dropna(subset=["ec","eo"])
        if len(sub)<5: continue
        diff=sub.eo-sub.ec
        print(f"{dx} n={len(sub)} EC_match {sub.ec.mean():.4f} EO {sub.eo.mean():.4f} "
              f"diff {diff.mean():.4f} wilcoxon p={wilcoxon(sub.ec,sub.eo).pvalue:.4f}")

# Сохраняем сводку
out=pd.DataFrame({"test":["EC_new","EO_new","EO_familiar"],"BA_mean":[ec.mean(),eo.mean(),fam.mean()],"BA_std":[ec.std(),eo.std(),fam.std()]})
out.to_csv("reports/stats_summary.csv", index=False)
print("\nSaved reports/stats_summary.csv")
# Bootstrap CI (было не вызывался) + FDR поправка.
#
# База сравнения - EC_match, а не полная EC. Полная запись с закрытыми глазами
# длиннее ГО примерно в 35 раз, спектр по ней оценивается точнее, и парный тест
# EC против EO ловит в том числе разницу в длительности. EC_match - та же
# запись, обрезанная до длительности ГО того же человека.
#
# Критерий - Вилкоксон, а не парный t: доли мощности в диапазоне ограничены
# нулём и единицей и распределены несимметрично.
from statsmodels.stats.multitest import multipletests

BASE = "EC_match"
ALPHA_TRANSITIONS = ["EO", "EO_photic", "EO_nophotic", "EO_broad"]
GROUPS = ["HC", "AD", "FTD"]


def paired(feat, base, cond, dx):
    m = pd.merge(df[df.condition == base][["participant_id", feat]].rename(columns={feat: "a"}),
                 df[df.condition == cond][["participant_id", feat]].rename(columns={feat: "b"}),
                 on="participant_id")
    m["diagnosis"] = m["participant_id"].map(subj_dx)
    return m[m.diagnosis == dx].dropna(subset=["a", "b"])


rows = []
# семья 1: реактивность альфы, все четыре варианта разметки ГО
for cond in ALPHA_TRANSITIONS:
    for dx in GROUPS:
        sub = paired("rel_alpha_occipital", BASE, cond, dx)
        if len(sub) >= 5:
            rows.append({"family": "alpha_reactivity",
                         "test": f"rel_alpha_occipital / {BASE}->{cond} / {dx}",
                         "n": len(sub), "mean_from": sub.a.mean(), "mean_to": sub.b.mean(),
                         "change_pct": 100 * (sub.b.mean() / sub.a.mean() - 1),
                         "p_raw": wilcoxon(sub.a, sub.b).pvalue})

# семья 2: остальные признаки на основном переходе
for feat in ["alpha_theta_ratio", "rel_theta_mean"]:
    for dx in GROUPS:
        sub = paired(feat, BASE, "EO", dx)
        if len(sub) >= 5:
            rows.append({"family": "other_features",
                         "test": f"{feat} / {BASE}->EO / {dx}",
                         "n": len(sub), "mean_from": sub.a.mean(), "mean_to": sub.b.mean(),
                         "change_pct": 100 * (sub.b.mean() / sub.a.mean() - 1),
                         "p_raw": wilcoxon(sub.a, sub.b).pvalue})

st = pd.DataFrame(rows)
# поправка внутри каждой семьи отдельно
st["p_fdr"] = np.nan
for fam, idx in st.groupby("family").groups.items():
    _, corr, _, _ = multipletests(st.loc[idx, "p_raw"], method="fdr_bh")
    st.loc[idx, "p_fdr"] = corr
st["significant"] = st.p_fdr < 0.05
st = st.round({"mean_from": 4, "mean_to": 4, "change_pct": 1, "p_raw": 5, "p_fdr": 5})
st.to_csv("reports/stats_paired_fdr.csv", index=False)
print("\nПарные тесты (Вилкоксон), база " + BASE + ", поправка BH внутри семьи:")
print(st.to_string(index=False))

# Пример bootstrap для LOSO
try:
    loso=pd.read_csv("reports/loso_preds.csv")
    lo, hi = bootstrap_subject(loso[loso.condition=="EC"].y_true, loso[loso.condition=="EC"].y_pred)
    print(f"LOSO EC bootstrap 95% CI: [{lo:.3f}, {hi:.3f}] (n={len(loso[loso.condition=='EC'])})")
except Exception as e:
    print(f"bootstrap skip: {e}")
print("WARN: t-тесты по 5 фолдам маломощны (n=5) — см. README")
