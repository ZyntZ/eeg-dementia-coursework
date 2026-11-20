"""
22_eval_state.py - проверка гипотезы руководителя про состояния.

Из письма от 5 октября: "сеть не должна увидеть существенной разницы между
ГЗ и ГО, ГО+фотостимуляция. По этим классам у пациентов должно быть множество
ложноположительных ответов. Тогда как на контроле должны получить достоверные
отличия по данным состояниям."

То есть классифицировать надо не диагноз, а СОСТОЯНИЕ, и отдельно внутри
контроля и внутри пациентов.

Пары заданы явно, а не всеми сочетаниями: EO_photic и EO_nophotic это части
EO, сравнивать множество с собственным подмножеством бессмысленно.

Считаем на двух наборах признаков:
  all   - все спектральные признаки;
  alpha - только альфа (относительная по областям, alpha/theta, пик).
Разделение нужно потому, что открывание глаз меняет не только альфу: растёт
мышечная активность и бета, и по ним состояние ловится у всех. Гипотеза
руководителя про реактивность - именно про альфу.

Контрольная пара EC vs EC_match: один и тот же кусок записи, разная длина.
Если модель её различает, значит она ловит длительность, а не состояние.

Запуск: python src/22_eval_state.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FEATURES_CSV = "data/features_spectral.csv"
OUT = "reports/eval_state.csv"
SEED = 42
N_PERM = 1000

GROUPS = {"HC": ["HC"], "patients": ["AD", "FTD"], "AD": ["AD"], "FTD": ["FTD"]}

PAIRS = [
    ("EC", "EC_match"),            # контроль: только длина, состояние то же
    ("EC_match", "EO"),            # основная пара
    ("EC_match", "EO_photic"),     # ГЗ против ГО со вспышками
    ("EC_match", "EO_nophotic"),   # ГЗ против ГО без вспышек
    ("EO_nophotic", "EO_photic"),  # влияют ли сами вспышки
    ("EC_match", "EO_broad"),      # широкое правило разметки глаз
]


def model():
    return make_pipeline(StandardScaler(),
                         LogisticRegression(C=1.0, max_iter=5000, class_weight="balanced"))


def perm_p(y, pred, n=N_PERM, seed=SEED):
    rng = np.random.default_rng(seed)
    obs = balanced_accuracy_score(y, pred)
    y = np.asarray(y)
    cnt = sum(balanced_accuracy_score(rng.permutation(y), pred) >= obs for _ in range(n))
    return (cnt + 1) / (n + 1)


def run_pair(df, feats, dxs, c1, c2):
    d = df[df.diagnosis.isin(dxs) & df.condition.isin([c1, c2])]
    ok = d.groupby("participant_id").condition.nunique() == 2
    ids = ok[ok].index.tolist()
    if len(ids) < 10:
        return None
    d = d[d.participant_id.isin(ids)]
    X, y, g = d[feats].values, d.condition.values, d.participant_id.values
    pred = np.empty(len(y), dtype=object)
    for s in ids:                       # обе записи человека уходят в тест
        te = g == s
        pred[te] = model().fit(X[~te], y[~te]).predict(X[te])
    cm = confusion_matrix(y, pred, labels=[c1, c2])
    return dict(n_subjects=len(ids),
                BA=balanced_accuracy_score(y, pred),
                p_perm=perm_p(y, pred),
                recall_c1=cm[0, 0] / max(cm[0].sum(), 1),
                recall_c2=cm[1, 1] / max(cm[1].sum(), 1))


def main():
    df = pd.read_csv(FEATURES_CSV)
    all_feats = [c for c in df.columns if c.startswith("rel_")] + \
                [c for c in ("alpha_theta_ratio", "theta_beta_ratio",
                             "spectral_entropy", "peak_alpha") if c in df.columns]
    alpha_feats = [c for c in df.columns if c.startswith("rel_alpha")] + \
                  [c for c in ("alpha_theta_ratio", "peak_alpha") if c in df.columns]
    for c in all_feats:
        df[c] = df.groupby("condition")[c].transform(lambda s: s.fillna(s.median()))

    have = set(df.condition)
    rows = []
    for c1, c2 in PAIRS:
        if c1 not in have or c2 not in have:
            print(f"пропуск {c1} vs {c2}: нет в таблице")
            continue
        for fname, feats in (("all", all_feats), ("alpha", alpha_feats)):
            for gname, dxs in GROUPS.items():
                r = run_pair(df, feats, dxs, c1, c2)
                if r:
                    rows.append(dict(pair=f"{c1} vs {c2}", features=fname,
                                     group=gname, **r))

    out = pd.DataFrame(rows).round(3)
    Path("reports").mkdir(exist_ok=True)
    out.to_csv(OUT, index=False)
    for f in ("all", "alpha"):
        print(f"\n===== признаки: {f} =====")
        print(out[out.features == f].drop(columns="features").to_string(index=False))
    print(f"\nсохранено {OUT}")
    print("\nСмотреть строку HC против patients в пределах одной пары.")
    print("Пара EC vs EC_match - контрольная, там BA должна быть около 0.5.")


if __name__ == "__main__":
    main()
