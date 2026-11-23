"""
27_feature_space.py - что даёт расширение набора признаков.

Руководитель предложил подправить признаковое пространство, чтобы поднять
точность. Здесь это проверяется на трёх вещах сразу:

  - помогает ли связность (wPLI из 14) поверх спектральных признаков;
  - помогает ли отбор признаков;
  - сколько прироста от отбора настоящее, а сколько появляется из-за того,
    что отбор сделан на всей выборке, включая проверяемого человека.

Отбор: SelectKBest по F-критерию, k = 8, чтобы наборы сравнивались в равном
размере. Два режима:
  honest - отбор внутри обучающей части каждого фолда, человек под проверкой
           на выбор признаков не влияет;
  leaky  - отбор один раз по всем 88 людям, дальше тот же LOSO. Так делают
           по умолчанию, и так оценка завышается.

Протокол тот же, что в 21 и 24: LOSO, обучение и проверка на ГЗ, решение
принимается по человеку. ГО здесь не трогается: wPLI зависит от числа окон
(см. 14), у ГЗ и ГО оно разное, и перенос между условиями мерил бы не
признаки, а длину записи.

Выход: reports/feature_space.csv, reports/feature_space_paired.csv,
       reports/feature_space_selected.csv
Запуск: python src/27_feature_space.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FEATURES_CSV = "data/features_with_conn.csv"
OUT = "reports/feature_space.csv"
OUT_PAIRED = "reports/feature_space_paired.csv"
OUT_SEL = "reports/feature_space_selected.csv"
COND = "EC"
K = 8
SEED = 42
N_BOOT = 2000
N_PERM = 1000

COMPACT = ["rel_delta_mean", "rel_theta_mean", "rel_alpha_mean", "rel_beta_mean",
           "rel_alpha_occipital", "alpha_theta_ratio", "spectral_entropy", "peak_alpha"]
WPLI = ["wpli_alpha_OF", "wpli_alpha_OO", "wpli_alpha_global",
        "wpli_theta_OF", "wpli_theta_OO", "wpli_theta_global"]


def model(k=None):
    steps = [StandardScaler()]
    if k is not None:
        steps.append(SelectKBest(f_classif, k=k))
    steps.append(LogisticRegression(C=1.0, max_iter=5000, class_weight="balanced"))
    return make_pipeline(*steps)


def label(df, task):
    if task == "3class":
        return df["diagnosis"]
    if task == "AD_vs_HC":
        return df["diagnosis"].where(df["diagnosis"].isin(["AD", "HC"]))
    if task == "PAT_vs_HC":
        return pd.Series(np.where(df["diagnosis"] == "HC", "HC", "PAT"), index=df.index)
    raise ValueError(task)


def boot_ci(y_true, y_pred, n=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    s = []
    for _ in range(n):
        i = rng.integers(0, len(y_true), len(y_true))
        if len(set(y_true[i])) < 2:
            continue
        s.append(balanced_accuracy_score(y_true[i], y_pred[i]))
    return (np.percentile(s, 2.5), np.percentile(s, 97.5)) if s else (np.nan, np.nan)


def perm_p(y_true, y_pred, n=N_PERM, seed=SEED):
    rng = np.random.default_rng(seed)
    obs = balanced_accuracy_score(y_true, y_pred)
    y = np.asarray(y_true)
    cnt = sum(balanced_accuracy_score(rng.permutation(y), y_pred) >= obs for _ in range(n))
    return (cnt + 1) / (n + 1)


def mcnemar(a, b):
    """Точный тест по людям, у которых исходы разошлись."""
    n01 = sum(1 for x, y in zip(a, b) if x == 1 and y == 0)
    n10 = sum(1 for x, y in zip(a, b) if x == 0 and y == 1)
    if n01 + n10 == 0:
        return n01, n10, 1.0
    return n01, n10, min(1.0, binomtest(min(n01, n10), n01 + n10, 0.5).pvalue)


def loso(df, feats, task, mode, picked=None):
    """Возвращает (ids, y_true, y_pred). mode: none | honest | leaky.

    В picked, если он передан, складываются имена признаков, выбранные
    в каждом фолде, - по ним потом видно, насколько отбор устойчив."""
    d = df.copy()
    d["y"] = label(d, task)
    d = d.dropna(subset=["y"]).reset_index(drop=True)
    k = None
    if mode == "honest":
        k = min(K, len(feats))
    elif mode == "leaky":
        # отбор по всем людям сразу, включая того, кого потом проверяем
        sel = SelectKBest(f_classif, k=min(K, len(feats)))
        sel.fit(StandardScaler().fit_transform(d[feats]), d["y"])
        feats = [f for f, keep in zip(feats, sel.get_support()) if keep]
    ids, yt, yp = [], [], []
    for s in d.participant_id:
        tr = d[d.participant_id != s]
        te = d[d.participant_id == s]
        m = model(k).fit(tr[feats], tr["y"])
        if picked is not None and k is not None:
            sup = m.named_steps["selectkbest"].get_support()
            picked.append([f for f, keep in zip(feats, sup) if keep])
        ids.append(s)
        yt.append(te["y"].iloc[0])
        yp.append(m.predict(te[feats])[0])
    return ids, yt, yp


def main():
    df = pd.read_csv(FEATURES_CSV)
    df = df[df.condition == COND].copy()
    wide = [c for c in df.columns if c.startswith("rel_")] + \
           [c for c in ("alpha_theta_ratio", "theta_beta_ratio", "spectral_entropy",
                        "peak_alpha") if c in df.columns]
    for c in wide + WPLI:
        df[c] = df[c].fillna(df[c].median())

    spaces = {
        "spec8": COMPACT,
        "conn6": WPLI,
        "spec8+conn": COMPACT + WPLI,
        "spec22": wide,
        "spec22+conn": wide + WPLI,
    }
    print(f"условие {COND}, людей {len(df)}, k отбора {K}")

    rows, preds, picked_all = [], {}, []
    for name, feats in spaces.items():
        modes = ["none"] if len(feats) <= K else ["none", "honest", "leaky"]
        for mode in modes:
            for task in ("AD_vs_HC", "PAT_vs_HC", "3class"):
                picked = [] if mode == "honest" else None
                ids, yt, yp = loso(df, feats, task, mode, picked)
                if picked:
                    n_folds = len(picked)
                    cnt = {}
                    for fold in picked:
                        for f in fold:
                            cnt[f] = cnt.get(f, 0) + 1
                    for f, c in sorted(cnt.items(), key=lambda x: -x[1]):
                        picked_all.append(dict(space=name, task=task, feature=f,
                                               folds=c, of=n_folds,
                                               share=round(c / n_folds, 3)))
                preds[(name, mode, task)] = (ids, yt, yp)
                lo, hi = boot_ci(yt, yp)
                rows.append(dict(space=name, n_feats=len(feats), selection=mode,
                                 task=task, n=len(yt),
                                 BA=balanced_accuracy_score(yt, yp),
                                 F1=f1_score(yt, yp, average="macro", zero_division=0),
                                 ci_lo=lo, ci_hi=hi, p_perm=perm_p(yt, yp),
                                 chance={"3class": 1 / 3}.get(task, 0.5)))
                print(f"  {name:12s} {mode:6s} {task:9s} BA={rows[-1]['BA']:.3f}")

    out = pd.DataFrame(rows).round(3)
    Path("reports").mkdir(exist_ok=True)
    out.to_csv(OUT, index=False)

    # парные сравнения: те же люди, те же фолды, разные наборы признаков
    pairs = []
    for task in ("AD_vs_HC", "PAT_vs_HC", "3class"):
        cand = [("spec8", "none", "spec8+conn", "none"),
                ("spec8", "none", "spec22", "none"),
                ("spec8", "none", "spec22+conn", "honest"),
                ("spec8+conn", "none", "spec8+conn", "honest"),
                ("spec22", "none", "spec22", "honest"),
                ("spec22", "honest", "spec22", "leaky"),
                ("spec22+conn", "honest", "spec22+conn", "leaky"),
                ("spec8+conn", "honest", "spec8+conn", "leaky")]
        for s1, m1, s2, m2 in cand:
            a, b = preds.get((s1, m1, task)), preds.get((s2, m2, task))
            if a is None or b is None:
                continue
            common = [i for i in a[0] if i in set(b[0])]
            ia = {s: j for j, s in enumerate(a[0])}
            ib = {s: j for j, s in enumerate(b[0])}
            ok_a = [int(a[1][ia[s]] == a[2][ia[s]]) for s in common]
            ok_b = [int(b[1][ib[s]] == b[2][ib[s]]) for s in common]
            ba_a = balanced_accuracy_score([a[1][ia[s]] for s in common],
                                           [a[2][ia[s]] for s in common])
            ba_b = balanced_accuracy_score([b[1][ib[s]] for s in common],
                                           [b[2][ib[s]] for s in common])
            n01, n10, p = mcnemar(ok_a, ok_b)
            pairs.append(dict(task=task, n=len(common),
                              a=f"{s1}/{m1}", b=f"{s2}/{m2}",
                              BA_a=round(ba_a, 3), BA_b=round(ba_b, 3),
                              delta=round(ba_b - ba_a, 3),
                              only_a=n01, only_b=n10, p_mcnemar=round(p, 4)))
    pr = pd.DataFrame(pairs)
    pr.to_csv(OUT_PAIRED, index=False)
    sel = pd.DataFrame(picked_all)
    sel.to_csv(OUT_SEL, index=False)

    print("\n" + out.to_string(index=False))
    print("\n" + pr.to_string(index=False))
    best = sel[(sel.space == "spec22") & (sel.task == "AD_vs_HC")]
    print("\nчто отбирается на spec22 / AD_vs_HC (доля фолдов):")
    print(best[["feature", "folds", "of", "share"]].to_string(index=False))
    print(f"\nсохранено {OUT}, {OUT_PAIRED}, {OUT_SEL}")
    print("delta - прибавка варианта b к варианту a на общих людях;")
    print("leaky против honest показывает, сколько прибавки даёт сам по себе")
    print("отбор признаков на всей выборке.")


if __name__ == "__main__":
    main()
