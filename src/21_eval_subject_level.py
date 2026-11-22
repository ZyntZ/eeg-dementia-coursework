"""
21_eval_subject_level.py - оценка по людям, как записано в плане курсовой.

Чего не хватало раньше:
  - в плане в центре внимания болезнь Альцгеймера и контроль, но бинарная
    задача AD vs HC нигде не сохранялась, только печаталась в 05 и 07;
  - руководитель просил посмотреть пациентов против контроля; в 15 это
    получалось схлопыванием предсказаний трёхклассовой модели, а это не то же
    самое, что обучить бинарную модель;
  - руководитель отдельно писал, что по совокупности признаков ГЗ и ГО
    пациенты и контроль должны разделяться хорошо. Модели на объединённом
    векторе ГЗ+ГО не было вообще;
  - не было доверительных интервалов и проверки, отличается ли результат от
    случайного угадывания.

Протокол: обучаемся только на ГЗ (EC) обучающих людей. Проверяем на
  new      - человек, которого модель не видела (LOSO),
  familiar - другая запись человека, чьё ГЗ было в обучении (5 фолдов).

Запуск: python src/21_eval_subject_level.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FEATURES_CSV = "data/features_spectral.csv"
OUT = "reports/eval_subject_level.csv"
TRAIN_COND = "EC"
SEED = 42
N_BOOT = 2000
N_PERM = 1000


def model():
    return make_pipeline(StandardScaler(),
                         LogisticRegression(C=1.0, max_iter=5000, class_weight="balanced"))


# Два набора признаков. "compact" - восемь штук, так было в 15_loso_eval.py с
# самого начала. "all" - всё, что посчитал 04. На 88 людях широкий набор
# переобучается: на три класса LOSO даёт 0.462 против 0.557 у компактного,
# на AD vs HC 0.730 против 0.764. Headline-числа берём по compact, широкий
# набор остаётся в таблице, чтобы разницу было видно.
COMPACT = ["rel_delta_mean", "rel_theta_mean", "rel_alpha_mean", "rel_beta_mean",
           "rel_alpha_occipital", "alpha_theta_ratio", "spectral_entropy", "peak_alpha"]


def load():
    df = pd.read_csv(FEATURES_CSV)
    wide = [c for c in df.columns if c.startswith("rel_")] + \
           [c for c in ("alpha_theta_ratio", "theta_beta_ratio", "spectral_entropy",
                        "peak_alpha") if c in df.columns]
    # пик альфы может не найтись, заполняем медианой по условию и помечаем
    for c in wide:
        df[c] = df.groupby("condition")[c].transform(lambda s: s.fillna(s.median()))
    sets = {"compact": [c for c in COMPACT if c in df.columns], "all": wide}
    return df, sets


def label(df, task):
    if task == "3class":
        return df["diagnosis"]
    if task == "AD_vs_HC":
        return df["diagnosis"].where(df["diagnosis"].isin(["AD", "HC"]))
    if task == "PAT_vs_HC":
        return np.where(df["diagnosis"] == "HC", "HC", "PAT")
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
    """Насколько часто перемешанные метки дают такую же точность."""
    rng = np.random.default_rng(seed)
    obs = balanced_accuracy_score(y_true, y_pred)
    y = np.asarray(y_true)
    cnt = sum(balanced_accuracy_score(rng.permutation(y), y_pred) >= obs for _ in range(n))
    return (cnt + 1) / (n + 1)


def run_loso(df, feats, task, train_cond, test_conds):
    """Обучение на train_cond остальных людей, проверка на новом человеке."""
    tr_all = df[df.condition == train_cond].copy()
    tr_all["y"] = label(tr_all, task)
    tr_all = tr_all.dropna(subset=["y"])
    subs = tr_all.participant_id.tolist()
    preds = {c: ([], []) for c in test_conds}
    for s in subs:
        tr = tr_all[tr_all.participant_id != s]
        m = model().fit(tr[feats], tr["y"])
        for c in test_conds:
            te = df[(df.participant_id == s) & (df.condition == c)].copy()
            te["y"] = label(te, task)
            te = te.dropna(subset=["y"])
            if te.empty:
                continue
            preds[c][0].append(te["y"].iloc[0])
            preds[c][1].append(m.predict(te[feats])[0])
    rows = []
    for c, (yt, yp) in preds.items():
        if len(yt) < 5:
            continue
        lo, hi = boot_ci(yt, yp)
        rows.append(dict(task=task, protocol="loso", train=train_cond, test=c,
                         group="new", n=len(yt),
                         BA=balanced_accuracy_score(yt, yp),
                         F1=f1_score(yt, yp, average="macro", zero_division=0),
                         ci_lo=lo, ci_hi=hi, p_perm=perm_p(yt, yp)))
    return rows


def run_familiar(df, feats, task, train_cond, test_conds):
    """5 фолдов по людям. Проверяем ещё и вторую запись обучающих людей -
    это и есть проверка на знакомых."""
    tr_all = df[df.condition == train_cond].copy()
    tr_all["y"] = label(tr_all, task)
    tr_all = tr_all.dropna(subset=["y"]).reset_index(drop=True)
    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    acc = {(c, g): ([], []) for c in test_conds for g in ("new", "familiar")}
    for tr_i, te_i in skf.split(tr_all, tr_all["y"]):
        tr = tr_all.iloc[tr_i]
        m = model().fit(tr[feats], tr["y"])
        groups = {"new": set(tr_all.iloc[te_i].participant_id),
                  "familiar": set(tr.participant_id)}
        for c in test_conds:
            for g, ids in groups.items():
                te = df[(df.condition == c) & (df.participant_id.isin(ids))].copy()
                te["y"] = label(te, task)
                te = te.dropna(subset=["y"])
                if te.empty:
                    continue
                acc[(c, g)][0].extend(te["y"])
                acc[(c, g)][1].extend(m.predict(te[feats]))
    rows = []
    for (c, g), (yt, yp) in acc.items():
        if len(yt) < 5:
            continue
        if g == "familiar" and c == train_cond:
            continue          # это просто обучающая выборка, смысла нет
        lo, hi = boot_ci(yt, yp)
        rows.append(dict(task=task, protocol="5fold", train=train_cond, test=c,
                         group=g, n=len(yt),
                         BA=balanced_accuracy_score(yt, yp),
                         F1=f1_score(yt, yp, average="macro", zero_division=0),
                         ci_lo=lo, ci_hi=hi, p_perm=perm_p(yt, yp)))
    return rows


def run_combined(df, feats, task, c1, c2):
    """Один вектор из признаков двух условий одного человека - то, что
    предлагал руководитель ("по совокупности признаков ГЗ и ГО")."""
    a = df[df.condition == c1].set_index("participant_id")
    b = df[df.condition == c2].set_index("participant_id")
    ids = a.index.intersection(b.index)
    if len(ids) < 20:
        return []
    X = pd.concat([a.loc[ids, feats], b.loc[ids, feats].add_suffix("__2")], axis=1)
    y = pd.Series(label(a.loc[ids], task), index=ids).dropna()
    X = X.loc[y.index]
    yt, yp = [], []
    for s in y.index:
        m = model().fit(X.drop(index=s), y.drop(index=s))
        yt.append(y[s])
        yp.append(m.predict(X.loc[[s]])[0])
    lo, hi = boot_ci(yt, yp)
    return [dict(task=task, protocol="loso", train=f"{c1}+{c2}", test=f"{c1}+{c2}",
                 group="new", n=len(yt),
                 BA=balanced_accuracy_score(yt, yp),
                 F1=f1_score(yt, yp, average="macro", zero_division=0),
                 ci_lo=lo, ci_hi=hi, p_perm=perm_p(yt, yp))]


def main():
    df, sets = load()
    conds = [c for c in df.condition.unique()]
    eo_conds = [c for c in conds if c.startswith("EO")]
    test_conds = [TRAIN_COND] + eo_conds + [c for c in conds if c == "EC_match"]
    print("условия в таблице:", conds)

    rows = []
    for fs_name, feats in sets.items():
        print(f"\n===== набор признаков: {fs_name} ({len(feats)} шт.)")
        for task in ("AD_vs_HC", "PAT_vs_HC", "3class"):
            print(f"\n--- {task}")
            new = []
            new += run_loso(df, feats, task, TRAIN_COND, test_conds)
            new += run_familiar(df, feats, task, TRAIN_COND, test_conds)
            for eo in eo_conds:
                new += run_combined(df, feats, task, TRAIN_COND, eo)
            for r in new:
                r["features"] = fs_name
            rows += new

    out = pd.DataFrame(rows)
    out = out[["features"] + [c for c in out.columns if c != "features"]]
    chance = {"3class": 1 / 3, "AD_vs_HC": 0.5, "PAT_vs_HC": 0.5}
    out["chance"] = out.task.map(chance)
    out = out.round(3)
    Path("reports").mkdir(exist_ok=True)
    out.to_csv(OUT, index=False)
    print("\n" + out.to_string(index=False))
    print(f"\nсохранено {OUT}")
    print("p_perm - доля перестановок меток, которые дают не хуже;")
    print("если он больше 0.05, результат от угадывания не отличается.")


if __name__ == "__main__":
    main()
