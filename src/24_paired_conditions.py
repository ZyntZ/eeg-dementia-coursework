"""
24_paired_conditions.py - сравнение условий на одних и тех же людях.

Зачем отдельный скрипт. В 21 каждое условие оценивается на всех, у кого оно
есть: ГЗ по 65 людям, ГО по 52, ГО без вспышек по 35. Разница между строками
таблицы тогда складывается из двух вещей сразу - из условия и из того, что
это разные люди. В плане записано иначе: сравнить результат у одних и тех же
новых людей в двух условиях.

Здесь для каждой пары условий берутся только те, у кого есть обе записи,
и обе точности считаются по этому общему списку. Разница проверяется точным
тестом Макнемара по парам "правильно / неправильно" у одного человека.

Протокол обучения тот же, что в 21: LOSO, модель учится на ГЗ остальных
людей, тестовый человек целиком вне обучения.

Запуск: python src/24_paired_conditions.py
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import load_features

OUT = "reports/paired_conditions.csv"
TRAIN_COND = "EC"
SEED = 42

COMPACT = ["rel_delta_mean", "rel_theta_mean", "rel_alpha_mean", "rel_beta_mean",
           "rel_alpha_occipital", "alpha_theta_ratio", "spectral_entropy", "peak_alpha"]

# Пары, которые имеют смысл для вопроса работы. EC_match нужен там, где
# сравнение идёт по длительности: полная ГЗ это 800 секунд против 20-25.
PAIRS = [("EC", "EO"), ("EC", "EO_photic"), ("EC", "EO_broad"),
         ("EC", "EO_nophotic"), ("EC_match", "EO"), ("EC_match", "EO_photic"),
         ("EC", "EC_match")]

TASKS = ("AD_vs_HC", "PAT_vs_HC", "3class")


def model():
    return make_pipeline(StandardScaler(),
                         LogisticRegression(C=1.0, max_iter=5000,
                                            class_weight="balanced"))


def label(df, task):
    if task == "3class":
        return df["diagnosis"]
    if task == "AD_vs_HC":
        return df["diagnosis"].where(df["diagnosis"].isin(["AD", "HC"]))
    if task == "PAT_vs_HC":
        return pd.Series(np.where(df["diagnosis"] == "HC", "HC", "PAT"),
                         index=df.index)
    raise ValueError(task)


def loso_preds(df, feats, task, conds):
    """Предсказание для каждого человека в каждом условии.

    Возвращает {условие: {человек: предсказанный класс}} и истинные метки.
    """
    tr_all = df[df.condition == TRAIN_COND].copy()
    tr_all["y"] = label(tr_all, task)
    tr_all = tr_all.dropna(subset=["y"])
    pred = {c: {} for c in conds}
    truth = {}
    for s in tr_all.participant_id.tolist():
        tr = tr_all[tr_all.participant_id != s]
        m = model().fit(tr[feats], tr["y"])
        for c in conds:
            te = df[(df.participant_id == s) & (df.condition == c)].copy()
            te["y"] = label(te, task)
            te = te.dropna(subset=["y"])
            if te.empty:
                continue
            pred[c][s] = m.predict(te[feats])[0]
            truth[s] = te["y"].iloc[0]
    return pred, truth


def mcnemar(a, b):
    """Точный тест по людям, у которых исходы разошлись."""
    n01 = sum(1 for x, y in zip(a, b) if x == 1 and y == 0)
    n10 = sum(1 for x, y in zip(a, b) if x == 0 and y == 1)
    if n01 + n10 == 0:
        return n01, n10, 1.0
    p = binomtest(min(n01, n10), n01 + n10, 0.5).pvalue
    return n01, n10, min(1.0, p)


def main():
    df = load_features()
    wide = [c for c in df.columns if c.startswith("rel_")] + \
           [c for c in ("alpha_theta_ratio", "theta_beta_ratio",
                        "spectral_entropy", "peak_alpha") if c in df.columns]
    sets = {"compact": [c for c in COMPACT if c in df.columns], "all": wide}
    conds = sorted({c for p in PAIRS for c in p})

    rows = []
    for fs_name, feats in sets.items():
        for task in TASKS:
            pred, truth = loso_preds(df, feats, task, conds)
            for ca, cb in PAIRS:
                common = sorted(set(pred[ca]) & set(pred[cb]))
                if len(common) < 10:
                    continue
                ya = [truth[s] for s in common]
                pred_a = [pred[ca][s] for s in common]
                pred_b = [pred[cb][s] for s in common]
                ba_a = balanced_accuracy_score(ya, pred_a)
                ba_b = balanced_accuracy_score(ya, pred_b)
                pa = [int(p == y) for p, y in zip(pred_a, ya)]
                pb = [int(p == y) for p, y in zip(pred_b, ya)]
                n01, n10, p = mcnemar(pa, pb)
                rows.append(dict(features=fs_name, task=task, metric="BA",
                                 cond_a=ca, cond_b=cb, n=len(common),
                                 score_a=round(ba_a, 3), score_b=round(ba_b, 3),
                                 delta=round(ba_b - ba_a, 3),
                                 only_a_correct=n01, only_b_correct=n10,
                                 p_mcnemar=round(p, 4)))

    out = pd.DataFrame(rows)
    Path("reports").mkdir(exist_ok=True)
    out.to_csv(OUT, index=False)

    print("Сравнение условий на общих людях, обучение всегда на ГЗ (LOSO)\n")
    for fs_name in sets:
        for task in TASKS:
            sub = out[(out.features == fs_name) & (out.task == task)]
            if sub.empty:
                continue
            print(f"--- {task}, признаки {fs_name}")
            print(sub[["cond_a", "cond_b", "n", "score_a", "score_b", "delta",
                       "only_a_correct", "only_b_correct",
                       "p_mcnemar"]].to_string(index=False))
            print()
    print("сохранено:", OUT)


if __name__ == "__main__":
    main()
