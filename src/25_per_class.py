"""
25_per_class.py - разбор трёхклассовой модели по классам.

В плане сказано: в центре внимания болезнь Альцгеймера и контроль,
а лобно-височную деменцию оцениваем только после того, как модель на этом
классе обучена. Трёхклассовая модель на FTD обучена, но в таблицах до сих
пор стояла одна усреднённая цифра, по которой не видно, что происходит
с каждым классом по отдельности.

Здесь считается полнота по каждому классу и матрица ошибок, протокол тот же,
что в 21 и 24: LOSO, обучение на ГЗ остальных людей.

Выход: reports/per_class_3class.csv

Запуск: python src/25_per_class.py
"""
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, recall_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def _load_paired():
    spec = importlib.util.spec_from_file_location(
        "paired24", HERE / "24_paired_conditions.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


P = _load_paired()
OUT = Path("reports/per_class_3class.csv")
CONDS = ["EC", "EC_match", "EO", "EO_photic", "EO_nophotic", "EO_broad"]
CLASSES = ["AD", "FTD", "HC"]       # метки как в data/features_spectral.csv
RUS = {c: c for c in CLASSES}


def main():
    df = P.load_features()
    wide = [c for c in df.columns if c.startswith("rel_")] + \
           [c for c in ("alpha_theta_ratio", "theta_beta_ratio",
                        "spectral_entropy", "peak_alpha") if c in df.columns]
    sets = {"compact": [c for c in P.COMPACT if c in df.columns], "all": wide}

    rows = []
    for fs_name, feats in sets.items():
        pred, truth = P.loso_preds(df, feats, "3class", CONDS)
        for cond in CONDS:
            subs = sorted(pred[cond])
            if len(subs) < 10:
                continue
            y = [truth[s] for s in subs]
            p = [pred[cond][s] for s in subs]
            rec = recall_score(y, p, labels=CLASSES, average=None, zero_division=0)
            cm = confusion_matrix(y, p, labels=CLASSES)
            row = dict(features=fs_name, condition=cond, n=len(subs),
                       BA=round(float(np.mean(rec)), 3))
            for cls, r in zip(CLASSES, rec):
                row[f"recall_{RUS[cls]}"] = round(float(r), 3)
                row[f"n_{RUS[cls]}"] = int((np.array(y) == cls).sum())
            # куда уходят ошибки каждого класса
            for i, cls in enumerate(CLASSES):
                for j, cls2 in enumerate(CLASSES):
                    row[f"{RUS[cls]}_as_{RUS[cls2]}"] = int(cm[i, j])
            rows.append(row)

    out = pd.DataFrame(rows)
    Path("reports").mkdir(exist_ok=True)
    out.to_csv(OUT, index=False)

    cols = ["condition", "n", "BA", "recall_AD", "recall_FTD", "recall_HC"]
    for fs_name in sets:
        sub = out[out.features == fs_name]
        if sub.empty:
            continue
        print(f"--- признаки {fs_name}, полнота по классам")
        print(sub[cols].to_string(index=False))
        print()
    ec = out[(out.features == "compact") & (out.condition == "EC")]
    if len(ec):
        r = ec.iloc[0]
        print(f"ГЗ, компактный набор: FTD узнаётся в {r.recall_FTD:.0%} случаев, "
              f"из {int(r.n_FTD)} человек {int(r.FTD_as_AD)} уходят в AD, "
              f"{int(r.FTD_as_HC)} в контроль.")
    print("сохранено:", OUT)


if __name__ == "__main__":
    main()
