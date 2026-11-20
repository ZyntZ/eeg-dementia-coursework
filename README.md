# ЭЭГ-деменция — курсовая

Базовый пайплайн и препроцессинг для 3-классовой классификации ЭЭГ (AD / FTD / HC) с оценкой без утечки. Сравнение протоколов EC vs EO и дообучение модели LEAD

## Данные

- OpenNeuro `ds004504` (ADFTD-RS, 200 Гц) и `ds006036` (ADFTD-PS, 500 Гц): 88 испытуемых с парными записями (36 AD, 23 FTD, 29 HC; у sub-011 нет EO). Всего 175 записей.
- Спектральные признаки по Welch (окно 4 с): относительные delta/theta/alpha/beta, low/high alpha, отношение alpha/theta, энтропия. Файл `data/features_spectral.csv`

## Пайплайн

```
src/01_check_overlap_real.py  — проверка пересечения 88 ID
src/04_extract_features.py    — признаки Welch (на Kaggle пропускается, если CSV есть)
src/05_train_eval_real.py     — 5-fold LogReg (L2 C=1), обучение на EC, тест EC_new / EO_new / EO_familiar
src/11_stats.py               — парные t-тесты EC => EO (alpha occipital, alpha/theta)
src/18_compare_protocols.py   — сводные метрики
src/08_lead_preprocessing.py  — сегментация LEAD (200/100/50 Гц, T=400, C=19)
src/09_lead_merge.py          — RS + PS  =>  ADFTD (171387 сегментов)
src/10_lead_finetune.py       — шаблон дообучения LEAD
```

Запуск локально (без raw, если есть `features_spectral.csv`):

```bash
pip install -r requirements.txt
python src/run_pipeline.py
# по шагам: 05  =>  11  =>  18  =>  19  =>  20
```

На Kaggle используется датасет `asoksci/eeg-dementia-coursach-full` (src + data + reports + figures в tar). `run_pipeline.py` пропускает `04_extract_features.py`, если есть `features_spectral.csv`

## Результаты

### 5-fold (LogReg, среднее по 5 фолдам)

| тест | n | BA | F1 |
|------|---|----|----|
| EC_new | 17 | 0.378 | 0.367 |
| EO_new | 17 | 0.332 | 0.251 |
| EO_familiar | 69 | 0.368 | 0.338 |

MLP (32×32, dropout 0.3): EC 0.453 / EO 0.422

### LOSO

| модель | EC BA/F1 | EO BA/F1 | n |
|-------|----------|----------|---|
| LogReg | 0.303 / 0.305 | 0.394 / 0.350 | 88 / 87 |
| MLP | 0.436 / 0.373 | 0.414 / 0.327 | 88 / 87 |

Падение EC => EO: +0.046 (5-fold) / -0.091 (LOSO) — в пределах разброса. EO_familiar ≈ EC, утечки сверх ожидаемого нет.

### Парные EC => EO

- `rel_alpha_occipital` HC: 0.0516  =>  0.0331, p=0.059
- `alpha_theta_ratio` HC: 0.740  =>  0.484, p=0.013 (у AD p=0.127 — нз)
- `rel_theta_mean`: без изменений (p>0.84)

### LEAD

- Препроцесинг: RS 123155 (3.5G) + PS 48232 (1.4G) = ADFTD 171387 (4.9G), T=400 C=19, монтаж standard_1020
- Дообучение с чекпоинтом `YiheWang/LEAD` `P-11-b1024-p50-g16-learnable-3D` (mccv, 2 эпохи, batch 64):
  - по сегментам: Acc 82.17% F1 80.32% AUROC 94.60%
  - по испытуемым (голосование): Acc 90.00% F1 89.63% AUROC 100%

## Структура репозитория

```
src/            # 01–20 + run_pipeline.py
configs/        # pipeline.yaml, split_stratified.json
data/           # features_spectral.csv, features_with_conn.csv
reports/        # метрики, compare_protocols, статистика, loso, lead_mccv.log
figures/        # 12 PNG: матрицы ошибок, столбцы, гистограммы, боксплоты
notebooks/      # ядра Kaggle (run, lead, lead3, lead4)
Processed/      # меммапы LEAD (не в репо, на Kaggle)
LEAD/           # клон модели (не в репо)
```

## Kaggle

- Датасет: `asoksci/eeg-dementia-coursach-full` (ready)
- Ядра: `eeg-dementia-run` (пайплайн), `eeg-dementia-lead` (PS 908M), `eeg-dementia-lead3` (ADFTD 4.9G), `eeg-dementia-lead4` (mccv 90%)

## Примечания

- Все разбиения — по испытуемым, без пересеченй. Обучение только на EC, EO — невидимый. EO_familiar — записи тех же людей из train, для оценки утечки
- LOSO n=88 EC / 87 EO из-за отсутствия EO у sub-011
