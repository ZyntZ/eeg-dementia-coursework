# ЭЭГ-деменция — курсовая

Базовый пайплайн и препроцессинг для 3-классовой классификации ЭЭГ (AD / FTD / HC) с оценкой без утечки. Сравнение протоколов EC vs EO и дообучение модели LEAD.

## Данные

- OpenNeuro `ds004504` (ADFTD-RS, **500 Гц нативно**, не 200 Гц) и `ds006036` (ADFTD-PS, 500 Гц): 88 испытуемых с парными записями (36 AD, 23 FTD, 29 HC; у sub-011 нет EO). Всего 175 записей. Частота 500 Гц подтверждается сеткой `peak_alpha` 0.25 Гц (n_fft 2000).
- Спектральные признаки по Welch (окно 4 с, 0.5–45 Гц): относительные delta/theta/alpha/beta, low/high alpha, отношение alpha/theta, энтропия. Файл `data/features_spectral.csv`.

**Важно:** `duration` EO — медиана 23 с (7–80 с, 2–17 окон Welch) против 814 с у EC. Парные сравнения EC↔EO шумные — падение в пределах разброса ожидаемо.

## Пайплайн

```
src/01_check_overlap_real.py  — проверка 88 ID
src/04_extract_features.py    — Welch + average reference + notch 50 Гц, каналы по именам (не индексам)
src/05_train_eval_real.py     — 5-fold LogReg (L2 C=1, class_weight balanced), train EC → EC_new/EO_new/EO_familiar
src/11_stats.py               — парные t-тесты + FDR BH + bootstrap CI
src/18_compare_protocols.py   — сводка (n без хардкода)
src/08_lead_preprocessing.py  — сегментация LEAD 200/100/50 Гц, T=400 C=19
src/09_lead_merge.py          — RS 123155 + PS 48232 → ADFTD 171387
```

Запуск:

```bash
pip install -r requirements.txt
python src/run_pipeline.py
```

На Kaggle (`asoksci/eeg-dementia-coursach-full`) `04_extract_features.py` пропускается, если есть `features_spectral.csv`.

## Результаты

### 5-fold (LogReg)

| тест | n | BA | F1 |
|------|---|----|----|
| EC_new | 17 | 0.378 | 0.367 |
| EO_new | 17 | 0.332 | 0.251 |
| EO_familiar | 69 | 0.368 | 0.338 |

MLP (32×32, dropout 0.3, seed=42): EC 0.453 / EO 0.422

### LOSO (88 EC / 87 EO — sub-011 без EO)

| модель | EC BA/F1 | EO BA/F1 |
|-------|----------|----------|
| LogReg | 0.303 / 0.305 | 0.394 / 0.350 |
| MLP | 0.436 / 0.373 | 0.414 / 0.327 |

EC→EO: +0.046 (5-fold) / -0.091 (LOSO) — в пределах разброса, утечки нет.

### Парные EC→EO

- `rel_alpha_occipital` HC: 0.0516 → 0.0331, p=0.059 (FDR n.s.)
- `alpha_theta_ratio` HC: 0.740 → 0.484, p=0.013 (сырое; после FDR BH по 9 тестам p<0.05 не выживает — см. Limitations)
- `rel_theta_mean`: p>0.84

### LEAD

- RS 123155 (3.5G) + PS 48232 (1.4G) = **171387** (4.9G), T=400 C=19, standard_1020. В ранних коммитах указывалось 167083 — опечатка, актуально 171387.
- Чекпоинт `YiheWang/LEAD` `P-11-b1024-p50-g16-learnable-3D` (mccv, 2 эпохи, batch 64, LOSO-подобный сплит mccv):
  - по сегментам: Acc 82.17% F1 80.32% AUROC 94.60%
  - по испытуемым: Acc 90.00% F1 89.63% AUROC 100% (тест — 10 субъектов, 1 ошибка = 10%)

## Структура

```
src/            # 01–20 + run_pipeline.py
configs/        # pipeline.yaml, split_stratified.json
data/           # features_spectral.csv (176 строк с заголовком), features_with_conn.csv
reports/        # metrics, compare, stats, loso, lead_mccv.log
figures/        # 12 PNG
notebooks/      # ядра Kaggle
Processed/      # LEAD меммапы (на Kaggle)
```

## Kaggle

- Датасет: `asoksci/eeg-dementia-coursach-full`
- Ядра: `run` (пайплайн), `lead3` (ADFTD 4.9G), `lead4` (mccv)

## Ограничения — честно

1. **Спектр задавлен дельтой** (rel_delta ~90%, rel_alpha ~3% у HC в EC) — только фильтр 0.5–45 Гц + re-ref + notch, без ICA/отбраковки. В `04` добавлен `average reference` и выбор каналов по именам (`O1/O2/P3/P4/T5/T6`), но артефакты глаз/ЭМГ не чищены. Это причина chance-level бейзлайна.
2. **Затылочные каналы ранее брались по индексам [7,8,13,14] (P4/O1/T4/T5) — исправлено на поиск по `raw.ch_names`.**
3. **LEAD: 2 эпохи, один MCCV-сплит (10 тест-субъектов, ±10% за человека), фотостимуляция внутри PS не отфильтрована** (несогласованность с EC/EO частью, где фотостимуляция исключалась). Цифры реальны (логи приложены), но не LOSO по 88 и не сравнимы напрямую с EC→EO протоколом.
4. **EO короче EC в ~35 раз** — сравнение шумное, в README исходно не оговорено.
5. **`features_with_conn.csv` (wPLI) — заглушка с NaN** (`14_connectivity.py` помечен как stub). `10_lead_finetune.py` / `16_lead_loso.py` — тоже шаблоны.
6. **N-несостыковка в истории коммитов** (167083 vs 171387) — актуально 171387.
7. **Множественные сравнения без поправки** в ранних версиях — добавлен FDR BH.
8. **Kaggle-токен ранее попал в репо** (`kaggle_logs/kaggle.json`) — отозван, история требует `filter-repo`.
