# EEG dementia — EC (ds004504) → EO+photo (ds006036)

88 subjects (36 AD, 23 FTD, 29 HC), 500 Hz, 19 ch 10-20. Same subjects in both conditions.

## Задача
Проверить сохранение классификации при переходе EC→EO и завышение оценки при тесте на другой записи знакомого человека vs нового. Фокус AD/HC, FTD после обучения на 3 классах. План согласован с Лазуренко Д.М. (АБиМ).

## Данные
- ds004504 eyesclosed (RS) — 5.4GB
- ds006036 photomark (PS) — 2.2GB, интервалы open/closed eyes по events.tsv, sub-011 без разметки
- Версии: ds004504 v1.0.8, ds006036 v1.0.6

## Протокол без утечки
- Деление по людям, обе записи тестового вне обучения
- Train только на EC обучающих
- Test: EC_new, EO_new (парно те же новые люди), EO_familiar (EO обучающих)
- Метрики по людям, 5-fold + LOSO (88), bootstrap CI, paired t/wilcoxon

## Как запустить
```
pip install -r requirements.txt
python src/04_extract_features.py
python src/run_pipeline.py
```

## LEAD (2-й этап)
Требует Processed/L400 + чекпоинт https://huggingface.co/YiheWang/LEAD
```
python src/08_lead_preprocessing.py data/raw/ds004504 Processed/L400/ADFTD-RS
python src/08_lead_preprocessing.py data/raw/ds006036 Processed/L400/ADFTD-PS
python src/09_lead_merge.py Processed/L400/ADFTD-RS Processed/L400/ADFTD-PS Processed/L400/ADFTD
python LEAD/run.py --task_name finetune --model LEAD --data ADFTD --root_path ./Processed --data_path L400 --cross_val loso
```

## Графики
- `figures/box_alpha_occipital.png` — alpha occ EC/EO по диагнозам
- `figures/scatter_ec_eo_alpha.png` — парные точки EC vs EO
- `figures/bar_ba_3class.png` — BA 3-class EC_new/EO_new/EO_familiar
- `figures/loso_logreg_EC_cm.png` — LOSO матрицы
- `figures/compare_protocols_bar.png` — 5-fold vs LOSO

## Результаты baseline
- HC alpha occ EC 0.0516 -> EO 0.0331 p=0.059, AD 0.0340->0.0287 p=0.62
- alpha/theta HC 0.74->0.48 p=0.013
- 5-fold LogReg 3-class: EC_new BA 0.378 EO_new 0.332
- LOSO MLP EC 0.436 EO 0.414
