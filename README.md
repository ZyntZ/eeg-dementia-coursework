# EEG dementia — EC (ds004504) → EO+photo (ds006036)

88 subjects (36 AD, 23 FTD, 29 HC), 500 Hz, 19 ch 10-20. Same subjects in both conditions.

## Задача
Проверить сохранение классификации при переходе EC→EO и завышение оценки при тесте на другой записи знакомого человека vs нового. Фокус AD/HC, FTD после обучения на 3 классах. План согласован с Лазуренко Д.М. (АБиМ).

## Данные
- ds004504 eyesclosed (RS) — 5.4GB
- ds006036 photomark (PS) — 2.2GB, интервалы open/closed eyes по events.tsv, sub-011 без разметки
- Фиксация версий: ds004504 v1.0.8, ds006036 v1.0.6

## Протокол без утечки
- Деление по людям, обе записи тестового вне обучения
- Train только на EC обучающих
- Test: EC_new, EO_new (парно те же новые люди), EO_familiar (EO обучающих)
- Метрики по людям, 5-fold + LOSO (88), bootstrap CI, paired t/wilcoxon

## Структура
```
src/01_check_overlap_real.py   сверка ID
src/04_extract_features.py     Welch 4s 0.5-45Hz -> data/features_spectral.csv (175 rows)
src/05_train_eval_real.py      LogReg 3-class 5-fold
src/08_lead_preprocessing.py   LEAD seg 400 step 200 -> Processed/L400/ADFTD-*
src/11_stats.py, 12_plots.py   p-значения, графики
src/13_baseline_nn.py, 17_mlp_loso.py
src/15_loso_eval.py            LOSO logreg
src/19_loso_plots.py, 20_export_tables.py
configs/pipeline.yaml
data/features_spectral.csv
reports/*.csv, *.tex
figures/*.png
```

## Результаты baseline (спектральные)
- HC alpha occ EC 0.0516 -> EO 0.0331 p=0.059, AD 0.0340->0.0287 p=0.62, FTD 0.0292->0.0248 p=0.41
- alpha/theta HC 0.74->0.48 p=0.013
- 5-fold LogReg 3-class: EC_new BA 0.378 EO_new 0.332 EO_fam 0.368
- LOSO LogReg: EC 0.303 EO 0.394 PATvsHC 0.632/0.552
- 5-fold MLP: EC 0.453 EO 0.422; LOSO MLP EC 0.436 EO 0.414
- Вывод: простая спектральная модель тривиальна для 3 классов, EC->EO падение только у HC, нужен LEAD

## LEAD
2-й этап: дообучение на 3 классах. Код в src/08-10,16. Требует Processed/L400 + чекпоинт https://huggingface.co/YiheWang/LEAD
```
python src/08_lead_preprocessing.py data/raw/ds004504 Processed/L400/ADFTD-RS
python src/08_lead_preprocessing.py data/raw/ds006036 Processed/L400/ADFTD-PS
python src/09_lead_merge.py Processed/L400/ADFTD-RS Processed/L400/ADFTD-PS Processed/L400/ADFTD
python LEAD/run.py --task_name finetune --model LEAD --data ADFTD --root_path ./Processed --data_path L400 --cross_val loso
```

## Воспроизведение
```
pip install -r requirements.txt
python src/04_extract_features.py
python src/run_pipeline.py
```

## Архивы
Каждый коммит — отдельный архив в archives/ только с новыми/обновленными файлами:
v01 baseline, v02 LEAD prep, v03 eval+MLP, v04 LOSO+conn, v05 MLP LOSO+compare, v06 plots+tables

## Ограничения
Фиксированный порядок сессий (EC всегда первый), в EO есть фотостимуляция — разницу нельзя приписывать только глазам.
