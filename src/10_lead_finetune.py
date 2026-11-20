"""
10_lead_finetune.py — шаблон дообучения LEAD на 3 класса (AD/FTD/HC)
Не запускать без весов: требует скачивания checkpoint с HuggingFace YiheWang/LEAD
и сырые Processed/L400/ADFTD-RS, ADFTD-PS, ADFTD (создаются src/08,09)
Протокол как в курсовой: train на EC train-subjects, test EC_new/EO_new/EO_familiar
"""
# Пример вызова как в LEAD/run.py:
# python LEAD/run.py --task_name finetune --model LEAD --data ADFTD --root_path ./Processed --data_path L400 --features M --seq_len 400 --enc_in 19 --label_len 0 --pred_len 0 --e_layers 2 --d_layers 1 --n_heads 4 --d_model 128 --d_ff 256 --dropout 0.1 --batch_size 32 --train_epochs 20 --cross_val loso --learning_rate 1e-4 --use_gpu True
# Для курсовой нужен loso или 5-fold с делением по людям без утечки второй записи того же человека

# ШАБЛОН: реальный finetune — через LEAD/run.py (см. kaggle_logs/lead_mccv.log)
# Важно: LEAD ожидает root_path + dataset_name, данные только для mccv/loso/fixed (не 5-fold в BaseLoader)
import json, pathlib, warnings
print("LEAD finetune template — не запускать напрямую")
print("Checkpoint: https://huggingface.co/YiheWang/LEAD  -> checkpoints/LEAD/pretrain/LEAD/P-11-b1024-p50-g16-learnable-3D/nh8_el12_dm128_df256_seed41/checkpoint.pth")
print("Ожидаемые папки: Processed/L400/ADFTD [+ дубль Processed/ADFTD для совместимости]")
for p in ["Processed/L400/ADFTD/meta.json","Processed/ADFTD/meta.json"]:
    print(p, pathlib.Path(p).exists())
warnings.warn("STUB: используйте LEAD/run.py --data MultiDatasets --training_dataset ADFTD --cross_val mccv --seq_len 400 (см. README Limitations)")
