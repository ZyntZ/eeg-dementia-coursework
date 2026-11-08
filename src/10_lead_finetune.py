"""
10_lead_finetune.py — шаблон дообучения LEAD на 3 класса (AD/FTD/HC)
Не запускать без весов: требует скачивания checkpoint с HuggingFace YiheWang/LEAD
и сырые Processed/L400/ADFTD-RS, ADFTD-PS, ADFTD (создаются src/08,09)
Протокол как в курсовой: train на EC train-subjects, test EC_new/EO_new/EO_familiar
"""
# Пример вызова как в LEAD/run.py:
# python LEAD/run.py --task_name finetune --model LEAD --data ADFTD --root_path ./Processed --data_path L400 --features M --seq_len 400 --enc_in 19 --label_len 0 --pred_len 0 --e_layers 2 --d_layers 1 --n_heads 4 --d_model 128 --d_ff 256 --dropout 0.1 --batch_size 32 --train_epochs 20 --cross_val loso --learning_rate 1e-4 --use_gpu True
# Для курсовой нужен loso или 5-fold с делением по людям без утечки второй записи того же человека

# Заглушка: показывает как подключить наши Processed данные к LEAD
import json, pathlib
print("LEAD finetune template ready")
print("Checkpoint: https://huggingface.co/YiheWang/LEAD")
print("Expected dirs: Processed/L400/ADFTD-RS, ADFTD-PS, ADFTD")
for p in ["Processed/L400/ADFTD-RS/meta.json","Processed/L400/ADFTD-PS/meta.json","Processed/L400/ADFTD/meta.json"]:
    print(p, pathlib.Path(p).exists())
