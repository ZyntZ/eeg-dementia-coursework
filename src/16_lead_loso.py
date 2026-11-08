import json, pathlib, subprocess, sys
# Адаптер для запуска LEAD loso finetune на наших Processed данных
# Требует: Processed/L400/ADFTD-RS, ADFTD-PS, ADFTD + checkpoint YiheWang/LEAD
# Запускает: python LEAD/run.py --task_name finetune --model LEAD --data ADFTD --root_path ./Processed --data_path L400 --cross_val loso

LEAD_CMD=[
    sys.executable, "LEAD/run.py",
    "--task_name","finetune",
    "--model","LEAD",
    "--data","ADFTD",
    "--root_path","./Processed",
    "--data_path","L400",
    "--features","M",
    "--seq_len","400",
    "--enc_in","19",
    "--cross_val","loso",
    "--train_epochs","20",
    "--batch_size","32",
    "--learning_rate","0.0001",
    "--use_gpu","True",
    "--cross_val","loso",
]
print("LEAD loso command:")
print(" ".join(LEAD_CMD))
# Проверка окружения
import pathlib
for p in ["LEAD/run.py","Processed/L400/ADFTD/meta.json","Processed/L400/ADFTD-RS/meta.json"]:
    print(p, pathlib.Path(p).exists())
# Не запускаем автоматически без GPU/чекпоинта, только показываем команду
print("Run manually when checkpoint downloaded: huggingface.co/YiheWang/LEAD")
