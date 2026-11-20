"""
00_fetch_data.py - качает оба набора с OpenNeuro в data/raw.

Версии зафиксированы: у ds006036 между релизами менялось имя колонки в
events.tsv (в ранних было value, в 1.1.0 стало trial_type), поэтому без
фиксации версии результаты не воспроизводятся.

Качаем derivatives (ASR + ICA), participants.tsv и events.tsv. Сырые записи
из корня набора не нужны, они занимают место и считать по ним нельзя.
"""
from pathlib import Path

import openneuro

VERSIONS = {"ds004504": "1.0.9", "ds006036": "1.1.0"}

INCLUDE = {
    "ds004504": ["participants.tsv", "dataset_description.json", "derivatives/sub-*"],
    "ds006036": ["participants.tsv", "dataset_description.json",
                 "derivatives/eeglab/sub-*", "sub-*/eeg/*_events.tsv"],
}

if __name__ == "__main__":
    for ds, tag in VERSIONS.items():
        target = Path("data/raw") / ds
        target.mkdir(parents=True, exist_ok=True)
        print(f"\n=== {ds} v{tag} -> {target}")
        openneuro.download(dataset=ds, tag=tag, target_dir=target, include=INCLUDE[ds])
    print("\nготово. версии:", VERSIONS)
