import pathlib
import subprocess
import sys

steps = [
    ["python", "src/04_extract_features.py"],
    ["python", "src/05_train_eval_real.py"],
    ["python", "src/14_connectivity.py"],
    ["python", "src/11_stats.py"],
    ["python", "src/12_plots.py"],
    ["python", "src/13_baseline_nn.py"],
    ["python", "src/15_loso_eval.py"],
    ["python", "src/17_mlp_loso.py"],
    ["python", "src/18_compare_protocols.py"],
    ["python", "src/19_loso_plots.py"],
    ["python", "src/20_export_tables.py"],
    ["python", "src/21_eval_subject_level.py"],
    ["python", "src/22_eval_state.py"],
    ["python", "src/24_paired_conditions.py"],
]

for cmd in steps:
    # признаки не пересчитываем, если нет сырых данных, но таблица уже лежит
    if cmd[1] in ("src/04_extract_features.py", "src/14_connectivity.py"):
        if not pathlib.Path("data/raw/ds004504").exists() and \
                pathlib.Path("data/features_spectral.csv").exists():
            print(f"skip {cmd[1]} - нет data/raw, беру готовые таблицы из data/")
            continue
    print(f">>> {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
print("pipeline done")
