import subprocess, sys, pathlib
steps=[
    ["python","src/04_extract_features.py"],
    ["python","src/05_train_eval_real.py"],
    ["python","src/11_stats.py"],
    ["python","src/12_plots.py"],
    ["python","src/13_baseline_nn.py"],
    ["python","src/15_loso_eval.py"],
    ["python","src/17_mlp_loso.py"],
    ["python","src/18_compare_protocols.py"],
    ["python","src/19_loso_plots.py"],
    ["python","src/20_export_tables.py"],
]
for cmd in steps:
    # пропускаем извлечение если нет raw но есть features
    if cmd[1]=="src/04_extract_features.py":
        if not pathlib.Path("data/raw/ds004504/participants.tsv").exists() and pathlib.Path("data/features_spectral.csv").exists():
            print(f"skip {cmd[1]} — raw not present, features exist")
            continue
    print(f">>> {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
print("pipeline done")
