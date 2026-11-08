import subprocess
steps=[
    ["python","src/04_extract_features.py"],
    ["python","src/05_train_eval_real.py"],
    ["python","src/11_stats.py"],
    ["python","src/12_plots.py"],
    ["python","src/13_baseline_nn.py"],
    ["python","src/15_loso_eval.py"],
    ["python","src/17_mlp_loso.py"],
    ["python","src/18_compare_protocols.py"],
]
for cmd in steps:
    print(f">>> {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
