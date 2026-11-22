import pathlib
import subprocess

steps = [
    ["python", "src/04_extract_features.py"],
    ["python", "src/05_train_eval_real.py"],
    ["python", "src/06_feature_analysis.py"],
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
    ["python", "src/25_per_class.py"],
    ["python", "src/26_figures_paper.py"],
]

def raw_status():
    """Сколько записей скачано и сколько ожидается.

    Проверять наличие каталога мало: при оборванной докачке в data/raw
    лежит часть записей, и пересчёт затирает готовые таблицы неполными.
    """
    root = pathlib.Path("data/raw")
    plist = root / "ds004504/participants.tsv"
    if not plist.exists():
        return 0, 0     # данных нет вовсе: считать нечего
    people = sum(1 for i, line in enumerate(plist.read_text().splitlines())
                 if i and line.strip())
    got = len(list(root.glob("ds004504/derivatives/**/*.set"))) + \
        len(list(root.glob("ds006036/derivatives/**/*.set")))
    return got, 2 * people


HEAVY = ("src/04_extract_features.py", "src/14_connectivity.py")
got, expect = raw_status()
raw_complete = expect > 0 and got >= expect

for cmd in steps:
    # признаки пересчитываем только при полной выгрузке data/raw
    if cmd[1] in HEAVY and not raw_complete:
        if not pathlib.Path("data/features_spectral.csv").exists():
            raise SystemExit(f"нет готовых таблиц, а в data/raw {got} записей "
                             f"из {expect}: запустите src/00_fetch_data.py")
        why = (f"в data/raw {got} записей из {expect}" if expect
               else "нет data/raw")
        print(f"skip {cmd[1]} - {why}, беру готовые таблицы из data/")
        continue
    print(f">>> {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
print("pipeline done")
