"""Рисунки для текста курсовой (подписи по-русски).

Три рисунка, которых не хватало:
  paired_conditions - парное сравнение условий на общих людях;
  lead_by_condition - LEAD по обучающим наборам, разброс по фолдам;
  wpli_window_bias  - смещение wPLI по числу окон.

Результат: figures/paper/*.png. Английские рисунки из 12/19 не трогаются.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures" / "paper"
OUT.mkdir(parents=True, exist_ok=True)

RU = {"EC": "ГЗ", "EC_match": "ГЗ обрез.", "EO": "ГО",
      "EO_photic": "ГО вспышки", "EO_nophotic": "ГО без вспышек",
      "EO_broad": "ГО широкое"}
C_A, C_B = "#4C72B0", "#C44E52"


def fig_paired():
    df = pd.read_csv(ROOT / "reports/paired_conditions.csv")
    df = df[df.features == "compact"]
    pairs = [("EC", "EO"), ("EC", "EO_photic"),
             ("EC_match", "EO"), ("EC_match", "EO_photic")]
    tasks = [("AD_vs_HC", "БА против контроля", 0.5),
             ("3class", "Три класса", 1 / 3)]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, (task, name, chance) in zip(axes, tasks):
        d = df[df.task == task].set_index(["cond_a", "cond_b"])
        x = np.arange(len(pairs))
        a = [d.loc[p, "score_a"] for p in pairs]
        b = [d.loc[p, "score_b"] for p in pairs]
        ax.bar(x - 0.2, a, 0.38, color=C_A, label="закрытые глаза")
        ax.bar(x + 0.2, b, 0.38, color=C_B, label="открытые глаза")
        for i, p in enumerate(pairs):
            row = d.loc[p]
            top = max(a[i], b[i])
            ax.text(i, top + 0.035, f"n={int(row.n)}\np={row.p_mcnemar:.2f}",
                    ha="center", va="bottom", fontsize=8.5)
        ax.axhline(chance, color="grey", ls="--", lw=1)
        ax.text(len(pairs) - 0.45, chance + 0.012, "случайный уровень",
                fontsize=8, color="grey", ha="right",
                bbox=dict(fc="white", ec="none", pad=1.2))
        ax.set_xticks(x)
        ax.set_xticklabels([f"{RU[p[0]]}\n{RU[p[1]]}" for p in pairs],
                           fontsize=9)
        ax.set_ylim(0, 1.0)
        ax.set_ylabel("сбалансированная точность")
        ax.set_title(name, fontsize=11)
    fig.suptitle("Сравнение условий у одних и тех же людей "
                 "(LOSO, обучение на ГЗ)", fontsize=12)
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=2, fontsize=9.5,
               frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(OUT / "paired_conditions.png", dpi=200)
    plt.close(fig)


def _lead_ba(path, binary):
    d = pd.read_csv(path)
    d = d[d.level == "subject"].sort_values("fold")
    ba = (d.recall + d.specificity) / 2 if binary else d.recall
    return ba.to_numpy() * 100


def fig_lead():
    runs = [("смесь ГЗ и ГО", "adhc", "multi"),
            ("только ГЗ", "rs_adhc", "rs_multi"),
            ("только ГО", "ps_adhc", "ps_multi")]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
    for ax, (task, binary) in zip(axes, [("БА против контроля", True),
                                         ("Три класса", False)]):
        for i, (name, b_slug, m_slug) in enumerate(runs):
            slug = b_slug if binary else m_slug
            f = ROOT / f"reports/lead_5fold_{slug}.csv"
            if not f.exists():
                ax.text(i, 52, "прогон\nне готов", ha="center", fontsize=9,
                        color="grey")
                continue
            ba = _lead_ba(f, binary)
            ax.bar(i, ba.mean(), 0.55, color=C_A, alpha=0.85,
                   yerr=ba.std(ddof=0), capsize=5, ecolor="black")
            ax.plot(np.full(len(ba), i) + np.linspace(-0.12, 0.12, len(ba)),
                    ba, "o", ms=4, color="black", alpha=0.65)
            top = max(ba.max(), ba.mean() + ba.std(ddof=0))
            ax.text(i, top + 2.5, f"{ba.mean():.1f}", ha="center",
                    fontsize=9.5)
        ax.axhline(50 if binary else 100 / 3, color="grey", ls="--", lw=1)
        ax.set_xticks(range(len(runs)))
        ax.set_xticklabels([r[0] for r in runs], fontsize=9)
        ax.set_ylim(0, 105)
        ax.set_ylabel("сбалансированная точность, %")
        ax.set_title(task, fontsize=11)
    fig.suptitle("Предобученная модель LEAD: разбиение по людям, "
                 "5 блоков, точки — отдельные блоки", fontsize=11.5)
    fig.tight_layout()
    fig.savefig(OUT / "lead_by_condition.png", dpi=200)
    plt.close(fig)


def fig_wpli():
    c = pd.read_csv(ROOT / "data/features_conn.csv")
    w = c.pivot_table(index="participant_id", columns="condition",
                      values="wpli_alpha_global")
    n = c.pivot_table(index="participant_id", columns="condition",
                      values="n_windows_conn")
    w = w[["EC", "EC_match"]].dropna()
    st = wilcoxon(w.EC, w.EC_match)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
    ax = axes[0]
    for _, r in w.iterrows():
        ax.plot([0, 1], [r.EC, r.EC_match], color="grey", lw=0.7, alpha=0.5)
    ax.plot([0, 1], [w.EC.median(), w.EC_match.median()], "o-",
            color=C_B, lw=2.5, ms=8, label="медиана")
    ax.set_xticks([0, 1])
    ax.set_xticklabels([f"полная ГЗ\n{n.EC.median():.0f} окон",
                        f"обрезанная ГЗ\n{n.EC_match.median():.0f} окна"])
    ax.set_xlim(-0.35, 1.35)
    ax.set_ylabel("wPLI, альфа, среднее по парам")
    ax.set_title(f"Одни и те же записи, n = {len(w)}\n"
                 f"критерий Вилкоксона p = {st.pvalue:.1e}", fontsize=10.5)
    ax.legend(fontsize=9)

    ax = axes[1]
    m = c.dropna(subset=["wpli_alpha_global", "n_windows_conn"])
    ax.semilogx(m.n_windows_conn, m.wpli_alpha_global, "o", ms=4, alpha=0.45,
                color=C_A)
    ax.set_xlabel("число окон в записи")
    ax.set_ylabel("wPLI, альфа, среднее по парам")
    ax.set_title("Все условия: чем короче запись,\nтем выше оценка связности",
                 fontsize=10.5)
    fig.suptitle("Смещение оценки wPLI при малом числе окон", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "wpli_window_bias.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    fig_paired()
    fig_lead()
    fig_wpli()
    print("figures/paper:", ", ".join(sorted(p.name for p in OUT.glob("*.png"))))
