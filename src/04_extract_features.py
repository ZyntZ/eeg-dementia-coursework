"""
04_extract_features.py - спектральные признаки, одна строка = человек + условие.

Переписано после проверки данных. Что было не так в первой версии:

1. Читались записи из корня датасета (data/raw/dsXXXXXX/sub-NNN/...), то есть
   НЕобработанные. У обоих наборов есть папка derivatives с теми же записями
   после Butterworth 0.5-45, перереференса на A1-A2, ASR и ICA/ICLabel.
   Разница на sub-001 EC: rel_delta 0.93 против 0.55, rel_alpha_occipital
   0.015 против 0.085. На сырых данных спектр забит глазными артефактами,
   отсюда и бейзлайн на уровне угадывания.

2. EO-куски склеивались через concatenate_raws и Welch считался по склейке.
   Каждый стык - это скачок сигнала, он даёт широкополосный выброс и ещё
   добавляет дельты. Теперь Welch считается по каждому куску отдельно и
   усредняется с весом по длине.

3. В derivatives после ASR остались метки boundary (вырезанные участки).
   Через них тоже нельзя считать спектр, режем куски по ним.

4. Фотостимуляция не исключалась, хотя в докстринге было написано обратное.
   По всем 88 испытуемым на вспышки приходится 20% времени ГО, причём среди
   блоков есть 10 Гц - прямо в альфе. Теперь EO_photic (под вспышками) и
   EO_nophotic (без) считаются отдельно.

5. EC длиннее EO примерно в 35 раз, спектральные оценки несопоставимы.
   Пишем ещё EC_match - тот же EC, обрезанный до длительности EO
   этого же человека.

6. Частота дискретизации была константой 500, окно Welch считалось от неё.
   Берём из файла.

7. Если затылочные каналы не находились по именам, функция молча считала
   среднее по всем каналам. Теперь падает.

8. Окно Welch было разным для разных условий, потому что подбиралось под
   длину куска. Тогда спектр длинной EC и короткой EO оценивается с разной
   точностью, и классификатор различает длительность записи, а не состояние.
   Окно жёстко 2 с для всех условий, n_fft фиксирован, куски короче окна
   отбрасываются. В таблицу пишется n_windows.

Условия на выходе: EC, EC_match, EO, EO_photic, EO_nophotic, EO_broad.
EO - основное правило "от open eyes до следующего closed eyes".
EO_broad - широкое правило "вся запись минус закрытые глаза", контрольное:
по нему альфа у контроля почти не падает, то есть в него попадают
и закрытые глаза тоже.
"""
import argparse
import json
import re
import sys
from pathlib import Path

import mne
import numpy as np
import pandas as pd

mne.set_log_level("ERROR")

RAW_ROOT = Path("data/raw")
OUT = Path("data/features_spectral.csv")

# derivatives лежат по-разному в двух наборах, поэтому пути перечислены явно
DATASETS = {
    "EC": dict(
        ds="ds004504",
        task="eyesclosed",
        deriv=["derivatives/{sub}/eeg/{sub}_task-{task}_eeg.set"],
        raw="{sub}/eeg/{sub}_task-{task}_eeg.set",
    ),
    "EO": dict(
        ds="ds006036",
        task="photomark",
        deriv=["derivatives/eeglab/{sub}/eeg/{sub}_task-{task}_eeg.set",
               "derivatives/{sub}/eeg/{sub}_task-{task}_eeg.set"],
        raw="{sub}/eeg/{sub}_task-{task}_eeg.set",
    ),
}

BANDS = {"delta": (1, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 30),
         "alpha_low": (8, 10), "alpha_high": (10, 13)}
F_TOTAL = (1.0, 45.0)        # знаменатель относительной мощности
OCCIPITAL = ["O1", "O2", "P3", "P4", "Pz"]
FRONTAL = ["Fp1", "Fp2", "F3", "F4", "Fz", "F7", "F8"]

WELCH_WIN = 2.0              # с, одинаково для ВСЕХ условий: иначе спектр у
                             # длинной ГЗ и короткой ГО оценивается с разной
                             # точностью и модель ловит это, а не состояние
MIN_SEG = 2.0                # куски короче окна не берём
MIN_TOTAL = 6.0              # меньше трёх окон на условие не считаем
PAD_ARTEFACT = 1.0           # с вокруг артефактных меток
PAD_BOUNDARY = 0.5           # с вокруг стыков ASR
FLASH_MERGE_GAP = 1.0        # вспышки ближе этого считаем одним блоком

ARTEFACT_MARKS = {"eye movement", "swallowing", "speech", "muscle",
                  "head movement", "chewing", "movement", "sweat"}
STOP_MARKS = {"cal mode", "reset condition"}   # с них и до конца - калибровка


def clean_label(s):
    """В .set метки приходят с хвостовыми пробелами и управляющими символами."""
    return re.sub(r"[^\x20-\x7e\u0400-\u04ff]", " ", str(s)).strip().lower()


def resolve_path(sub, cond):
    """derivatives, если есть; иначе сырые с предупреждением."""
    cfg = DATASETS[cond]
    root = RAW_ROOT / cfg["ds"]
    for tpl in cfg["deriv"]:
        p = root / tpl.format(sub=sub, task=cfg["task"])
        if p.exists():
            return p, "derivatives"
    p = root / cfg["raw"].format(sub=sub, task=cfg["task"])
    if p.exists():
        print(f"  ВНИМАНИЕ {sub} {cond}: derivatives не найдены, беру сырую запись")
        return p, "raw"
    return None, None


# ---------------------------------------------------------------- интервалы

def reref(raw):
    """Общий средний референс. В derivatives записи стоят на A1-A2, общая для
    всех каналов медленная составляющая никуда не делась; без этого шага
    относительная дельта выходит около 0.85 вместо примерно 0.5."""
    return raw.set_eeg_reference("average")


def merge(spans, gap=0.0):
    spans = sorted(s for s in spans if s[1] > s[0])
    out = []
    for a, b in spans:
        if out and a - out[-1][1] <= gap:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [tuple(s) for s in out]


def intersect(a_spans, b_spans):
    out = []
    for a, b in a_spans:
        for c, d in b_spans:
            lo, hi = max(a, c), min(b, d)
            if hi - lo >= MIN_SEG:
                out.append((lo, hi))
    return merge(out)


def subtract(spans, holes):
    out = []
    for a, b in spans:
        cur = [(a, b)]
        for ha, hb in holes:
            nxt = []
            for ca, cb in cur:
                if hb <= ca or ha >= cb:
                    nxt.append((ca, cb))
                    continue
                if ha > ca:
                    nxt.append((ca, ha))
                if hb < cb:
                    nxt.append((hb, cb))
            cur = nxt
        out += cur
    return [s for s in out if s[1] - s[0] >= MIN_SEG]


def annotations_table(raw, sub, cond):
    """Метки берём из самого .set: после ASR латентности уже пересчитаны.
    Если их нет - из events.tsv (в разных версиях наборов колонка называется
    то value, то trial_type)."""
    if len(raw.annotations):
        return pd.DataFrame({"onset": raw.annotations.onset,
                             "label": [clean_label(d) for d in raw.annotations.description]})
    cfg = DATASETS[cond]
    ev = RAW_ROOT / cfg["ds"] / f"{sub}/eeg/{sub}_task-{cfg['task']}_events.tsv"
    if not ev.exists():
        return pd.DataFrame(columns=["onset", "label"])
    df = pd.read_csv(ev, sep="\t")
    col = next((c for c in ("trial_type", "value") if c in df.columns), None)
    if col is None:
        raise ValueError(f"{ev}: нет ни trial_type, ни value")
    return pd.DataFrame({"onset": df["onset"].astype(float),
                         "label": df[col].map(clean_label)})


def excluded_spans(ann, tmax):
    holes = []
    for _, r in ann.iterrows():
        if r.label in ARTEFACT_MARKS:
            holes.append((r.onset - PAD_ARTEFACT, r.onset + PAD_ARTEFACT))
        elif r.label == "boundary":
            holes.append((r.onset - PAD_BOUNDARY, r.onset + PAD_BOUNDARY))
        elif r.label in STOP_MARKS or "cal" in r.label:
            holes.append((r.onset - 1.0, tmax))
    return merge(holes)


def photic_spans(ann):
    flashes = ann[ann.label.str.contains("photo/hv", na=False)].onset.values
    if len(flashes) == 0:
        return []
    return merge([(f - 0.1, f + 0.3) for f in flashes], gap=FLASH_MERGE_GAP)


def eye_spans(ann, tmax, rule):
    """rule='strict' - только от open eyes до следующего closed eyes (как было).
    rule='broad'  - вся запись минус промежутки с закрытыми глазами. Автор
    набора в обсуждении на OpenNeuro пишет, что глаза были открыты всю запись,
    то есть метки отмечают кратковременные закрывания. Правило спорное,
    поэтому считаем оба варианта и сравниваем."""
    op = ann[ann.label.str.contains("open eyes", na=False)].onset.values
    cl = ann[ann.label.str.contains("closed eyes", na=False)].onset.values
    if rule == "strict":
        out = []
        for o in op:
            nxt = cl[cl > o]
            if len(nxt):
                out.append((o, nxt[0]))
        return merge(out)
    closed = []
    for c in cl:
        nxt = op[op > c]
        closed.append((c, nxt[0] if len(nxt) else tmax))
    return subtract([(0.0, tmax)], merge(closed))


# ---------------------------------------------------------------- спектр

def welch_mean(raw, spans):
    """Welch по каждому куску отдельно, среднее с весом по числу окон."""
    sf = raw.info["sfreq"]
    # окно фиксированное: сетка частот обязана совпадать у всех кусков,
    # иначе усреднять нечего
    n_fft = int(round(WELCH_WIN * sf))
    psds, weights, freqs, total = [], [], None, 0.0
    for a, b in spans:
        a, b = max(a, 0.0), min(b, raw.times[-1])
        if b - a < MIN_SEG:
            continue
        seg = raw.copy().crop(a, b, include_tmax=False)
        if seg.n_times < n_fft:
            continue
        p = seg.compute_psd(method="welch", fmin=F_TOTAL[0], fmax=F_TOTAL[1],
                            n_fft=n_fft, n_per_seg=n_fft, n_overlap=n_fft // 2)
        if freqs is not None and len(p.freqs) != len(freqs):
            continue
        psds.append(p.get_data())
        weights.append(seg.n_times / sf)
        total += seg.n_times / sf
        freqs = p.freqs
    if not psds or total < MIN_TOTAL:
        return None, None, 0.0, 0
    w = np.asarray(weights)[:, None, None]
    return (np.sum(np.stack(psds) * w, axis=0) / w.sum()), freqs, total, len(psds)


def peak_alpha(psd, freqs):
    """Старая версия брала argmax в окне 8-13 Гц. На спаде 1/f максимум почти
    всегда оказывается на границе окна: в старой таблице ровно 8.00 Гц стояло
    у 36 из 88 человек. Сначала убираем аперионический наклон, потом ищем
    локальный максимум; если его нет - NaN."""
    m = psd.mean(0)
    fit = (freqs >= 2) & (freqs <= 40) & ~((freqs > 6) & (freqs < 15))
    k, b = np.polyfit(np.log(freqs[fit]), np.log(m[fit]), 1)
    flat = np.log(m) - (k * np.log(freqs) + b)
    win = (freqs >= 6) & (freqs <= 14)
    f, v = freqs[win], flat[win]
    loc = [i for i in range(1, len(v) - 1) if v[i] > v[i - 1] and v[i] >= v[i + 1]]
    if not loc:
        return np.nan, 0.0, False
    i = max(loc, key=lambda j: v[j])
    return float(f[i]), float(v[i]), True


def features(raw, spans):
    psd, freqs, dur, n_seg = welch_mean(raw, spans)
    if psd is None:
        return None
    names = [c.strip() for c in raw.ch_names]
    occ = [names.index(c) for c in OCCIPITAL if c in names]
    fro = [names.index(c) for c in FRONTAL if c in names]
    if not occ or not fro:
        raise ValueError(f"не нашёл каналы по именам, есть только: {names}")
    tot = psd.sum(1) + 1e-20
    f = {}
    for n, (lo, hi) in BANDS.items():
        rel = psd[:, (freqs >= lo) & (freqs < hi)].sum(1) / tot
        f[f"rel_{n}_mean"] = rel.mean()
        f[f"rel_{n}_occipital"] = rel[occ].mean()
        f[f"rel_{n}_frontal"] = rel[fro].mean()
    f["alpha_theta_ratio"] = f["rel_alpha_mean"] / (f["rel_theta_mean"] + 1e-12)
    f["theta_beta_ratio"] = f["rel_theta_mean"] / (f["rel_beta_mean"] + 1e-12)
    pn = psd.mean(0) / psd.mean(0).sum()
    f["spectral_entropy"] = float(-(pn * np.log(pn + 1e-20)).sum())
    pk, amp, ok = peak_alpha(psd, freqs)
    f["peak_alpha"] = pk
    f["peak_alpha_amp"] = amp
    f["peak_alpha_found"] = ok
    f["duration"] = dur
    f["n_segments"] = n_seg
    f["n_windows"] = int(dur // (WELCH_WIN / 2)) - 1
    return f


def head_spans(spans, seconds):
    """Первые seconds секунд суммарно - для EC_match."""
    out, left = [], seconds
    for a, b in spans:
        if left <= 0:
            break
        take = min(b - a, left)
        out.append((a, a + take))
        left -= take
    return out


# ---------------------------------------------------------------- основной цикл

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="взять первых N человек")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--allow-partial", action="store_true",
                    help="разрешить перезапись таблицы неполным прогоном")
    args = ap.parse_args()

    # список людей берём из participants.tsv, а не из имён папок: если скачаны
    # только derivatives, папок sub-* в корне набора нет
    plist = RAW_ROOT / "ds004504/participants.tsv"
    if not plist.exists():
        sys.exit(f"нет {plist}, сначала src/00_fetch_data.py")
    subs = sorted(pd.read_csv(plist, sep="\t")["participant_id"].astype(str))
    if args.limit:
        subs = subs[:args.limit]
    print(f"испытуемых: {len(subs)}")

    rows, sources = [], {}
    for sub in subs:
        print(sub, flush=True)
        per_sub = {}

        # --- EO
        p, src = resolve_path(sub, "EO")
        eo_rest_dur = None
        if p is None:
            print(f"  {sub}: нет записи EO")
        else:
            sources[f"{sub}/EO"] = src
            raw = reref(mne.io.read_raw_eeglab(p, preload=True))
            ann = annotations_table(raw, sub, "EO")
            tmax = raw.times[-1]
            holes = excluded_spans(ann, tmax)
            photic = photic_spans(ann)
            open_broad = subtract(eye_spans(ann, tmax, "broad"), holes)
            open_strict = subtract(eye_spans(ann, tmax, "strict"), holes)
            cond_spans = {
                # основное условие ГО - строгое правило
                "EO": open_strict,
                # разбивка ГО на со вспышками и без
                "EO_photic": intersect(open_strict, photic),
                "EO_nophotic": subtract(open_strict, photic),
                # контроль: широкое правило "вся запись минус закрытые глаза".
                # если оно верное, альфа тут должна падать как в EO
                "EO_broad": open_broad,
            }
            for name, spans in cond_spans.items():
                f = features(raw, spans)
                if f:
                    per_sub[name] = f
            if "EO" in per_sub:
                eo_rest_dur = per_sub["EO"]["duration"]
            del raw

        # --- EC
        p, src = resolve_path(sub, "EC")
        if p is None:
            print(f"  {sub}: нет записи EC")
        else:
            sources[f"{sub}/EC"] = src
            raw = reref(mne.io.read_raw_eeglab(p, preload=True))
            ann = annotations_table(raw, sub, "EC")
            tmax = raw.times[-1]
            spans = subtract([(0.0, tmax)], excluded_spans(ann, tmax))
            f = features(raw, spans)
            if f:
                per_sub["EC"] = f
            if eo_rest_dur:
                fm = features(raw, head_spans(spans, eo_rest_dur))
                if fm:
                    per_sub["EC_match"] = fm
            del raw

        for cond, f in per_sub.items():
            rows.append({**f, "participant_id": sub, "condition": cond})

    df = pd.DataFrame(rows)
    meta = pd.read_csv(RAW_ROOT / "ds004504/participants.tsv", sep="\t")
    meta.columns = meta.columns.str.replace("\ufeff", "").str.strip()
    meta = meta[["participant_id", "Group", "Age", "MMSE", "Gender"]].rename(
        columns={"Group": "diagnosis"})
    meta["diagnosis"] = meta["diagnosis"].map({"A": "AD", "F": "FTD", "C": "HC"})
    df = df.merge(meta, on="participant_id", how="left")

    # Частичная докачка data/raw - обычное дело, и тогда таблица выходит
    # куцей. Молча затирать готовую полную таблицу нельзя.
    got, expect = df["participant_id"].nunique(), len(meta)
    out_path = Path(args.out)
    if got < expect and out_path.exists() and not (args.limit or args.allow_partial):
        sys.exit(f"ОТКАЗ: записи найдены у {got} человек из {expect}, "
                 f"а {out_path} уже лежит готовый. Докачайте данные "
                 f"(src/00_fetch_data.py) или запустите с --allow-partial")
    df.to_csv(args.out, index=False)

    print("\nстрок:", len(df))
    print(df.groupby(["condition", "diagnosis"]).size().unstack(fill_value=0))
    print("\nдлительность, с:")
    print(df.groupby("condition")["duration"].describe()[["count", "min", "50%", "max"]].round(1))
    print("\nпик альфа найден:", df.groupby("condition")["peak_alpha_found"].mean().round(2).to_dict())
    n_raw = sum(1 for v in sources.values() if v == "raw")
    if n_raw:
        print(f"\nВНИМАНИЕ: {n_raw} записей прочитано из сырых данных, не из derivatives")
    Path("reports").mkdir(exist_ok=True)
    Path("reports/feature_sources.json").write_text(json.dumps(sources, indent=1))
    print("сохранено", args.out)


if __name__ == "__main__":
    main()
