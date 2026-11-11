"""loading, caching and shared helpers for the dashboard. only reads files from OUTPUT_DIR."""
import glob
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------- settings
OUTPUT_DIR = "output"          # folder with the result files (relative to the app folder, or absolute)
NAIVE_LABEL = "Naive (no change)"
SPLIT_ORDER = ["train", "valid", "test"]
LEVEL_LABEL = {"state": "Karnataka-wide", "district": "District"}
LEVEL_FROM_LABEL = {v: k for k, v in LEVEL_LABEL.items()}
APP_ROOT = Path(__file__).resolve().parents[1]

# colour-blind-friendly palette (okabe-ito); one colour per model, the same on every chart
PALETTE = ["#0072B2", "#E69F00", "#009E73", "#CC79A8", "#56B4E9", "#D55E00", "#F0E442", "#7F7F7F"]
NAIVE_COLOR = "#8A8A8A"
ACTUAL_COLOR = "#222222"      # works on light and dark backgrounds because charts use a light plot area
SPLIT_SHADE = {"valid": "rgba(230,159,0,0.16)", "test": "rgba(204,121,167,0.20)"}
GOOD_COLOR, BAD_COLOR = "#1a9850", "#d73027"


# ---------------------------------------------------------------- paths and messages
def output_dir():
    p = Path(OUTPUT_DIR)
    return p if p.is_absolute() else APP_ROOT / p


def output_path(name):
    return str(output_dir() / name)


def file_exists(name):
    return os.path.isfile(output_path(name))


def show_missing(name, what=""):
    """clear message for a missing or unreadable file; the page carries on."""
    extra = f" It is needed for {what}." if what else ""
    st.warning(f"File not found or unreadable: `{OUTPUT_DIR}/{name}`.{extra}")


# ---------------------------------------------------------------- generic csv loading (cached)
@st.cache_data(show_spinner=False)
def _read_csv(path, mtime):
    """mtime is part of the cache key, so a changed file is read again."""
    return pd.read_csv(path)


def load_csv(name):
    """dataframe, or None if the file is missing or cannot be read."""
    path = output_path(name)
    if not os.path.isfile(path):
        return None
    try:
        return _read_csv(path, os.path.getmtime(path))
    except Exception:
        return None


def clean_key(text):
    return re.sub(r"[^a-z0-9]", "", str(text).lower())


# ---------------------------------------------------------------- result files
def load_comparison():
    """model_comparison.csv (one row per model, commodity, horizon, split, district)."""
    df = load_csv("model_comparison.csv")
    if df is None:
        return None
    df = df.copy()
    df["horizon"] = pd.to_numeric(df["horizon"], errors="coerce").astype("Int64")
    df["model_family"] = df["model"].map(model_family)
    df["level_label"] = df["level"].map(LEVEL_LABEL).fillna(df["level"])
    return df


def load_baseline():
    return load_csv("baseline_scores.csv")


def load_report_text():
    path = output_path("training_report.txt")
    if not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return f.read()
    except OSError:
        return None


def load_importance(commodity, horizon):
    return load_csv(f"feature_importance_{commodity.lower()}_h{int(horizon)}.csv")


def model_family(label):
    """'XGBoost (no weather)' -> 'XGBoost'; 'SARIMAX (weather)' -> 'SARIMAX'."""
    return re.sub(r"\s*\((no )?weather\)", "", str(label)).strip()


def prediction_files():
    """list of (level, commodity, horizon) that have a predictions file."""
    found = []
    for p in glob.glob(output_path("predictions_*_h*.csv")):
        m = re.match(r"predictions_([a-z]+)_([a-z]+)_h(\d+)\.csv$", os.path.basename(p))
        if m:
            found.append((m.group(1), m.group(2).capitalize(), int(m.group(3))))
    return sorted(found)


def available_levels():
    return sorted({lv for lv, _, _ in prediction_files()}, key=lambda x: ["state", "district"].index(x)
                  if x in ("state", "district") else 9)


def available_commodities():
    names = {c for _, c, _ in prediction_files()}
    comp = load_comparison()
    if comp is not None:
        names |= set(comp["commodity"].unique())
    return sorted(names)


def available_horizons(level=None, commodity=None):
    hs = {h for lv, c, h in prediction_files()
          if (level is None or lv == level) and (commodity is None or c == commodity)}
    if not hs:
        comp = load_comparison()
        if comp is not None:
            hs = set(int(x) for x in comp["horizon"].dropna().unique())
    return sorted(hs)


def all_models():
    """model labels from model_comparison.csv (stable order)."""
    comp = load_comparison()
    return sorted(comp["model"].unique()) if comp is not None else []


def load_predictions(level, commodity, horizon):
    """predictions file with tidy column names, or None.
    columns: date (day t), target_date, split, [district], price_today, actual, naive, one column per model."""
    df = load_csv(f"predictions_{level}_{commodity.lower()}_h{int(horizon)}.csv")
    if df is None:
        return None
    df = df.copy()
    labels = {clean_key(m): m for m in all_models()}
    rename = {"date_t": "date", "actual_price_at_t_plus_h": "actual", "naive": NAIVE_LABEL}
    for col in df.columns:
        if col not in ("date_t", "target_date", "split", "district", "price_today",
                       "actual_price_at_t_plus_h", "naive"):
            rename[col] = labels.get(clean_key(col), col.replace("_", " "))
    df = df.rename(columns=rename)
    for c in ("date", "target_date"):
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    return df


def model_columns(df):
    """prediction columns of a tidy predictions dataframe, without the naive rule."""
    base = {"date", "target_date", "split", "district", "price_today", "actual", NAIVE_LABEL}
    return [c for c in df.columns if c not in base]


def load_history(level, commodity):
    """full price history with the split label (model_data_* files), or None."""
    df = load_csv(f"model_data_{level}_{commodity.lower()}.csv")
    if df is None:
        return None
    cols = [c for c in ("date", "district", "price", "split") if c in df.columns]
    out = df[cols].copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    return out


def split_periods(commodity):
    """{'train': (start, end), 'valid': (...), 'test': (...)} from the state history file."""
    h = load_history("state", commodity)
    if h is None or "split" not in h.columns:
        h = load_history("district", commodity)
    if h is None or "split" not in h.columns:
        return {}
    g = h.groupby("split")["date"].agg(["min", "max"])
    return {s: (g.loc[s, "min"], g.loc[s, "max"]) for s in SPLIT_ORDER if s in g.index}


def districts_for(commodity):
    """districts that have district-level predictions."""
    comp = load_comparison()
    if comp is None:
        return []
    c = comp[(comp["level"] == "district") & (comp["commodity"] == commodity)]
    return sorted(d for d in c["district"].unique() if d != "ALL")


# ---------------------------------------------------------------- colours
def model_color(label):
    """same model, same colour everywhere (position in the sorted model list)."""
    if label == NAIVE_LABEL or str(label).lower().startswith("naive"):
        return NAIVE_COLOR
    models = all_models()
    if label in models:
        return PALETTE[models.index(label) % len(PALETTE)]
    return PALETTE[abs(hash(label)) % len(PALETTE)]


# ---------------------------------------------------------------- number formatting
def fmt_rs(x, digits=0):
    return "-" if pd.isna(x) else f"Rs {x:,.{digits}f}"


def fmt_pct(x, digits=1):
    return "-" if pd.isna(x) else f"{x:,.{digits}f}%"


def fmt_int(x):
    return "-" if pd.isna(x) else f"{int(x):,}"


# ---------------------------------------------------------------- metrics on rows
def row_metrics(actual, pred, today):
    """MAE, MAPE and direction accuracy for one prediction column on the given rows."""
    ok = actual.notna() & pred.notna() & today.notna()
    a, p, t = actual[ok].to_numpy(float), pred[ok].to_numpy(float), today[ok].to_numpy(float)
    if len(a) == 0:
        return {"n": 0, "MAE": np.nan, "MAPE_percent": np.nan, "direction_accuracy": np.nan}
    moved = a != t
    direction = (np.sign(p - t)[moved] == np.sign(a - t)[moved]).mean() if moved.any() else np.nan
    return {"n": int(len(a)), "MAE": float(np.abs(p - a).mean()),
            "MAPE_percent": float((np.abs(p - a) / a).mean() * 100), "direction_accuracy": direction}


# ---------------------------------------------------------------- shared calculations
def overall_rows(comp):
    """rows for the whole level (district == ALL)."""
    return comp[comp["district"] == "ALL"]


def beat_naive_on_valid(comp):
    """(number of model-horizon combinations that beat naive on validation, total combinations)."""
    v = overall_rows(comp)
    v = v[v["split"] == "valid"].drop_duplicates(["level", "commodity", "horizon", "model"])
    return int((v["skill_vs_naive_percent"] > 0).sum()), len(v)


def weather_pairs(comp):
    """each weather model next to its no-weather twin (same level, commodity, horizon)."""
    o = overall_rows(comp)
    rows = []
    no_w = o[o["uses_weather"].astype(str).str.lower() == "no"]
    with_w = o[o["uses_weather"].astype(str).str.lower() == "yes"]
    for (level, commodity, horizon, model), g in with_w.groupby(["level", "commodity", "horizon", "model"]):
        fam = model_family(model)
        twins = no_w[(no_w["level"] == level) & (no_w["commodity"] == commodity) & (no_w["horizon"] == horizon)]
        match = twins[twins["model"].map(model_family) == fam]
        if match.empty and fam.endswith("X"):          # SARIMAX pairs with SARIMA
            match = twins[twins["model"].map(model_family) == fam[:-1]]
        if match.empty:
            continue
        twin = match["model"].iloc[0]
        rec = {"level": level, "commodity": commodity, "horizon": int(horizon),
               "model_with_weather": model, "model_without_weather": twin}
        for split in ("valid", "test"):
            a = g[g["split"] == split]
            b = match[(match["model"] == twin) & (match["split"] == split)]
            if a.empty or b.empty:
                continue
            rec[f"MAE_{split}_with"] = a["MAE"].iloc[0]
            rec[f"MAE_{split}_without"] = b["MAE"].iloc[0]
            rec[f"change_{split}_percent"] = (a["MAE"].iloc[0] - b["MAE"].iloc[0]) / b["MAE"].iloc[0] * 100
        rows.append(rec)
    return pd.DataFrame(rows)


def weather_verdict(pairs, tolerance=1.0):
    """one sentence about whether weather helped, calculated from the pairs table."""
    if pairs is None or pairs.empty:
        return "There are no with/without-weather pairs to compare."
    parts, verdicts = [], []
    for split in ("valid", "test"):
        col = f"change_{split}_percent"
        if col not in pairs:
            continue
        ch = pairs[col].dropna()
        if ch.empty:
            continue
        better, worse = int((ch < -tolerance).sum()), int((ch > tolerance).sum())
        same = len(ch) - better - worse
        name = "validation" if split == "valid" else "test"
        parts.append(f"on {name} weather lowered the error in {better} of {len(ch)} comparisons, raised it in "
                     f"{worse} and made no real difference (within {tolerance:.0f}%) in {same} "
                     f"(average change {ch.mean():+.1f}%)")
        verdicts.append((better / len(ch), worse / len(ch)))
    if not parts:
        return "There are no with/without-weather pairs to compare."
    helped = all(b >= 0.75 for b, _ in verdicts)
    hurt = all(w >= 0.75 for _, w in verdicts)
    conclusion = ("Weather helped consistently." if helped else
                  "Weather made results worse consistently." if hurt else
                  "Weather did not give a stable improvement: the effect changes sign between models, "
                  "horizons and periods.")
    return conclusion + " In detail, " + "; ".join(parts) + "."


def summary_stats():
    """numbers for the home page cards, all read from the files (None if not available)."""
    out = {"districts": None, "date_min": None, "date_max": None, "models": None,
           "beat": None, "combos": None, "test_start": None, "test_end": None,
           "test_n_min": None, "test_n_max": None}
    comp = load_comparison()
    if comp is not None:
        out["models"] = int(comp["model"].nunique())
        out["beat"], out["combos"] = beat_naive_on_valid(comp)
        t = overall_rows(comp)
        t = t[t["split"] == "test"]
        if len(t):
            out["test_n_min"], out["test_n_max"] = int(t["n"].min()), int(t["n"].max())
    mins, maxs, test_starts, test_ends, district_names = [], [], [], [], set()
    for commodity in available_commodities():
        for level in ("state", "district"):
            h = load_history(level, commodity)
            if h is None:
                continue
            mins.append(h["date"].min())
            maxs.append(h["date"].max())
            if level == "district" and "district" in h.columns:
                district_names |= set(h["district"].dropna().unique())
                out["districts"] = len(district_names)
            if "split" in h.columns and (h["split"] == "test").any():
                t = h.loc[h["split"] == "test", "date"]
                test_starts.append(t.min())
                test_ends.append(t.max())
    if mins:
        out["date_min"], out["date_max"] = min(mins), max(maxs)
    if test_starts:
        out["test_start"], out["test_end"] = min(test_starts), max(test_ends)
    if out["districts"] is None and comp is not None:
        out["districts"] = len([d for d in comp["district"].unique() if d != "ALL"]) or None
    return out
