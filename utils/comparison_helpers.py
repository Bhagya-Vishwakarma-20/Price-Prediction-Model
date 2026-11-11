"""helper functions for the model comparison page (pure data work, no page layout)."""
import pandas as pd
import streamlit as st

from utils import load_data as ld

GREEN = "background-color: #b7e4c0; color: #0b3d1a; font-weight: 600"
RED = "background-color: #f6c3be; color: #5c0f0a; font-weight: 600"

# friendly column names for display
NAMES = {
    "level_label": "Level", "level": "Level code", "model": "Model", "model_family": "Model type",
    "uses_weather": "Uses weather", "commodity": "Commodity", "horizon": "Horizon (days ahead)",
    "split": "Period", "district": "District", "target_mode": "Target", "n": "Predictions (n)",
    "MAE": "MAE (Rs per quintal)", "RMSE": "RMSE (Rs per quintal)", "MAPE_percent": "MAPE (%)",
    "direction_accuracy": "Direction right (share)", "n_direction": "Direction checks (n)",
    "naive_MAE": "Naive MAE (Rs per quintal)", "naive_MAE_baseline_file": "Naive MAE from baseline file",
    "skill_vs_naive_percent": "Skill vs naive (%)", "settings": "Settings",
}


def style_map(styler, func, subset):
    """Styler.map in new pandas, applymap in old ones."""
    fn = getattr(styler, "map", None) or styler.applymap
    return fn(func, subset=subset)


def color_sign(good_when_positive):
    """cell colour from the sign of a number; green is good."""
    def f(v):
        if pd.isna(v) or v == 0:
            return ""
        good = (v > 0) if good_when_positive else (v < 0)
        return GREEN if good else RED
    return f


def color_yes_no(v):
    return GREEN if v == "Yes" else RED if v == "No" else ""


def splits_label(s):
    return {"valid": "Validation", "test": "Test", "train": "Training"}.get(s, str(s))


# ---------------------------------------------------------------- table 1
def mark_best(df):
    """boolean series: lowest MAE inside each (level, commodity, horizon, split, district) group."""
    keys = [k for k in ("level", "commodity", "horizon", "split", "district") if k in df.columns]
    best = df.groupby(keys, dropna=False)["MAE"].transform("min")
    return df["MAE"] == best


def filter_results(df, commodities, levels, horizons, splits, families, text):
    out = df
    if commodities:
        out = out[out["commodity"].isin(commodities)]
    if levels:
        out = out[out["level_label"].isin(levels)]
    if horizons:
        out = out[out["horizon"].isin(horizons)]
    if splits:
        out = out[out["split"].isin(splits)]
    if families:
        out = out[out["model_family"].isin(families)]
    text = (text or "").strip().lower()
    if text:
        hit = (out["model"].astype(str).str.lower().str.contains(text, regex=False)
               | out["district"].astype(str).str.lower().str.contains(text, regex=False))
        out = out[hit]
    return out


def results_view(df, show_settings):
    """(display dataframe with friendly names, boolean best-flags aligned to it)."""
    best = mark_best(df).reset_index(drop=True) if len(df) else pd.Series(dtype=bool)
    cols = ["level_label", "commodity", "horizon", "split", "district", "model", "model_family", "uses_weather",
            "target_mode", "n", "MAE", "RMSE", "MAPE_percent", "direction_accuracy", "n_direction",
            "naive_MAE", "skill_vs_naive_percent"]
    if show_settings:
        cols.append("settings")
    cols = [c for c in cols if c in df.columns]
    out = df[cols].reset_index(drop=True).rename(columns=NAMES)
    out["Period"] = out["Period"].map(splits_label)
    out["Best in group"] = best.map({True: "Best", False: ""})
    return out, best


def style_results(view, best):
    fmt = {"Predictions (n)": "{:,.0f}", "Direction checks (n)": "{:,.0f}",
           "MAE (Rs per quintal)": "{:,.1f}", "RMSE (Rs per quintal)": "{:,.1f}",
           "Naive MAE (Rs per quintal)": "{:,.1f}", "MAPE (%)": "{:.1f}%",
           "Direction right (share)": "{:.1%}", "Skill vs naive (%)": "{:+.1f}%"}
    fmt = {k: v for k, v in fmt.items() if k in view.columns}

    def paint(frame):
        styles = pd.DataFrame("", index=frame.index, columns=frame.columns)
        for col in ("Model", "MAE (Rs per quintal)", "Best in group"):
            if col in styles.columns:
                styles.loc[best.values, col] = GREEN
        return styles

    return view.style.apply(paint, axis=None).format(fmt, na_rep="-")


# ---------------------------------------------------------------- table 2
@st.cache_data(show_spinner=False)
def winners_table(comp):
    """per level, commodity and horizon: model with the lowest validation MAE and how it did on test."""
    o = ld.overall_rows(comp)
    rows = []
    for (level, commodity, horizon), g in o.groupby(["level", "commodity", "horizon"]):
        v = g[g["split"] == "valid"]
        if v.empty:
            continue
        w = v.loc[v["MAE"].idxmin()]
        t = g[(g["split"] == "test") & (g["model"] == w["model"])]
        t = t.iloc[0] if len(t) else None
        rows.append({
            "Level": ld.LEVEL_LABEL.get(level, level), "Commodity": commodity, "Horizon (days ahead)": int(horizon),
            "Chosen model": w["model"],
            "Validation MAE (Rs per quintal)": w["MAE"],
            "Validation skill vs naive (%)": w["skill_vs_naive_percent"],
            "Beat naive on validation?": "Yes" if w["skill_vs_naive_percent"] > 0 else "No",
            "Test MAE (Rs per quintal)": t["MAE"] if t is not None else float("nan"),
            "Naive test MAE (Rs per quintal)": t["naive_MAE"] if t is not None else float("nan"),
            "Test difference vs naive (%)": t["skill_vs_naive_percent"] if t is not None else float("nan"),
            "Beat naive on test?": ("Yes" if t["skill_vs_naive_percent"] > 0 else "No") if t is not None else "-",
        })
    return pd.DataFrame(rows)


def winners_sentence(w):
    n = len(w)
    if n == 0:
        return ""
    v = int((w["Beat naive on validation?"] == "Yes").sum())
    t = int((w["Beat naive on test?"] == "Yes").sum())
    t_known = int((w["Beat naive on test?"] != "-").sum())
    return (f"Of {n} chosen models (one per level, commodity and horizon), {v} beat the naive rule on validation "
            f"and {t} of {t_known} beat it on the test period. A positive difference means the model has a lower "
            f"error than simply repeating today's price.")


# ---------------------------------------------------------------- weather
@st.cache_data(show_spinner=False)
def weather_table(comp):
    return ld.weather_pairs(comp)


def weather_view(pairs):
    names = {"level": "Level", "commodity": "Commodity", "horizon": "Horizon (days ahead)",
             "model_with_weather": "Model with weather", "model_without_weather": "Model without weather",
             "MAE_valid_with": "Validation MAE with weather", "MAE_valid_without": "Validation MAE without weather",
             "change_valid_percent": "Validation change (%)",
             "MAE_test_with": "Test MAE with weather", "MAE_test_without": "Test MAE without weather",
             "change_test_percent": "Test change (%)"}
    cols = [c for c in names if c in pairs.columns]
    out = pairs[cols].rename(columns=names).reset_index(drop=True)
    if "Level" in out:
        out["Level"] = out["Level"].map(lambda x: ld.LEVEL_LABEL.get(x, x))
    return out


# ---------------------------------------------------------------- by district
def district_rows(comp, commodity, horizon, split):
    d = comp[(comp["level"] == "district") & (comp["commodity"] == commodity) & (comp["horizon"] == horizon)
             & (comp["split"] == split) & (comp["district"] != "ALL")]
    return d


def default_district_model(comp, commodity, horizon):
    """model with the lowest overall validation MAE at district level (None if there is none)."""
    o = comp[(comp["level"] == "district") & (comp["commodity"] == commodity) & (comp["horizon"] == horizon)
             & (comp["split"] == "valid") & (comp["district"] == "ALL")]
    return o.loc[o["MAE"].idxmin(), "model"] if len(o) else None


def lowest_highest(d, k=10):
    """(lowest-k, highest-k) by MAE; if there are 2k or fewer rows, all rows sorted and nothing for highest."""
    d = d.sort_values("MAE")
    if len(d) <= 2 * k:
        return d, d.iloc[0:0]
    return d.head(k), d.tail(k)


# ---------------------------------------------------------------- feature importance
def importance_color(comp, family, featureset):
    """colour of the model that matches an importance file's model type and feature set."""
    wants = "no" if str(featureset).lower().startswith("no") else "yes"
    m = comp[(comp["model_family"] == family) & (comp["uses_weather"].astype(str).str.lower() == wants)]
    if m.empty:
        m = comp[comp["model_family"] == family]
    return ld.model_color(m["model"].iloc[0]) if len(m) else ld.PALETTE[0]


# ---------------------------------------------------------------- charts
def test_mae_by_model(comp, commodity, level, horizons):
    """overall test MAE per model and horizon, plus the naive MAE per horizon."""
    o = ld.overall_rows(comp)
    o = o[(o["commodity"] == commodity) & (o["level"] == level) & (o["split"] == "test")
          & (o["horizon"].isin(horizons))]
    naive = o.groupby("horizon")["naive_MAE"].mean()
    return o, naive


def skill_matrix(comp, commodity, level, split):
    """models as rows, horizons as columns, skill vs naive (%)."""
    o = ld.overall_rows(comp)
    o = o[(o["commodity"] == commodity) & (o["level"] == level) & (o["split"] == split)]
    if o.empty:
        return pd.DataFrame()
    return o.pivot_table(index="model", columns="horizon", values="skill_vs_naive_percent", aggfunc="mean")
