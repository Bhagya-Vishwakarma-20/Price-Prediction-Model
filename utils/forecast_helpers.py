"""helper functions for the price forecast page (no streamlit calls, only data and figures)."""
import pandas as pd
import plotly.graph_objects as go

from utils import load_data as ld

SPLIT_NAME = {"valid": "Validation period", "test": "Test period"}
PRICE_AXIS = "Price (Rs per quintal)"
ERROR_AXIS = "Forecast minus actual (Rs per quintal)"


def match_label(comp, label):
    """model label as written in the comparison file (labels are normally identical)."""
    if comp is None or label in set(comp["model"]):
        return label
    keys = {ld.clean_key(m): m for m in comp["model"].unique()}
    return keys.get(ld.clean_key(label), label)


def scope_rows(comp, level, commodity, horizon, district, split):
    """comparison rows for one selection. returns (rows, scope_text).
    state level uses district == ALL; district level uses the chosen district if present, else ALL."""
    base = comp[(comp["level"] == level) & (comp["commodity"] == commodity)
                & (comp["horizon"] == int(horizon)) & (comp["split"] == split)]
    if level == "district" and district:
        own = base[base["district"] == district]
        if len(own):
            return own, f"the {district.title()} row"
        return base[base["district"] == "ALL"], "the overall all-districts row (no row for this district)"
    return base[base["district"] == "ALL"], "the Karnataka-wide row" if level == "state" else "the overall row"


def default_models(comp, level, commodity, horizon, district, available):
    """the model with the lowest validation MAE among the models available in the predictions file."""
    if comp is None or not available:
        return list(available[:1])
    rows, _ = scope_rows(comp, level, commodity, horizon, district, "valid")
    rows = rows[rows["model"].map(lambda m: m in available) & rows["MAE"].notna()]
    if rows.empty:
        return list(available[:1])
    return [rows.sort_values("MAE")["model"].iloc[0]]


def validation_warnings(comp, level, commodity, horizon, district, models):
    """list of warning texts for models that do not beat the naive rule on validation."""
    if comp is None:
        return []
    rows, scope = scope_rows(comp, level, commodity, horizon, district, "valid")
    notes = []
    for m in models:
        r = rows[rows["model"] == match_label(comp, m)]
        if r.empty:
            notes.append(f"**{m}**: no validation score was found in model_comparison.csv, so it cannot be "
                         f"compared with the naive rule.")
            continue
        r = r.iloc[0]
        skill = r["skill_vs_naive_percent"]
        small = f" Only {int(r['n']):,} validation rows were scored, so this result is uncertain." \
            if pd.notna(r["n"]) and r["n"] < 100 else ""
        if pd.isna(skill):
            notes.append(f"**{m}**: the validation skill versus the naive rule is not available "
                         f"(using {scope}).{small}")
        elif skill <= 0:
            notes.append(f"**{m}** is not better than the naive (no change) rule on validation: its error is "
                         f"{abs(skill):,.1f}% {'higher' if skill < 0 else 'the same'} "
                         f"(MAE {ld.fmt_rs(r['MAE'])} against {ld.fmt_rs(r['naive_MAE'])}). "
                         f"Numbers taken from {scope}.{small}")
        elif pd.notna(r["n"]) and r["n"] < 100:
            notes.append(f"**{m}** beats the naive rule on validation ({skill:,.1f}% lower error, {scope}), "
                         f"but only {int(r['n']):,} validation rows were scored, so treat it with care.")
    return notes


def metrics_table(df, models):
    """MAE, MAPE, direction accuracy, rows and skill for each selected model and the naive rule."""
    naive = ld.row_metrics(df["actual"], df[ld.NAIVE_LABEL], df["price_today"]) if ld.NAIVE_LABEL in df else None
    rows = []
    for m in models:
        r = ld.row_metrics(df["actual"], df[m], df["price_today"])
        skill = (naive["MAE"] - r["MAE"]) / naive["MAE"] * 100 if naive and naive["MAE"] else float("nan")
        rows.append({"Model": m, "MAE (Rs)": r["MAE"], "MAPE (%)": r["MAPE_percent"],
                     "Direction accuracy (%)": r["direction_accuracy"] * 100, "Rows": r["n"],
                     "Skill vs naive (%)": skill})
    if naive:
        rows.append({"Model": ld.NAIVE_LABEL, "MAE (Rs)": naive["MAE"], "MAPE (%)": naive["MAPE_percent"],
                     "Direction accuracy (%)": float("nan"), "Rows": naive["n"],
                     "Skill vs naive (%)": float("nan")})
    return pd.DataFrame(rows)


def style_metrics(table):
    """green rows for models better than naive on MAE, red for worse (soft backgrounds, readable in both themes)."""
    good = "background-color: rgba(26,152,80,0.22)"
    bad = "background-color: rgba(215,48,39,0.22)"

    def colour(row):
        s = row["Skill vs naive (%)"]
        css = "" if pd.isna(s) else (good if s > 0 else bad)
        return [css] * len(row)

    return (table.style.apply(colour, axis=1)
            .format({"MAE (Rs)": "{:,.0f}", "MAPE (%)": "{:,.1f}", "Direction accuracy (%)": "{:,.1f}",
                     "Rows": "{:,.0f}", "Skill vs naive (%)": "{:+,.1f}"}, na_rep="-"))


def _base_layout(fig, title, y_title, height):
    fig.update_layout(
        title=dict(text=title, x=0.0, xanchor="left"),
        height=height, margin=dict(l=10, r=10, t=90, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#ffffff",
        hovermode="x unified", hoverlabel=dict(bgcolor="#ffffff", font=dict(color="#222222")),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        yaxis=dict(title=y_title, tickformat=",", gridcolor="#e3e3e3", zeroline=False),
        xaxis=dict(title="Date the price applies to", gridcolor="#e3e3e3", hoverformat="%d %b %Y"),
    )


def _shade_splits(fig, df, splits):
    """shade the validation and test periods that are inside the shown data."""
    for s in splits:
        part = df[df["split"] == s]
        if part.empty or s not in ld.SPLIT_SHADE:
            continue
        fig.add_vrect(x0=part["target_date"].min().to_pydatetime(), x1=part["target_date"].max().to_pydatetime(),
                      fillcolor=ld.SPLIT_SHADE[s], line_width=0, layer="below",
                      annotation_text=SPLIT_NAME[s], annotation_position="top left",
                      annotation_font=dict(size=11, color="#444444"))


def forecast_figure(df, models, splits, title):
    """actual price, model predictions and naive rule against the date the price applies to."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["target_date"], y=df["actual"], name="Actual price", mode="lines",
                             line=dict(color=ld.ACTUAL_COLOR, width=2.4), hovertemplate="%{y:,.0f}"))
    if ld.NAIVE_LABEL in df:
        fig.add_trace(go.Scatter(x=df["target_date"], y=df[ld.NAIVE_LABEL], name=ld.NAIVE_LABEL, mode="lines",
                                 line=dict(color=ld.NAIVE_COLOR, width=1.6, dash="dash"),
                                 hovertemplate="%{y:,.0f}"))
    for m in models:
        fig.add_trace(go.Scatter(x=df["target_date"], y=df[m], name=m, mode="lines",
                                 line=dict(color=ld.model_color(m), width=1.8), hovertemplate="%{y:,.0f}"))
    _shade_splits(fig, df, splits)
    _base_layout(fig, title, PRICE_AXIS, 560)
    fig.update_xaxes(
        rangeslider=dict(visible=True, thickness=0.07),
        rangeselector=dict(
            buttons=[dict(count=1, label="1M", step="month", stepmode="backward"),
                     dict(count=3, label="3M", step="month", stepmode="backward"),
                     dict(count=6, label="6M", step="month", stepmode="backward"),
                     dict(step="all", label="All")],
            bgcolor="#e8e8e8", activecolor="#bdbdbd", font=dict(color="#222222")))
    return fig


def error_figure(df, models, splits, title):
    """prediction minus actual over time, with a zero line."""
    fig = go.Figure()
    for m in models:
        fig.add_trace(go.Scatter(x=df["target_date"], y=df[m] - df["actual"], name=m, mode="lines",
                                 line=dict(color=ld.model_color(m), width=1.6),
                                 hovertemplate="%{y:+,.0f}"))
    fig.add_hline(y=0, line_color=ld.ACTUAL_COLOR, line_width=1.2)
    _shade_splits(fig, df, splits)
    _base_layout(fig, title, ERROR_AXIS, 380)
    return fig


def chart_config(filename):
    """plotly modebar settings: keep the png download button."""
    return {"displayModeBar": True, "displaylogo": False,
            "toImageButtonOptions": {"format": "png", "filename": filename, "scale": 2}}
