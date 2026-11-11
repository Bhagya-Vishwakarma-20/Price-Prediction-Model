"""model comparison page: tables first, charts second. reads result files only."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils import comparison_helpers as ch
from utils import load_data as ld

st.set_page_config(page_title="Model comparison", layout="wide")

CHART_FONT = dict(color="#222222", size=13)


# ---------------------------------------------------------------- small helpers
def chart_layout(fig, title, xtitle, ytitle, height=450):
    """light plot area so colours read the same in light and dark themes."""
    fig.update_layout(template="plotly_white", title=title, height=height, font=CHART_FONT,
                      paper_bgcolor="white", plot_bgcolor="white", margin=dict(l=10, r=10, t=60, b=10),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
    fig.update_xaxes(title=xtitle)
    fig.update_yaxes(title=ytitle)
    return fig


def show_chart(fig, key):
    st.plotly_chart(fig, width="stretch", theme=None, key=key)


def pick(label, options, key, default=None):
    """selectbox that copes with an empty option list."""
    options = list(options)
    if not options:
        st.info(f"Nothing to choose for '{label}'.")
        return None
    index = options.index(default) if default in options else 0
    return st.selectbox(label, options, index=index, key=key)


# ---------------------------------------------------------------- load
st.title("Model comparison")
comp = ld.load_comparison()
if comp is None or comp.empty:
    ld.show_missing("model_comparison.csv", "every table and chart on this page")
    st.stop()

commodities = sorted(comp["commodity"].dropna().unique())
level_labels = [ld.LEVEL_LABEL[k] for k in ("state", "district") if k in set(comp["level"])] or \
    sorted(comp["level_label"].unique())
horizons_all = sorted(int(h) for h in comp["horizon"].dropna().unique())
families = sorted(comp["model_family"].unique())

tabs = st.tabs(["All results", "Winners", "Weather: with vs without", "By district",
                "Feature importance", "Charts"])


# ---------------------------------------------------------------- 1. all results
def tab_all_results():
    st.caption("Every model, commodity, horizon and period. MAE is the average miss in Rs per quintal "
               "(lower is better). Click a column heading to sort.")
    c1, c2, c3, c4, c5 = st.columns(5)
    f_com = c1.multiselect("Commodity", commodities, default=commodities, key="all_com")
    f_lev = c2.multiselect("Level", level_labels, default=level_labels, key="all_lev")
    f_hor = c3.multiselect("Horizon (days ahead)", horizons_all, default=horizons_all, key="all_hor")
    f_spl = c4.multiselect("Period", ["valid", "test"], default=["valid", "test"], key="all_spl",
                           format_func=ch.splits_label)
    f_fam = c5.multiselect("Model type", families, default=families, key="all_fam")
    text = st.text_input("Search model or district", key="all_search", placeholder="for example: kolar or XGBoost")
    show_settings = st.checkbox("Show the settings column (long JSON text)", value=False, key="all_settings")

    sel = ch.filter_results(comp, f_com, f_lev, f_hor, f_spl, f_fam, text)
    if sel.empty:
        st.info("No rows match these filters. Widen the filters or clear the search box.")
        return
    view, best = ch.results_view(sel, show_settings)
    st.dataframe(ch.style_results(view, best), width="stretch", hide_index=True, height=520)
    st.caption(f"{len(sel):,} rows shown. Green = lowest MAE within the same level, commodity, horizon, "
               f"period and district among the rows shown.")
    st.download_button("Download this table as CSV", sel.drop(columns=["model_family", "level_label"]).to_csv(index=False),
                       file_name="model_comparison_filtered.csv", mime="text/csv", key="all_download")
    if "settings" in sel.columns:
        with st.expander("Model settings (JSON) for the rows shown"):
            s = sel[["level_label", "commodity", "horizon", "split", "district", "model", "settings"]]
            st.dataframe(s.rename(columns=ch.NAMES), width="stretch", hide_index=True)


# ---------------------------------------------------------------- 2. winners
def tab_winners():
    st.caption("For each level, commodity and horizon the model is chosen by its error on the validation "
               "period (whole-level rows only). Test columns show how that choice did on unseen data.")
    w = ch.winners_table(comp)
    if w.empty:
        st.info("No validation results are available to pick winners from.")
        return
    fmt = {"Validation MAE (Rs per quintal)": "{:,.1f}", "Validation skill vs naive (%)": "{:+.1f}%",
           "Test MAE (Rs per quintal)": "{:,.1f}", "Naive test MAE (Rs per quintal)": "{:,.1f}",
           "Test difference vs naive (%)": "{:+.1f}%"}
    sty = w.style.format(fmt, na_rep="-")
    sty = ch.style_map(sty, ch.color_sign(True), ["Test difference vs naive (%)", "Validation skill vs naive (%)"])
    sty = ch.style_map(sty, ch.color_yes_no, ["Beat naive on validation?", "Beat naive on test?"])
    st.dataframe(sty, width="stretch", hide_index=True)
    st.markdown(ch.winners_sentence(w))
    st.caption("Naive rule = predict that the price stays at today's value. Green = better than naive, red = worse.")


# ---------------------------------------------------------------- 3. weather
def tab_weather():
    st.caption("Each model that uses weather is placed next to the same kind of model without weather. "
               "A negative change means weather lowered the error.")
    pairs = ch.weather_table(comp)
    if pairs is None or pairs.empty:
        st.info("There are no with/without-weather pairs in the results.")
        return
    view = ch.weather_view(pairs)
    fmt = {c: "{:,.1f}" for c in view.columns if "MAE" in c}
    fmt.update({c: "{:+.1f}%" for c in view.columns if c.endswith("change (%)")})
    sty = view.style.format(fmt, na_rep="-")
    change_cols = [c for c in view.columns if c.endswith("change (%)")]
    if change_cols:
        sty = ch.style_map(sty, ch.color_sign(False), change_cols)
    st.dataframe(sty, width="stretch", hide_index=True)
    st.markdown(ld.weather_verdict(pairs))
    st.caption("Green = weather lowered the error, red = weather raised it.")


# ---------------------------------------------------------------- 4. by district
def tab_district():
    st.caption("District-level models only. One row per district for the chosen model.")
    d_all = comp[(comp["level"] == "district") & (comp["district"] != "ALL")]
    if d_all.empty:
        st.info("The results contain no per-district rows.")
        return
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        commodity = pick("Commodity", sorted(d_all["commodity"].unique()), "dist_com")
    with c2:
        hs = sorted(int(h) for h in d_all[d_all["commodity"] == commodity]["horizon"].unique())
        horizon = pick("Horizon (days ahead)", hs, "dist_hor")
    with c3:
        split = pick("Period", [s for s in ("valid", "test") if s in set(d_all["split"])], "dist_spl",
                     default="test")
    if commodity is None or horizon is None or split is None:
        return
    rows = ch.district_rows(comp, commodity, horizon, split)
    models = sorted(rows["model"].unique())
    default = ch.default_district_model(comp, commodity, horizon)
    with c4:
        # the key includes the choices so the default is recomputed when they change
        model = pick("Model", models, f"dist_mod_{commodity}_{horizon}", default=default)
    if model is None:
        st.info("No district results for this choice.")
        return
    d = rows[rows["model"] == model]
    if d.empty:
        st.info("No district results for this choice.")
        return
    st.caption(f"Default model = lowest validation MAE for the whole level: {default}.")
    table = d[["district", "n", "MAE", "MAPE_percent", "naive_MAE", "skill_vs_naive_percent"]].sort_values("MAE")
    table = table.rename(columns={"district": "District", "n": "Predictions (n)", "MAE": "MAE (Rs per quintal)",
                                  "MAPE_percent": "MAPE (%)", "naive_MAE": "Naive MAE (Rs per quintal)",
                                  "skill_vs_naive_percent": "Skill vs naive (%)"}).reset_index(drop=True)
    sty = table.style.format({"Predictions (n)": "{:,.0f}", "MAE (Rs per quintal)": "{:,.1f}", "MAPE (%)": "{:.1f}%",
                              "Naive MAE (Rs per quintal)": "{:,.1f}", "Skill vs naive (%)": "{:+.1f}%"}, na_rep="-")
    sty = ch.style_map(sty, ch.color_sign(True), ["Skill vs naive (%)"])
    st.dataframe(sty, width="stretch", hide_index=True)

    low, high = ch.lowest_highest(d, 10)
    color = ld.model_color(model)
    fig = go.Figure()
    split_name = ch.splits_label(split).lower()
    if high.empty:
        fig.add_bar(y=low["district"], x=low["MAE"], orientation="h", marker_color=color,
                    name="All districts", text=low["MAE"].map(lambda v: f"{v:,.0f}"), textposition="outside")
        order = list(low["district"])
        title = f"Average miss by district ({split_name} period) - {model}"
    else:
        fig.add_bar(y=low["district"], x=low["MAE"], orientation="h", marker_color=color,
                    name=f"{len(low)} lowest error", text=low["MAE"].map(lambda v: f"{v:,.0f}"), textposition="outside")
        fig.add_bar(y=high["district"], x=high["MAE"], orientation="h", marker_color=color, opacity=0.5,
                    name=f"{len(high)} highest error", text=high["MAE"].map(lambda v: f"{v:,.0f}"),
                    textposition="outside")
        order = list(low["district"]) + list(high["district"])
        title = f"Best and worst districts by average miss ({split_name} period) - {model}"
    fig = chart_layout(fig, title, "Average miss, MAE (Rs per quintal, lower is better)", "District",
                       height=max(380, 28 * len(order) + 140))
    fig.update_yaxes(categoryorder="array", categoryarray=order[::-1])
    fig.update_xaxes(rangemode="tozero")
    show_chart(fig, "dist_chart")


# ---------------------------------------------------------------- 5. feature importance
def tab_importance():
    st.caption("Permutation importance is the increase in error when a feature is shuffled; a bigger value means "
               "the model leans on that feature more. Built-in importance is the model's own internal score.")
    c1, c2 = st.columns(2)
    with c1:
        commodity = pick("Commodity", commodities, "fi_com")
    with c2:
        horizon = pick("Horizon (days ahead)", horizons_all, "fi_hor")
    if commodity is None or horizon is None:
        return
    fi = ld.load_importance(commodity, horizon)
    if fi is None or fi.empty:
        ld.show_missing(f"feature_importance_{commodity.lower()}_h{int(horizon)}.csv", "this tab")
        return
    c3, c4, c5 = st.columns(3)
    has_model, has_set = "model" in fi.columns, "featureset" in fi.columns
    with c3:
        model = pick("Model", sorted(fi["model"].unique()) if has_model else [], "fi_mod")
    with c4:
        fset = pick("Feature set", sorted(fi["featureset"].unique()) if has_set else [], "fi_set")
    with c5:
        kinds = [k for k, col in (("Permutation", "permutation_importance_mean"),
                                  ("Built-in", "builtin_importance")) if col in fi.columns]
        kind = pick("Importance type", kinds, "fi_kind")
    if model is None or fset is None or kind is None:
        return
    sel = fi[(fi["model"] == model) & (fi["featureset"] == fset)]
    col = "permutation_importance_mean" if kind == "Permutation" else "builtin_importance"
    sel = sel.dropna(subset=[col])
    if sel.empty:
        st.info("No importance values for this choice.")
        return
    top = sel.sort_values(col, ascending=False).head(15)
    color = ch.importance_color(comp, model, fset)
    err = None
    if kind == "Permutation" and "permutation_importance_std" in top.columns:
        err = dict(type="data", array=top["permutation_importance_std"].fillna(0), color="#555555")
    fig = go.Figure(go.Bar(y=top["feature"], x=top[col], orientation="h", marker_color=color, error_x=err,
                           hovertemplate="%{y}: %{x:.4f}<extra></extra>"))
    xt = ("Increase in error when the feature is shuffled (permutation importance)" if kind == "Permutation"
          else "Built-in importance (share of the model's splits)")
    fig = chart_layout(fig, f"Top {len(top)} features - {model}, {fset} features, {commodity}, {horizon} days ahead",
                       xt, "Feature", height=max(380, 30 * len(top) + 140))
    fig.update_yaxes(categoryorder="array", categoryarray=list(top["feature"])[::-1])
    show_chart(fig, "fi_chart")
    if kind == "Permutation":
        st.caption("Permutation importance = increase in error when a feature is shuffled. "
                   "Error bars show the spread over repeated shuffles.")
    names = {"feature": "Feature", "builtin_importance": "Built-in importance",
             "permutation_importance_mean": "Permutation importance (mean)",
             "permutation_importance_std": "Permutation importance (std)"}
    full = sel.sort_values(col, ascending=False)[[c for c in names if c in sel.columns]].rename(columns=names)
    with st.expander(f"Full table ({len(full):,} features)", expanded=True):
        st.dataframe(full.style.format({c: "{:.4f}" for c in full.columns if c != "Feature"}, na_rep="-"),
                     width="stretch", hide_index=True)


# ---------------------------------------------------------------- 6. charts
def tab_charts():
    st.markdown("**Test error of every model against the naive rule**")
    c1, c2, c3 = st.columns(3)
    with c1:
        commodity = pick("Commodity", commodities, "ch_com")
    with c2:
        level_label = pick("Level", level_labels, "ch_lev")
    if commodity is None or level_label is None:
        return
    level = ld.LEVEL_FROM_LABEL.get(level_label, level_label)
    hs = sorted(int(h) for h in comp[(comp["level"] == level) & (comp["commodity"] == commodity)]["horizon"].unique())
    with c3:
        h_choice = pick("Horizon (days ahead)", ["All horizons"] + hs, "ch_hor")
    if h_choice is None:
        return
    chosen = hs if h_choice == "All horizons" else [h_choice]
    o, naive = ch.test_mae_by_model(comp, commodity, level, chosen)
    if o.empty:
        st.info("No test results for this choice.")
    else:
        fig = go.Figure()
        for m in sorted(o["model"].unique()):
            g = o[o["model"] == m].sort_values("horizon")
            fig.add_bar(x=[f"{int(h)} days" for h in g["horizon"]], y=g["MAE"], name=m, marker_color=ld.model_color(m),
                        hovertemplate="%{x}<br>" + m + ": Rs %{y:,.0f}<extra></extra>")
        fig.add_bar(x=[f"{int(h)} days" for h in naive.index], y=naive.values, name=ld.NAIVE_LABEL,
                    marker_color=ld.NAIVE_COLOR, marker_pattern_shape="/",
                    hovertemplate="%{x}<br>Naive: Rs %{y:,.0f}<extra></extra>")
        fig.update_layout(barmode="group")
        fig = chart_layout(fig, f"Test-period error by model - {commodity}, {level_label}",
                           "How far ahead the price is predicted", "Average miss, MAE (Rs per quintal, lower is better)")
        show_chart(fig, "ch_bars")
        st.caption("Any model bar taller than the grey hatched naive bar made a bigger error than "
                   "'the price stays the same'.")

    st.divider()
    st.markdown("**Skill against the naive rule**")
    d1, d2, d3 = st.columns(3)
    with d1:
        commodity2 = pick("Commodity", commodities, "hm_com")
    with d2:
        level2_label = pick("Level", level_labels, "hm_lev")
    with d3:
        split = pick("Period", [s for s in ("valid", "test") if s in set(comp["split"])], "hm_spl", default="test")
    if commodity2 is None or level2_label is None or split is None:
        return
    level2 = ld.LEVEL_FROM_LABEL.get(level2_label, level2_label)
    mat = ch.skill_matrix(comp, commodity2, level2, split)
    if mat.empty:
        st.info("No results for this choice.")
        return
    zmax = max(float(mat.abs().max().max()), 1.0)
    fig = go.Figure(go.Heatmap(
        z=mat.values, x=[f"{int(h)} days" for h in mat.columns], y=list(mat.index),
        colorscale=[[0, ld.BAD_COLOR], [0.5, "#ffffff"], [1, ld.GOOD_COLOR]], zmid=0, zmin=-zmax, zmax=zmax,
        text=mat.round(1).astype(str) + "%", texttemplate="%{text}", textfont=dict(color="#111111"),
        colorbar=dict(title="Skill (%)"), hovertemplate="%{y}<br>%{x}: %{z:.1f}%<extra></extra>"))
    fig = chart_layout(fig, f"Skill against the naive rule - {commodity2}, {level2_label}, {ch.splits_label(split).lower()} period",
                       "How far ahead the price is predicted", "Model", height=max(300, 55 * len(mat) + 160))
    show_chart(fig, "hm_chart")
    st.caption("Green = better than predicting no change, red = worse. Values are the percentage change in "
               "average miss compared with the naive rule.")


for tab, fn in zip(tabs, (tab_all_results, tab_winners, tab_weather, tab_district, tab_importance, tab_charts)):
    with tab:
        try:
            fn()
        except Exception as exc:   # one broken tab must not take the page down
            st.warning(f"This tab could not be drawn ({type(exc).__name__}: {exc}).")
