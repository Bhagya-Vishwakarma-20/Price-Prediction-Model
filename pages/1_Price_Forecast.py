"""price forecast page: saved predictions against the real prices, per commodity, level, district and horizon."""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st

from utils import forecast_helpers as fh
from utils import load_data as ld

st.set_page_config(page_title="Price Forecast", layout="wide")
st.title("Price Forecast")
st.caption("This page shows saved predictions only. Nothing is retrained and no model is loaded here.")

SPLIT_CHOICES = {"Validation": ["valid"], "Test": ["test"], "Both": ["valid", "test"]}


def sidebar_choices(comp):
    """read the filters from the sidebar; returns a dict, or None when nothing can be chosen."""
    commodities = ld.available_commodities()
    levels = ld.available_levels()
    if not commodities or not levels:
        ld.show_missing("predictions_<level>_<commodity>_h<horizon>.csv", "the forecast chart")
        return None
    st.sidebar.header("Filters")
    commodity = st.sidebar.selectbox("Commodity", commodities)
    level_label = st.sidebar.radio("Level", [ld.LEVEL_LABEL[lv] for lv in levels if lv in ld.LEVEL_LABEL])
    level = ld.LEVEL_FROM_LABEL[level_label]
    horizons = ld.available_horizons(level, commodity)
    if not horizons:
        st.info(f"No predictions files were found for {commodity} at the {level_label} level.")
        return None
    horizon = st.sidebar.selectbox("Forecast horizon (days ahead)", horizons)
    pred = ld.load_predictions(level, commodity, horizon)
    if pred is None:
        ld.show_missing(f"predictions_{level}_{commodity.lower()}_h{int(horizon)}.csv", "the forecast chart")
        return None
    district = None
    if level == "district":
        if "district" not in pred.columns:
            st.info("This predictions file has no district column.")
            return None
        names = sorted(pred["district"].dropna().unique())
        if not names:
            st.info("No districts were found in the predictions file.")
            return None
        district = st.sidebar.selectbox("District", names, format_func=lambda d: str(d).title())
        pred = pred[pred["district"] == district]
    available = ld.model_columns(pred)
    key = f"{level}-{commodity}-{horizon}-{district}"          # new default when the selection changes
    models = st.sidebar.multiselect("Models", available, key=f"models-{key}",
                                    default=fh.default_models(comp, level, commodity, horizon, district, available))
    split_name = st.sidebar.radio("Period to show", list(SPLIT_CHOICES), index=2)
    return {"commodity": commodity, "level": level, "level_label": level_label, "horizon": int(horizon),
            "district": district, "pred": pred, "models": models, "splits": SPLIT_CHOICES[split_name],
            "split_name": split_name}


def main():
    comp = ld.load_comparison()
    if comp is None:
        ld.show_missing("model_comparison.csv", "the validation warnings and the default model")
    sel = sidebar_choices(comp)
    if sel is None:
        return
    models, splits = sel["models"], sel["splits"]
    pred = sel["pred"]
    if not models:
        st.info("Pick at least one model in the sidebar to see its forecast.")
    if comp is not None:
        for note in fh.validation_warnings(comp, sel["level"], sel["commodity"], sel["horizon"], sel["district"],
                                           models):
            st.warning(note)

    df = pred[pred["split"].isin(splits)].dropna(subset=["target_date", "actual"]).sort_values("target_date")
    if df.empty:
        st.info("There are no rows for this selection and period. Try the other period or another horizon.")
        return

    where = sel["level_label"] + (f", {str(sel['district']).title()}" if sel["district"] else "")
    title = f"{sel['commodity']} price, {where}: actual and forecast {sel['horizon']} days ahead"
    file_stub = re.sub(r"[^a-z0-9]+", "_", f"{sel['commodity']}_{where}_h{sel['horizon']}".lower()).strip("_")

    st.plotly_chart(fh.forecast_figure(df, models, splits, title), width="stretch",
                    config=fh.chart_config(f"forecast_{file_stub}"))
    st.caption("Each point sits on the date the price applies to (the forecast day plus the horizon). "
               "Click a legend entry to hide or show a line.")

    if models:
        err_title = f"Forecast error over time ({sel['commodity']}, {where}, {sel['horizon']} days ahead)"
        st.plotly_chart(fh.error_figure(df, models, splits, err_title), width="stretch",
                        config=fh.chart_config(f"error_{file_stub}"))
        st.caption("Above zero means the forecast was higher than the real price; below zero means it was lower.")

    st.subheader(f"Accuracy on the shown period ({sel['split_name'].lower()})")
    table = fh.metrics_table(df, models)
    st.dataframe(fh.style_metrics(table), hide_index=True, width="stretch")
    st.caption("Green rows have a lower average error (MAE) than the naive rule that repeats today's price; "
               "red rows have a higher one. A no-change rule never predicts a direction, so it has no "
               "direction accuracy. Skill vs naive is the percentage drop in MAE compared with the naive rule.")


main()
