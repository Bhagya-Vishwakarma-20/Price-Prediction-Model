"""home page of the karnataka onion and tomato price forecasting dashboard.
run with: streamlit run app.py   (reads result files only, nothing is retrained)"""
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import load_data as ld  # noqa: E402

st.set_page_config(page_title="Karnataka Price Forecasts", page_icon=None, layout="wide")


def date_text(d):
    return "-" if d is None else f"{d:%d %b %Y}"


def show_cards(stats):
    """four summary cards, all numbers come from the result files."""
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Districts", ld.fmt_int(stats["districts"]) if stats["districts"] else "n/a")
    if stats["date_min"] is not None:
        c2.metric("Data range", f"{stats['date_min']:%b %Y} to {stats['date_max']:%b %Y}")
    else:
        c2.metric("Data range", "n/a")
    c3.metric("Models compared", ld.fmt_int(stats["models"]) if stats["models"] else "n/a")
    if stats["combos"]:
        c4.metric("Beat the naive rule on validation", f"{stats['beat']} of {stats['combos']}",
                  help="Model, commodity, level and horizon combinations whose validation error is "
                       "lower than the 'price stays the same' rule.")
    else:
        c4.metric("Beat the naive rule on validation", "n/a")


def keep_in_mind(stats, comp):
    """plain-language points taken from the real results."""
    points = []
    if stats["combos"]:
        share = stats["beat"] / stats["combos"]
        points.append(
            f"**The naive rule is the bar to beat.** It simply says the price N days from now equals today's "
            f"price. Only {stats['beat']} of {stats['combos']} model combinations ({share:.0%}) were better than it "
            f"on the validation period. A model that is not better than naive on validation is not worth trusting.")
    if stats["test_start"] is not None:
        rows = ""
        if stats["test_n_min"] is not None:
            rows = (f" Each test score is based on only {ld.fmt_int(stats['test_n_min'])} to "
                    f"{ld.fmt_int(stats['test_n_max'])} rows.")
        points.append(f"**The test period is short.** It runs from {date_text(stats['test_start'])} to "
                      f"{date_text(stats['test_end'])}.{rows} Treat test results as noisy, and be careful with "
                      f"models that look good on test but did not beat naive on validation.")
    if comp is not None:
        pairs = ld.weather_pairs(comp)
        points.append("**Weather:** " + ld.weather_verdict(pairs))
    if not points:
        st.info("The notes appear here once model_comparison.csv is available.")
        return
    with st.container(border=True):
        st.subheader("What to keep in mind")
        for p in points:
            st.markdown("- " + p)


def main():
    st.title("Karnataka onion and tomato price forecasts")
    st.write(
        "This dashboard compares forecasting models for daily mandi prices of onion and tomato in Karnataka "
        "(Agmarknet data, rupees per quintal). Models predict the price several days ahead, either for the "
        "Karnataka-wide average or for individual districts, and every model is judged against the simple "
        "rule that the price will not change. It only reads saved results; nothing is retrained here.")

    comp = ld.load_comparison()
    if comp is None:
        ld.show_missing("model_comparison.csv", "the summary cards and the model comparison")
    stats = ld.summary_stats()
    if not ld.available_commodities() and comp is None:
        st.error("No result files were found. Check the OUTPUT_DIR setting in utils/load_data.py "
                 f"(currently '{ld.OUTPUT_DIR}').")
        return
    show_cards(stats)
    st.write("")
    keep_in_mind(stats, comp)

    st.write("")
    left, right = st.columns(2)
    with left:
        st.page_link("pages/1_Price_Forecast.py", label="Price forecast: actual against predicted prices")
    with right:
        st.page_link("pages/2_Model_Comparison.py", label="Model comparison: tables, winners and charts")

    report = ld.load_report_text()
    with st.expander("Full training report (training_report.txt)"):
        if report is None:
            ld.show_missing("training_report.txt")
        else:
            st.code(report, language=None)


main()
