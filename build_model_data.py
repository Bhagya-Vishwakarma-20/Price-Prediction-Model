"""build modelling datasets: cleaned karnataka onion/tomato prices joined with daily weather.

install:
    pip install pandas numpy matplotlib

run:
    python build_model_data.py                              # uses PRICE_FILE below
    python build_model_data.py karnataka_onion_tomato.csv   # or pass the price file name (inside output/)

input files are only read, never modified. everything is written to output/.
"""
import math
import os
import random
import sys

import matplotlib

matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------- settings
OUTPUT_DIR = "output"
PRICE_FILE = "temp"                                    # cleaned price csv inside OUTPUT_DIR
WEATHER_LONG_FILE = "output/weather_daily_long.csv"    # date, location, district_original, weather...
COMMODITIES = ["Onion", "Tomato"]
PRICE_AGG = "mean"            # "mean" or "median" of Modal_Price over markets, varieties, grades
MIN_COVERAGE = 0.5            # keep a district-commodity pair with a price on >= this share of days
MIN_DISTRICTS_REPORTING = 5   # state series: keep a day only if >= this many districts have a price
MAX_FILL_DAYS = 3             # carry the last price forward over gaps of up to this many days
HORIZONS = [7, 14, 30]
TRAIN_END = "2024-12-31"
VALID_END = "2025-12-31"
START_DATE = "2021-01-01"

LAGS = [1, 7, 14, 30]
ROLL_WINDOWS = [7, 30]
ROLL_MIN_PERIODS = {7: 4, 30: 15}   # price rolling stats need at least this many real/filled days
RAIN_COL = "rain_sum"               # weather columns used for the weather features
TEMP_COL = "temperature_2m_mean"
WEATHER_FEATURES = ["rain_sum_7d", "rain_sum_30d", "temp_mean_7d", "temp_mean_30d"]
KS_WEATHER_FEATURES = ["ks_" + c for c in WEATHER_FEATURES]
CALENDAR_COLS = ["month", "week_of_year", "day_of_week", "doy_sin", "doy_cos"]
RANDOM_SEED = 42


def log(msg=""):
    print(msg, flush=True)


# ---------------------------------------------------------------- reading
def normalise(series):
    return series.astype(str).str.strip().str.lower()


def read_prices(path):
    """cleaned price csv -> rows for the wanted commodities with a normalised district."""
    if not os.path.isfile(path):
        sys.exit(f"price file '{path}' not found. Set PRICE_FILE at the top of the script "
                 "or pass the file name (inside output/) as an argument.")
    df = pd.read_csv(path, usecols=["District", "Market", "Commodity", "Arrival_Date", "Modal_Price"],
                     dtype={"District": str, "Market": str, "Commodity": str})
    wanted = {c.strip().casefold(): c for c in COMMODITIES}
    key = df["Commodity"].str.strip().str.casefold()
    df = df[key.isin(wanted)].copy()
    df["commodity"] = key[df.index].map(wanted)
    df["district"] = normalise(df["District"])
    # the same market appears with and without an " APMC" suffix, so count them as one
    df["market_key"] = (df["Market"].str.strip().str.lower()
                        .str.replace(r"\s+apmc$", "", regex=True))
    df["date"] = pd.to_datetime(df["Arrival_Date"], format="%Y-%m-%d")
    df["modal"] = pd.to_numeric(df["Modal_Price"], errors="coerce")
    df = df.dropna(subset=["modal"])
    log(f"read {len(df):,} price rows for {', '.join(wanted.values())} from {path}")
    return df[["district", "commodity", "date", "market_key", "modal"]]


def read_weather(path):
    if not os.path.isfile(path):
        sys.exit(f"weather file '{path}' not found. Run fetch_weather.py first.")
    w = pd.read_csv(path, parse_dates=["date"])
    w["location"] = normalise(w["location"])
    for col in (RAIN_COL, TEMP_COL):
        if col not in w.columns:
            sys.exit(f"weather file has no column '{col}'")
    log(f"read weather for {w['location'].nunique()} locations, "
        f"{w['date'].min():%Y-%m-%d} to {w['date'].max():%Y-%m-%d}")
    return w


# ---------------------------------------------------------------- districts, daily prices, coverage
def drop_districts_without_weather(prices, weather):
    have = set(weather["location"])
    missing = sorted(set(prices["district"]) - have)
    for d in missing:
        n = int((prices["district"] == d).sum())
        log(f"NO WEATHER DATA for district '{d}' ({n:,} price rows), dropping it")
    return prices[prices["district"].isin(have)].copy(), missing


def aggregate_daily(prices):
    """one price per district, commodity and date, plus how many markets reported."""
    return (prices.groupby(["district", "commodity", "date"])
            .agg(price=("modal", PRICE_AGG), n_markets=("market_key", "nunique"))
            .reset_index())


def coverage_table(daily, calendar):
    rows = []
    for (district, commodity), g in daily.groupby(["district", "commodity"]):
        dates = g["date"].sort_values()
        longest = int(dates.diff().dt.days.sub(1).max()) if len(dates) > 1 else 0
        share = len(dates) / len(calendar)
        rows.append({"district": district, "commodity": commodity,
                     "first_date": dates.iloc[0].strftime("%Y-%m-%d"),
                     "last_date": dates.iloc[-1].strftime("%Y-%m-%d"),
                     "days_with_price": len(dates), "calendar_days": len(calendar),
                     "share_covered": round(share, 4), "longest_gap_days": longest,
                     "kept": share >= MIN_COVERAGE})
    return pd.DataFrame(rows).sort_values(["commodity", "district"]).reset_index(drop=True)


def fill_pair(g, calendar):
    """full daily calendar for one pair; short gaps filled by carrying the last price forward."""
    s = g.set_index("date").reindex(calendar)
    filled = s["price"].ffill(limit=MAX_FILL_DAYS)
    return pd.DataFrame({"price": filled,
                         "n_markets": s["n_markets"].fillna(0).astype(int),
                         "price_was_filled": (s["price"].isna() & filled.notna()).astype(int)})


# ---------------------------------------------------------------- features
def calendar_features(index):
    doy = index.dayofyear
    return pd.DataFrame({"month": index.month,
                         "week_of_year": index.isocalendar().week.astype(int).values,
                         "day_of_week": index.dayofweek,
                         "doy_sin": np.sin(2 * math.pi * doy / 365.25),
                         "doy_cos": np.cos(2 * math.pi * doy / 365.25)}, index=index)


def price_features(price):
    """lags and rolling stats; every value at day t uses price on days <= t only."""
    out = pd.DataFrame(index=price.index)
    for k in LAGS:
        out[f"price_lag_{k}"] = price.shift(k)
    for w in ROLL_WINDOWS:
        roll = price.rolling(w, min_periods=ROLL_MIN_PERIODS[w])
        out[f"price_roll_mean_{w}"] = roll.mean()
        out[f"price_roll_std_{w}"] = roll.std()
    return out


def add_targets(df, price):
    """price N days after day t (shift backwards), and the relative change from day t."""
    for h in HORIZONS:
        df[f"target_h{h}"] = price.shift(-h)
        df[f"target_change_h{h}"] = df[f"target_h{h}"] / price - 1
    return df


def weather_feature_table(weather, districts):
    """rolling rain sums and temperature means per district (trailing windows, include day t)."""
    parts = []
    for loc, g in weather[weather["location"].isin(districts)].groupby("location"):
        g = g.sort_values("date").set_index("date")
        f = pd.DataFrame(index=g.index)
        for w in ROLL_WINDOWS:
            f[f"rain_sum_{w}d"] = g[RAIN_COL].rolling(w, min_periods=w).sum()
            f[f"temp_mean_{w}d"] = g[TEMP_COL].rolling(w, min_periods=w).mean()
        f["district"] = loc
        parts.append(f.reset_index())
    return pd.concat(parts, ignore_index=True)


def split_labels(dates):
    return pd.Series(np.select([dates <= pd.Timestamp(TRAIN_END), dates <= pd.Timestamp(VALID_END)],
                               ["train", "valid"], default="test"), index=dates.index)


# ---------------------------------------------------------------- build tables
def build_district_table(commodity, pairs, wfeat, ks_wfeat, district_codes):
    """pairs: {district: filled dataframe}. returns one row per district and date with a price."""
    parts = []
    for district, f in pairs.items():
        t = pd.concat([f, price_features(f["price"]), calendar_features(f.index)], axis=1)
        t = add_targets(t, f["price"])
        t.index.name = "date"
        t["district"] = district
        parts.append(t.reset_index())
    table = pd.concat(parts, ignore_index=True)
    table["commodity"] = commodity
    table["district_code"] = table["district"].map(district_codes)
    # karnataka-wide average of the t-7 price over the kept districts
    table["ks_price_lag_7"] = table.groupby("date")["price_lag_7"].transform("mean")
    table = table.merge(wfeat, on=["district", "date"], how="left")
    table = table.merge(ks_wfeat, on="date", how="left")
    table = table.dropna(subset=["price"])                 # no price on day t, even after filling
    table["split"] = split_labels(table["date"])
    table = table.sort_values(["district", "date"]).reset_index(drop=True)
    first = ["date", "district", "district_code", "commodity", "price", "n_markets", "price_was_filled"]
    price_cols = [f"price_lag_{k}" for k in LAGS] + \
                 [f"price_roll_{s}_{w}" for w in ROLL_WINDOWS for s in ("mean", "std")]
    target_cols = [c for h in HORIZONS for c in (f"target_h{h}", f"target_change_h{h}")]
    cols = (first + price_cols + WEATHER_FEATURES + KS_WEATHER_FEATURES + ["ks_price_lag_7"]
            + CALENDAR_COLS + target_cols + ["split"])
    return table[cols]


def build_state_table(commodity, pairs, calendar, daily_weather_avg, ks_wfeat):
    """karnataka-wide series: mean of the filled district prices on days with enough districts."""
    prices = pd.DataFrame({d: f["price"] for d, f in pairs.items()}).reindex(calendar)
    n_districts = prices.notna().sum(axis=1)
    state_price = prices.mean(axis=1).where(n_districts >= MIN_DISTRICTS_REPORTING)
    t = pd.DataFrame({"price": state_price, "n_districts": n_districts}, index=calendar)
    t = pd.concat([t, price_features(state_price), calendar_features(calendar)], axis=1)
    t = add_targets(t, state_price)
    t.index.name = "date"
    t = t.reset_index()
    t["commodity"] = commodity
    t = t.merge(daily_weather_avg, on="date", how="left")
    t = t.merge(ks_wfeat.rename(columns={"ks_" + c: c for c in WEATHER_FEATURES}), on="date", how="left")
    t = t.dropna(subset=["price"])
    t["split"] = split_labels(t["date"])
    first = ["date", "commodity", "price", "n_districts"]
    price_cols = [f"price_lag_{k}" for k in LAGS] + \
                 [f"price_roll_{s}_{w}" for w in ROLL_WINDOWS for s in ("mean", "std")]
    target_cols = [c for h in HORIZONS for c in (f"target_h{h}", f"target_change_h{h}")]
    cols = (first + ["rain_sum", "temp_mean"] + WEATHER_FEATURES + price_cols
            + CALENDAR_COLS + target_cols + ["split"])
    return t[cols].reset_index(drop=True), state_price


# ---------------------------------------------------------------- scores and correlation
def score_rows(df, level, commodity, district):
    """naive rule: price at t+N equals price at t."""
    rows = []
    for h in HORIZONS:
        for split in ("valid", "test"):
            d = df[(df["split"] == split) & df[f"target_h{h}"].notna() & (df[f"target_h{h}"] > 0)]
            err = (d[f"target_h{h}"] - d["price"]).abs()
            rows.append({"level": level, "commodity": commodity, "district": district,
                         "horizon": h, "split": split, "n": len(d),
                         "MAE": round(err.mean(), 2) if len(d) else np.nan,
                         "MAPE_percent": round((err / d[f"target_h{h}"]).mean() * 100, 2) if len(d) else np.nan})
    return rows


def baseline_scores(district_tables, state_tables):
    rows = []
    for commodity, t in district_tables.items():
        rows += score_rows(t, "district", commodity, "ALL")
        for district, g in t.groupby("district"):
            rows += score_rows(g, "district", commodity, district)
    for commodity, t in state_tables.items():
        rows += score_rows(t, "state", commodity, "ALL")
    return pd.DataFrame(rows)


def weather_correlation(district_tables):
    """correlation of each weather feature with target_change_h14, training rows only."""
    rows = []
    for commodity, t in district_tables.items():
        train = t[(t["split"] == "train") & t["target_change_h14"].notna()]
        part = []
        for feat in WEATHER_FEATURES + KS_WEATHER_FEATURES:
            ok = train[[feat, "target_change_h14"]].dropna()
            part.append({"commodity": commodity, "feature": feat,
                         "correlation": round(ok[feat].corr(ok["target_change_h14"]), 4),
                         "n": len(ok)})
        part.sort(key=lambda r: -abs(r["correlation"]) if pd.notna(r["correlation"]) else 0)
        rows += part
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- plot
def plot_state_series(commodity, table):
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(table["date"], table["price"], color="#1f3a5f", linewidth=1)
    lo, hi = table["date"].min(), table["date"].max()
    t_end, v_end = pd.Timestamp(TRAIN_END), pd.Timestamp(VALID_END)
    ax.axvspan(lo, min(t_end, hi), color="#4c9f70", alpha=0.15, label="train")
    if hi > t_end:
        ax.axvspan(t_end, min(v_end, hi), color="#e0a030", alpha=0.20, label="valid")
    if hi > v_end:
        ax.axvspan(v_end, hi, color="#c0504d", alpha=0.18, label="test")
    ax.set_title(f"Karnataka-wide daily {commodity.lower()} price ({PRICE_AGG} modal price, Rs/quintal)")
    ax.set_ylabel("Rs per quintal")
    ax.legend(loc="upper left")
    ax.margins(x=0)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, f"state_series_{commodity.lower()}.png"), dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------- checks
def recompute_by_hand(t, price_s, rain_s):
    """recompute features for day t from data cut off at day t (plain python, no pandas rolling)."""
    price_s = price_s[price_s.index <= t]     # nothing after day t is visible from here
    rain_s = rain_s[rain_s.index <= t]
    day = pd.Timedelta(days=1)
    lag7 = price_s.get(t - 7 * day, np.nan)
    window7 = [price_s.get(t - i * day, np.nan) for i in range(7)]
    window30 = [price_s.get(t - i * day, np.nan) for i in range(30)]
    window7 = [v for v in window7 if pd.notna(v)]
    window30 = [v for v in window30 if pd.notna(v)]
    roll_mean_30 = sum(window30) / len(window30) if len(window30) >= ROLL_MIN_PERIODS[30] else np.nan
    roll_std_7 = (float(np.std(window7, ddof=1)) if len(window7) >= ROLL_MIN_PERIODS[7] else np.nan)
    rain30 = [rain_s.get(t - i * day, np.nan) for i in range(30)]
    rain_sum_30 = sum(rain30) if not any(pd.isna(rain30)) else np.nan
    return {"price_lag_7": lag7, "price_roll_mean_30": roll_mean_30,
            "price_roll_std_7": roll_std_7, "rain_sum_30d": rain_sum_30}


def verify_no_lookahead(label, table, price_lookup, rain_lookup, n=3):
    """pick random rows, recompute features from data up to day t, compare with the table."""
    rng = random.Random(RANDOM_SEED)
    needed = ["price_lag_7", "price_roll_mean_30", "price_roll_std_7", "rain_sum_30d"]
    pool = table.dropna(subset=needed)
    picks = pool.iloc[[rng.randrange(len(pool)) for _ in range(n)]]
    all_ok = True
    for _, row in picks.iterrows():
        key = row["district"] if "district" in row.index else None
        mine = recompute_by_hand(row["date"], price_lookup(key, row["commodity"]), rain_lookup(key))
        parts = []
        for col, value in mine.items():
            ok = bool(np.isclose(value, row[col], rtol=1e-9, atol=1e-9, equal_nan=True))
            all_ok &= ok
            parts.append(f"{col}: table={row[col]:.4f} hand={value:.4f} {'ok' if ok else 'MISMATCH'}")
        who = f"{key}, " if key else ""
        log(f"  {label} {who}{row['commodity']}, day t={row['date']:%Y-%m-%d} (data cut at t)")
        for p in parts:
            log(f"      {p}")
        # the target must be exactly the price h days later
        t = row["date"]
        full = price_lookup(key, row["commodity"])
        h = HORIZONS[0]
        later = full.get(t + pd.Timedelta(days=h), np.nan)
        ok = bool(np.isclose(later, row[f"target_h{h}"], equal_nan=True))
        all_ok &= ok
        log(f"      target_h{h}: table={row[f'target_h{h}']:.4f} price at t+{h}={later:.4f} {'ok' if ok else 'MISMATCH'}")
    log(f"  {label}: {'all recomputed features use only data up to day t' if all_ok else 'PROBLEM FOUND'}")
    return all_ok


def split_counts(level, commodity, table):
    rows = []
    for h in HORIZONS:
        ok = table[table[f"target_h{h}"].notna()]
        counts = ok["split"].value_counts()
        rows.append({"level": level, "commodity": commodity, "horizon": h,
                     "train": int(counts.get("train", 0)), "valid": int(counts.get("valid", 0)),
                     "test": int(counts.get("test", 0))})
    return rows


# ---------------------------------------------------------------- report
def missing_values_text(name, table):
    miss = table.isna().sum()
    miss = miss[miss > 0]
    if not len(miss):
        return [f"  {name}: no missing values"]
    lines = [f"  {name} ({len(table):,} rows):"]
    for col, n in miss.items():
        lines.append(f"      {col}: {n:,} ({n / len(table):.1%})")
    return lines


def write_report(path, coverage, dropped_no_weather, last_date, n_calendar, district_tables,
                 state_tables, filled_counts, counts_df, checks_ok):
    L = ["DATASET REPORT", "",
         f"Calendar: {START_DATE} to {last_date:%Y-%m-%d} ({n_calendar:,} days)",
         f"Price aggregation: {PRICE_AGG} of Modal_Price; MIN_COVERAGE={MIN_COVERAGE}; "
         f"MAX_FILL_DAYS={MAX_FILL_DAYS}; MIN_DISTRICTS_REPORTING={MIN_DISTRICTS_REPORTING}",
         f"Split: train <= {TRAIN_END}, valid <= {VALID_END}, test after that", "",
         f"Districts dropped for missing weather: {', '.join(dropped_no_weather) or 'none'}", ""]
    L.append("Coverage by district-commodity pair (share of calendar days with a price)")
    for commodity in COMMODITIES:
        c = coverage[coverage["commodity"] == commodity]
        kept = c[c["kept"]]
        dropped = c[~c["kept"]]
        L.append(f"  {commodity}: {len(c)} pairs, kept {len(kept)}, dropped {len(dropped)}; "
                 f"coverage min {c['share_covered'].min():.1%}, median {c['share_covered'].median():.1%}, "
                 f"max {c['share_covered'].max():.1%}")
        if len(dropped):
            L.append("      dropped (below MIN_COVERAGE): " + ", ".join(
                f"{r.district} ({r.share_covered:.1%})" for r in dropped.itertuples()))
        L.append("      kept: " + ", ".join(f"{r.district} ({r.share_covered:.0%})" for r in kept.itertuples()))
    L += ["", "Filled days (price carried forward over gaps of up to MAX_FILL_DAYS)"]
    for commodity, per in filled_counts.items():
        total = sum(per.values())
        L.append(f"  {commodity}: {total:,} filled days in total; most in: " + ", ".join(
            f"{d} ({n})" for d, n in sorted(per.items(), key=lambda kv: -kv[1])[:5]))
    L += ["", "Rows per split (rows with a price on day t)"]
    for level, tables in (("district", district_tables), ("state", state_tables)):
        for commodity, t in tables.items():
            c = t["split"].value_counts()
            L.append(f"  {level} {commodity}: total {len(t):,}, train {c.get('train', 0):,}, "
                     f"valid {c.get('valid', 0):,}, test {c.get('test', 0):,}")
    L += ["", "Rows with a target, per horizon and split"]
    L.append(counts_df.to_string(index=False))
    L += ["", "Missing values per column (columns not listed are complete)"]
    for level, tables in (("district", district_tables), ("state", state_tables)):
        for commodity, t in tables.items():
            L += missing_values_text(f"{level} {commodity}", t)
    L += ["", "Look-ahead check: " + ("passed" if checks_ok else "FAILED, see console")]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    return "\n".join(L)


# ---------------------------------------------------------------- main
def main():
    price_name = sys.argv[1] if len(sys.argv) > 1 else PRICE_FILE
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # read inputs
    prices = read_prices(os.path.join(OUTPUT_DIR, price_name))
    weather = read_weather(WEATHER_LONG_FILE)
    prices, dropped_no_weather = drop_districts_without_weather(prices, weather)

    # daily price per district and commodity
    daily = aggregate_daily(prices)
    last_date = daily["date"].max()
    calendar = pd.date_range(START_DATE, last_date, freq="D")

    # coverage
    log("")
    log("coverage by district and commodity:")
    coverage = coverage_table(daily, calendar)
    show = coverage.copy()
    show["share_covered"] = (show["share_covered"] * 100).round(1).astype(str) + "%"
    log(show.drop(columns="calendar_days").to_string(index=False))
    coverage.to_csv(os.path.join(OUTPUT_DIR, "coverage_by_district.csv"), index=False)
    dropped_pairs = coverage[~coverage["kept"]]
    log("")
    log("dropped for low coverage: " + (", ".join(
        f"{r.district}/{r.commodity} ({r.share_covered:.1%})" for r in dropped_pairs.itertuples())
        or "none"))

    # weather features
    districts_used = sorted(set(coverage["district"]))
    wfeat = weather_feature_table(weather, districts_used)
    ks_wfeat = (wfeat.groupby("date")[WEATHER_FEATURES].mean()
                .rename(columns=lambda c: "ks_" + c).reset_index())
    wd = weather[weather["location"].isin(districts_used)]
    daily_weather_avg = (wd.groupby("date").agg(rain_sum=(RAIN_COL, "mean"), temp_mean=(TEMP_COL, "mean"))
                         .reset_index())
    rain_by_district = {loc: g.set_index("date")[RAIN_COL].sort_index() for loc, g in wd.groupby("location")}
    rain_state = daily_weather_avg.set_index("date")["rain_sum"]

    # fill gaps, build tables per commodity
    kept = coverage[coverage["kept"]]
    district_codes = {d: i for i, d in enumerate(sorted(set(kept["district"])))}
    district_tables, state_tables, state_series, filled_counts, filled_series = {}, {}, {}, {}, {}
    for commodity in COMMODITIES:
        pairs = {}
        for district in kept.loc[kept["commodity"] == commodity, "district"]:
            g = daily[(daily["district"] == district) & (daily["commodity"] == commodity)]
            pairs[district] = fill_pair(g, calendar)
        if not pairs:
            log(f"{commodity}: no district-commodity pair kept, skipping")
            continue
        filled_counts[commodity] = {d: int(f["price_was_filled"].sum()) for d, f in pairs.items()}
        filled_series[commodity] = pairs
        dt = build_district_table(commodity, pairs, wfeat, ks_wfeat, district_codes)
        st, sp = build_state_table(commodity, pairs, calendar, daily_weather_avg, ks_wfeat)
        district_tables[commodity], state_tables[commodity], state_series[commodity] = dt, st, sp
        name = commodity.lower()
        dt.assign(date=dt["date"].dt.strftime("%Y-%m-%d")).to_csv(
            os.path.join(OUTPUT_DIR, f"model_data_district_{name}.csv"), index=False)
        st.assign(date=st["date"].dt.strftime("%Y-%m-%d")).to_csv(
            os.path.join(OUTPUT_DIR, f"model_data_state_{name}.csv"), index=False)
        plot_state_series(commodity, st)
        log(f"{commodity}: {len(pairs)} districts, district table {len(dt):,} rows, "
            f"state table {len(st):,} rows")

    if not district_tables:
        sys.exit("no data left after filtering, nothing to write")

    # baseline scores and weather correlation
    scores = baseline_scores(district_tables, state_tables)
    scores.to_csv(os.path.join(OUTPUT_DIR, "baseline_scores.csv"), index=False)
    corr = weather_correlation(district_tables)
    with open(os.path.join(OUTPUT_DIR, "weather_correlation.csv"), "w", encoding="utf-8") as f:
        f.write("# NOTE: correlation is not proof of cause. Pearson correlation of each weather feature "
                "with target_change_h14, training rows only, district level.\n")
        corr.to_csv(f, index=False)

    # checks
    log("")
    log("look-ahead check (3 random rows per level, recomputed by hand from data cut at day t):")
    checks_ok = True
    for commodity, dt in district_tables.items():
        pairs = filled_series[commodity]
        checks_ok &= verify_no_lookahead(
            f"district level", dt,
            lambda key, com, pairs=pairs: pairs[key]["price"],
            lambda key: rain_by_district[key])
        sp = state_series[commodity]
        st = state_tables[commodity]
        checks_ok &= verify_no_lookahead(
            "state level", st.assign(date=st["date"]),
            lambda key, com, sp=sp: sp,
            lambda key: rain_state)
        break  # one commodity is enough for the hand check
    log("")
    log("rows with a target, per level, commodity and horizon:")
    counts = []
    for commodity in district_tables:
        counts += split_counts("district", commodity, district_tables[commodity])
        counts += split_counts("state", commodity, state_tables[commodity])
    counts_df = pd.DataFrame(counts)
    log(counts_df.to_string(index=False))

    # report
    report = write_report(os.path.join(OUTPUT_DIR, "dataset_report.txt"), coverage,
                          dropped_no_weather, last_date, len(calendar), district_tables,
                          state_tables, filled_counts, counts_df, checks_ok)
    log("")
    log("done. files are in output/.")


if __name__ == "__main__":
    main()
