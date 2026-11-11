"""clean agmarknet mandi price csvs and keep only karnataka onion/tomato rows.

usage:
    python clean_agmarknet.py              # uses INPUT_FILES below
    python clean_agmarknet.py a.csv b.csv  # optional: override INPUT_FILES from the command line

input files are only read, never modified.
"""
import os
import re
import sys
from collections import Counter

import numpy as np
import pandas as pd

# ---------------------------------------------------------------- settings
# replace these placeholder names with your six csv files (2021 to 2026)
INPUT_FILES = ["temp", "temp", "temp", "temp", "temp", "temp"]

# commodity names to keep (exact match, case-insensitive, spaces stripped)
COMMODITIES = ["Onion", "Tomato"]

STATE = "Karnataka"
OUTPUT_DIR = "output"
CHUNKSIZE = 500_000

COLUMNS = ["State", "District", "Market", "Commodity", "Variety", "Grade",
           "Arrival_Date", "Min_Price", "Max_Price", "Modal_Price", "Commodity_Code"]
PRICE_COLS = ["Min_Price", "Max_Price", "Modal_Price"]
SORT_COLS = ["Arrival_Date", "District", "Market", "Commodity", "Variety", "Grade"]

# d/m/yyyy, m/d/yyyy or yyyy-mm-dd (an optional time part is ignored)
DATE_RE = re.compile(r"^(\d{1,4})[/-](\d{1,2})[/-](\d{1,4})(?:[ T].*)?$")


def log(msg):
    print(msg, flush=True)


# ---------------------------------------------------------------- dates
def detect_date_order(values, label):
    """work out day/month order from all distinct date strings of one file."""
    ymd_b_over_12 = ymd_c_over_12 = False  # year-first values: yyyy-b-c
    max_a = max_b = 0                      # slash values: a/b/yyyy
    n_year_first = n_slash = 0
    for v in values:
        m = DATE_RE.match(v)
        if not m:
            continue
        a, b, c = m.groups()
        if len(a) == 4:
            n_year_first += 1
            ymd_b_over_12 |= int(b) > 12
            ymd_c_over_12 |= int(c) > 12
        elif len(c) == 4:
            n_slash += 1
            max_a = max(max_a, int(a))
            max_b = max(max_b, int(b))

    orders = {"year_first": None, "slash": None}
    names = []
    if n_year_first:
        if ymd_b_over_12 and ymd_c_over_12:
            sys.exit(f"STOP: {label}: year-first dates have numbers above 12 in both "
                     "the 2nd and 3rd position, cannot tell the format.")
        orders["year_first"] = "ydm" if ymd_b_over_12 else "ymd"
        names.append("year-first " + ("YYYY-DD-MM" if ymd_b_over_12 else "YYYY-MM-DD"))
    if n_slash:
        if max_a > 12 and max_b > 12:
            sys.exit(f"STOP: {label}: first number goes above 12 and second number "
                     "also goes above 12, so the dates are inconsistent.")
        if max_a > 12:
            orders["slash"] = "dmy"
            names.append("day-first (D/M/YYYY)")
        elif max_b > 12:
            orders["slash"] = "mdy"
            names.append("month-first (M/D/YYYY)")
        else:
            sys.exit(f"STOP: {label}: date format is ambiguous. No value has a first or "
                     "second number above 12, so day-first and month-first both fit. "
                     "Tell me which one it is.")
    return orders, " + ".join(names) if names else "no parseable dates"


def parse_one_date(value, orders):
    m = DATE_RE.match(value)
    if not m:
        return pd.NaT
    a, b, c = m.groups()
    try:
        if len(a) == 4 and orders["year_first"]:
            y, mo, d = int(a), int(b), int(c)
            if orders["year_first"] == "ydm":
                mo, d = d, mo
        elif len(c) == 4 and orders["slash"]:
            y = int(c)
            d, mo = (int(a), int(b)) if orders["slash"] == "dmy" else (int(b), int(a))
        else:
            return pd.NaT
        return pd.Timestamp(y, mo, d)
    except ValueError:  # impossible date such as 31/2/2023
        return pd.NaT


def parse_dates(series, orders):
    """parse a string series using a lookup of its distinct values (fast)."""
    lookup = {v: parse_one_date(v, orders) for v in series.unique()}
    return pd.to_datetime(series.map(lookup))


# ---------------------------------------------------------------- reading one file
def read_one_file(path):
    """stream one file in chunks; return filtered raw rows plus bookkeeping."""
    allowed = {c.strip().casefold() for c in COMMODITIES}
    kept_parts = []
    date_strings = set()
    names_all, names_ka, states = Counter(), Counter(), Counter()
    rows_read = 0

    reader = pd.read_csv(path, dtype=str, keep_default_na=False, chunksize=CHUNKSIZE,
                         encoding="utf-8-sig", encoding_errors="replace")
    for i, chunk in enumerate(reader):
        if i == 0:
            missing = [c for c in COLUMNS if c not in chunk.columns]
            if missing:
                sys.exit(f"STOP: {path}: missing columns {missing}")
        rows_read += len(chunk)

        state = chunk["State"].str.strip().str.casefold()
        commodity = chunk["Commodity"].str.strip()
        commodity_cf = commodity.str.casefold()
        is_state = state == STATE.casefold()

        # tally state spellings and onion/tomato spellings for the check report
        states.update(chunk.loc[state.str.contains("karnat", regex=False), "State"]
                      .str.strip().value_counts().to_dict())
        has_name = commodity_cf.str.contains("onion|tomato", regex=True)
        names_all.update(commodity[has_name].value_counts().to_dict())
        names_ka.update(commodity[has_name & is_state].value_counts().to_dict())

        # keep every date string seen (all rows) for format detection
        date_strings.update(chunk["Arrival_Date"].str.strip().unique())

        keep = is_state & commodity_cf.isin(allowed)
        part = chunk.loc[keep, COLUMNS].apply(lambda s: s.str.strip())
        kept_parts.append(part)
        log(f"  {rows_read:>12,} rows read, {sum(len(p) for p in kept_parts):,} matched")

    kept = pd.concat(kept_parts, ignore_index=True) if kept_parts else pd.DataFrame(columns=COLUMNS)
    return kept, rows_read, date_strings, names_all, names_ka, states


# ---------------------------------------------------------------- cleaning one file
def clean_one_file(kept, orders, source_name):
    """convert dates and prices; split into good rows and rejected rows."""
    kept = kept.copy()
    date = parse_dates(kept["Arrival_Date"], orders)
    prices = pd.DataFrame({
        c: pd.to_numeric(kept[c].str.replace(",", "", regex=False), errors="coerce")
        for c in PRICE_COLS
    })
    prices = prices.where(np.isfinite(prices))

    bad_date = date.isna()
    bad_price = prices.isna().any(axis=1)
    reject_mask = bad_date | bad_price

    rejects = kept[reject_mask].copy()
    rejects["source_file"] = source_name
    reason = pd.Series("", index=rejects.index, dtype=object)
    reason = reason + np.where(bad_date[reject_mask], "bad date; ", "")
    reason = reason + np.where(bad_price[reject_mask], "non-numeric price; ", "")
    rejects["reject_reason"] = reason.str.rstrip("; ")

    good = kept[~reject_mask].copy()
    good["Arrival_Date"] = date[~reject_mask]
    for c in PRICE_COLS:
        good[c] = prices.loc[~reject_mask, c]
    good["source_file"] = source_name
    return good, rejects


# ---------------------------------------------------------------- flags
def flag_rows(df):
    """return a reason string per row (empty if the row is fine)."""
    checks = {
        "price 0 or below": (df[PRICE_COLS] <= 0).any(axis=1),
        "min > modal": df["Min_Price"] > df["Modal_Price"],
        "modal > max": df["Modal_Price"] > df["Max_Price"],
    }
    reason = pd.Series("", index=df.index, dtype=object)
    for name, mask in checks.items():
        reason = reason + np.where(mask, name + "; ", "")
    return reason.str.rstrip("; ")


# ---------------------------------------------------------------- writing
def with_iso_date(df):
    out = df.copy()
    out["Arrival_Date"] = out["Arrival_Date"].dt.strftime("%Y-%m-%d")
    return out


def write_outputs(clean, flagged, rejected):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_cols = COLUMNS

    with_iso_date(clean)[out_cols].to_csv(
        os.path.join(OUTPUT_DIR, "karnataka_onion_tomato.csv"), index=False)
    for year, part in clean.groupby(clean["Arrival_Date"].dt.year):
        with_iso_date(part)[out_cols].to_csv(
            os.path.join(OUTPUT_DIR, f"karnataka_onion_tomato_{year}.csv"), index=False)

    with_iso_date(flagged)[out_cols + ["source_file", "flag_reason"]].to_csv(
        os.path.join(OUTPUT_DIR, "flagged_rows.csv"), index=False)
    rejected[out_cols + ["source_file", "reject_reason"]].to_csv(
        os.path.join(OUTPUT_DIR, "rejected_rows.csv"), index=False)


def build_summary(file_stats, clean, dupes_removed, n_flagged, n_rejected,
                  names_all, names_ka, states):
    lines = ["AGMARKNET KARNATAKA ONION/TOMATO CLEANING SUMMARY", ""]
    lines.append("Per input file")
    for s in file_stats:
        lines.append(f"  {s['file']}: read {s['read']:,}, matched filter {s['matched']:,}, "
                     f"rejected {s['rejected']:,}, kept {s['kept']:,}, date format: {s['format']}")
    lines += ["", f"Total rows kept: {len(clean):,}", "", "Rows per commodity (kept)"]
    for name, n in clean["Commodity"].value_counts().items():
        lines.append(f"  {name}: {n:,}")
    lines += ["", f"Date range: {clean['Arrival_Date'].min():%Y-%m-%d} to "
                  f"{clean['Arrival_Date'].max():%Y-%m-%d}",
              f"Markets: {clean['Market'].nunique():,}",
              f"Districts: {clean['District'].nunique():,}",
              f"Duplicate rows removed: {dupes_removed:,}",
              f"Flagged rows (kept in data, listed in flagged_rows.csv): {n_flagged:,}",
              f"Rejected rows (rejected_rows.csv): {n_rejected:,}", "",
              "State spellings containing 'karnat' (all files)"]
    for name, n in states.most_common():
        lines.append(f"  {name!r}: {n:,}")
    lines += ["", "Commodity values containing 'onion' or 'tomato' (all files)",
              f"  {'value':<35}{'karnataka rows':>16}{'all-state rows':>16}{'kept?':>8}"]
    allowed = {c.strip().casefold() for c in COMMODITIES}
    for name in sorted(names_all):
        kept_flag = "yes" if name.casefold() in allowed else "no"
        lines.append(f"  {name!r:<35}{names_ka.get(name, 0):>16,}{names_all[name]:>16,}{kept_flag:>8}")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- main
def main():
    files = sys.argv[1:] or INPUT_FILES
    log(f"state={STATE}, commodities={COMMODITIES}")

    cleans, rejects_all, file_stats = [], [], []
    names_all, names_ka, states = Counter(), Counter(), Counter()

    for path in files:
        if not os.path.isfile(path):
            log(f"MISSING: input file '{path}' not found, skipping it")
            continue
        log(f"reading {path} ...")
        kept, rows_read, date_strings, n_all, n_ka, st = read_one_file(path)
        names_all.update(n_all)
        names_ka.update(n_ka)
        states.update(st)

        orders, fmt = detect_date_order(date_strings, path)
        log(f"  date format detected: {fmt}")

        good, rejects = clean_one_file(kept, orders, os.path.basename(path))
        log(f"  matched {len(kept):,}, rejected {len(rejects):,}")
        cleans.append(good)
        rejects_all.append(rejects)
        file_stats.append({"file": path, "read": rows_read, "matched": len(kept),
                           "rejected": len(rejects), "format": fmt})

    if not cleans:
        sys.exit("no input files were found, nothing to do. Edit INPUT_FILES at the top of the script.")

    # combine, remove exact duplicates
    log("combining files and removing duplicates ...")
    clean = pd.concat(cleans, ignore_index=True)
    dup = clean.duplicated(subset=COLUMNS, keep="first")
    dupes_removed = int(dup.sum())
    clean = clean[~dup].copy()
    kept_by_file = clean["source_file"].value_counts()
    for s in file_stats:
        s["kept"] = int(kept_by_file.get(os.path.basename(s["file"]), 0))
    log(f"  removed {dupes_removed:,} duplicate rows")

    # flag suspicious prices (rows stay in the data)
    log("flagging rows ...")
    clean["flag_reason"] = flag_rows(clean)
    flagged = clean[clean["flag_reason"] != ""].copy()
    log(f"  flagged {len(flagged):,} rows")

    # sort
    log("sorting ...")
    clean = clean.sort_values(SORT_COLS, kind="mergesort").reset_index(drop=True)
    flagged = flagged.sort_values(SORT_COLS, kind="mergesort").reset_index(drop=True)
    rejected = pd.concat(rejects_all, ignore_index=True)

    # write outputs
    log(f"writing outputs to '{OUTPUT_DIR}/' ...")
    write_outputs(clean, flagged, rejected)
    summary = build_summary(file_stats, clean, dupes_removed, len(flagged), len(rejected),
                            names_all, names_ka, states)
    with open(os.path.join(OUTPUT_DIR, "summary.txt"), "w", encoding="utf-8") as f:
        f.write(summary)

    log("")
    log(summary)
    log("done.")


if __name__ == "__main__":
    main()
