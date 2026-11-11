"""download daily historical weather (open-meteo archive api) for every district in the price data.

install:
    pip install requests pandas

run:
    python fetch_weather.py                       # uses PRICE_FILE below
    python fetch_weather.py karnataka_onion_tomato.csv   # or pass the price file name (inside output/)

steps: read districts from the cleaned price csv -> find coordinates with the open-meteo
geocoding api (or LOCATION_OVERRIDES) -> download daily weather -> checks -> csv outputs.
no api key is needed. raw responses are cached in output/raw/<location>.json.
"""
import json
import os
import re
import sys
import time
import unicodedata
from datetime import date, timedelta

import pandas as pd
import requests

# ---------------------------------------------------------------- settings
# cleaned price csv inside OUTPUT_DIR (placeholder, change it or pass the name on the command line)
PRICE_FILE = "temp"

# district name -> (latitude, longitude); wins over the geocoding lookup. keys are matched
# after strip + lowercase, so "Bangalore" and "bangalore" are the same. example:
#   LOCATION_OVERRIDES = {"bangalore": (12.97, 77.59)}
LOCATION_OVERRIDES = {}

STATE_NAME = "Karnataka"     # geocoding result must be in this state (admin1)

START_DATE = "2021-01-01"
# the archive lags a few days, so default to today minus 7 days
END_DATE = (date.today() - timedelta(days=7)).isoformat()

DAILY_VARIABLES = [
    "temperature_2m_max",
    "temperature_2m_min",
    "temperature_2m_mean",
    "precipitation_sum",
    "rain_sum",
    "relative_humidity_2m_mean",
]
TIMEZONE = "Asia/Kolkata"

FORCE_REFRESH = False        # True = ignore cached files and download/look up again
OUTPUT_DIR = "output"

API_URL = "https://archive-api.open-meteo.com/v1/archive"
GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
PAUSE_SECONDS = 1.0          # pause between requests
MAX_TRIES = 5
TIMEOUT_SECONDS = 90

# old or alternate spellings to try when the district name finds nothing (lowercase keys)
ALTERNATE_SPELLINGS = {
    "bangalore": ["bengaluru", "bangalore urban"],
    "belgaum": ["belagavi"],
    "bellary": ["ballari"],
    "bijapur": ["vijayapura"],
    "kalburgi": ["kalaburagi", "gulbarga"],
    "gulbarga": ["kalaburagi"],
    "shimoga": ["shivamogga"],
    "mysore": ["mysuru"],
    "tumkur": ["tumakuru"],
    "chikmagalur": ["chikkamagaluru", "chikmagaluru"],
    "davangere": ["davanagere"],
    "chamrajnagar": ["chamarajanagar"],
    "mangalore": ["mangaluru"],
    "dakshin kannad": ["dakshina kannada", "mangaluru"],
    "madikeri": ["kodagu", "mercara"],
    "kodagu": ["madikeri"],
    "dharwad": ["dharwar"],
    "bagalkot": ["bagalakote", "bagalkote"],
    "koppal": ["koppala"],
    "raichur": ["raichuru"],
    "hubli": ["hubballi"],
}


def log(msg):
    print(msg, flush=True)


class BadRequest(Exception):
    """http 400 from the api (not retried)."""


# ---------------------------------------------------------------- names
def normalise(name):
    return str(name).strip().lower()


def slug(name):
    """safe text for file names and column names."""
    return re.sub(r"[^a-z0-9]+", "_", name).strip("_")


# ---------------------------------------------------------------- generic api call
def api_get(url, params):
    """one get request with retries on 429, 5xx and connection errors."""
    last_error = ""
    for attempt in range(1, MAX_TRIES + 1):
        wait = 2 ** attempt  # 2, 4, 8, 16, 32 seconds
        try:
            resp = requests.get(url, params=params, timeout=TIMEOUT_SECONDS)
        except requests.RequestException as exc:
            last_error = f"connection error: {exc}"
        else:
            if resp.status_code == 200:
                return resp.json()
            try:
                reason = resp.json().get("reason", "")
            except ValueError:
                reason = resp.text[:200]
            if resp.status_code == 400:
                raise BadRequest(reason)
            last_error = f"http {resp.status_code}: {reason}"
            if resp.status_code < 500 and resp.status_code != 429:
                raise RuntimeError(last_error)
            retry_after = resp.headers.get("Retry-After", "")
            if retry_after.isdigit():
                wait = max(wait, int(retry_after))
        if attempt < MAX_TRIES:
            log(f"    {last_error}; retry {attempt}/{MAX_TRIES - 1} in {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"gave up after {MAX_TRIES} tries ({last_error})")


# ---------------------------------------------------------------- districts from price data
def read_districts(price_name):
    """distinct districts from the price file: {normalised name: most common original spelling}."""
    path = os.path.join(OUTPUT_DIR, price_name)
    if not os.path.isfile(path):
        sys.exit(f"price file '{path}' not found. Set PRICE_FILE at the top of the script "
                 "or pass the file name (inside output/) as an argument.")
    prices = pd.read_csv(path, usecols=["District"], dtype=str, keep_default_na=False)
    raw = prices["District"].dropna()
    raw = raw[raw.str.strip() != ""]
    districts = {}
    for norm, grp in raw.groupby(raw.map(normalise)):
        districts[norm] = grp.value_counts().index[0]
    log(f"{len(districts)} distinct districts in {path} ({len(prices):,} rows)")
    return districts


# ---------------------------------------------------------------- geocoding
def candidate_names(name):
    """names to search for: the name, text before/inside brackets, then alternate spellings."""
    base = name.split("(")[0].strip()
    inside = [t.strip() for t in re.findall(r"\(([^)]*)\)", name)]
    names = [base] + inside
    for n in list(names):
        names += ALTERNATE_SPELLINGS.get(n, [])
    seen, ordered = set(), []
    for n in names:
        if n and n not in seen:
            seen.add(n)
            ordered.append(n)
    return ordered


def fold(text):
    """lowercase and remove accents so 'Kolār' equals 'kolar'."""
    text = unicodedata.normalize("NFKD", str(text))
    return "".join(c for c in text if not unicodedata.combining(c)).strip().lower()


def geocode_name(query):
    """search one name; return populated places in STATE_NAME (exact name matches first)."""
    data = api_get(GEOCODING_URL, {"name": query, "count": 10, "language": "en",
                                   "format": "json", "countryCode": "IN"})
    # only towns and cities (feature code PPL*), so airports, roads and wards are skipped
    results = [r for r in data.get("results", [])
               if r.get("admin1") == STATE_NAME and str(r.get("feature_code", "")).startswith("PPL")]
    # biggest population first (the real town, not a village of the same name), then exact name
    results.sort(key=lambda r: (-(r.get("population") or 0), fold(r["name"]) != fold(query)))
    return results


def geocode_district(name):
    """try the district name, then alternate spellings; prefer an exact name match."""
    fallback = None
    for i, query in enumerate(candidate_names(name)):
        time.sleep(PAUSE_SECONDS)
        results = geocode_name(query)
        if not results:
            continue
        hit = results[0]
        match = {"query": query, "alt_spelling": i > 0, "place": hit["name"],
                 "admin2": hit.get("admin2", ""), "state": hit["admin1"],
                 "feature_code": hit.get("feature_code", ""),
                 "latitude": hit["latitude"], "longitude": hit["longitude"],
                 "exact": fold(hit["name"]) == fold(query),
                 "population": hit.get("population") or 0}
        if match["exact"] or match["population"] > 0:
            return match  # exact name, or a town with a known population
        fallback = fallback or match  # remember the first inexact hit, keep trying spellings
    return fallback


def resolve_locations(districts):
    """coordinates for every district; returns (rows for the table, coords, not_found)."""
    cache_file = os.path.join(OUTPUT_DIR, "raw", "geocoding.json")
    cache = {}
    if os.path.isfile(cache_file) and not FORCE_REFRESH:
        try:
            with open(cache_file, encoding="utf-8") as f:
                cache = json.load(f)
        except (OSError, ValueError):
            cache = {}

    overrides = {normalise(k): v for k, v in LOCATION_OVERRIDES.items()}
    rows, coords, not_found = [], {}, []
    for norm, original in districts.items():
        if norm in overrides:
            lat, lon = overrides[norm]
            match = {"query": "", "alt_spelling": False, "place": "(override)", "admin2": "",
                     "state": "(override)", "feature_code": "", "latitude": lat, "longitude": lon}
        elif norm in cache and cache[norm] is not None:
            match = cache[norm]
        else:
            log(f"  looking up '{norm}' ...")
            try:
                match = geocode_district(norm)
            except Exception as exc:
                log(f"    lookup failed for '{norm}': {exc}")
                match = None
            cache[norm] = match
        if match is None:
            not_found.append(original)
            continue
        coords[norm] = (match["latitude"], match["longitude"])
        notes = []
        if match["place"] == "(override)":
            notes.append("from LOCATION_OVERRIDES")
        else:
            if match["alt_spelling"]:
                notes.append(f"found as '{match['query']}'")
            if not match.get("exact", True):
                if match.get("population", 0) > 0:
                    notes.append(f"matched name differs (population {match['population']:,})")
                else:
                    notes.append("CHECK: weak match, no exact name and no population")
        rows.append({"district": norm, "original": original, "place": match["place"],
                     "admin2": match["admin2"], "state": match["state"],
                     "latitude": match["latitude"], "longitude": match["longitude"],
                     "note": "; ".join(notes)})

    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=1)
    return rows, coords, not_found


def format_match_table(rows, not_found):
    """text table of districts and the place each one matched."""
    cols = ["district", "place", "admin2", "state", "latitude", "longitude", "note"]
    table = pd.DataFrame(rows, columns=cols)
    lines = [table.to_string(index=False, justify="left")]
    if not_found:
        lines.append("")
        lines.append("Districts NOT found (add them to LOCATION_OVERRIDES): "
                     + ", ".join(not_found))
    return "\n".join(lines)


# ---------------------------------------------------------------- weather api calls
def make_params(lat, lon, start, end, variables):
    return {"latitude": lat, "longitude": lon, "start_date": start, "end_date": end,
            "daily": ",".join(variables), "timezone": TIMEZONE}


def find_supported_variables(lat, lon):
    """probe the api with a tiny request; drop variables it rejects."""
    probe_end = (pd.Timestamp(START_DATE) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    try:
        api_get(API_URL, make_params(lat, lon, START_DATE, probe_end, DAILY_VARIABLES))
        return list(DAILY_VARIABLES), []
    except BadRequest:
        pass  # some variable is not supported, test one by one
    except RuntimeError as exc:
        log(f"  probe failed ({exc}); assuming all variables are supported")
        return list(DAILY_VARIABLES), []

    supported, dropped = [], []
    for var in DAILY_VARIABLES:
        time.sleep(PAUSE_SECONDS)
        try:
            api_get(API_URL, make_params(lat, lon, START_DATE, probe_end, [var]))
            supported.append(var)
        except BadRequest as exc:
            log(f"  variable '{var}' is not supported by the api ({exc}), dropping it")
            dropped.append(var)
        except RuntimeError:
            supported.append(var)  # inconclusive, keep it
    return supported, dropped


def fetch_range(lat, lon, start, end, variables):
    """whole range in one request; if the api rejects it, one request per year."""
    try:
        return api_get(API_URL, make_params(lat, lon, start, end, variables))
    except BadRequest as exc:
        log(f"    api rejected the full range ({exc}), retrying one request per year")

    merged = None
    for year in range(int(start[:4]), int(end[:4]) + 1):
        y_start = max(start, f"{year}-01-01")
        y_end = min(end, f"{year}-12-31")
        time.sleep(PAUSE_SECONDS)
        part = api_get(API_URL, make_params(lat, lon, y_start, y_end, variables))
        if merged is None:
            merged = part
        else:
            for key, values in part["daily"].items():
                merged["daily"][key] = merged["daily"][key] + values
    return merged


# ---------------------------------------------------------------- cache
def cache_path(name):
    return os.path.join(OUTPUT_DIR, "raw", f"{slug(name)}.json")


def load_cache(name, lat, lon):
    """return cached response if it matches the current settings, else None."""
    path = cache_path(name)
    if FORCE_REFRESH or not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            cached = json.load(f)
        req = cached["_request"]
    except (OSError, ValueError, KeyError):
        return None
    same = (req["latitude"] == lat and req["longitude"] == lon
            and req["start_date"] == START_DATE and req["timezone"] == TIMEZONE
            and req["requested_variables"] == DAILY_VARIABLES
            and req["end_date"] >= END_DATE)
    return cached if same else None


def save_cache(name, response, lat, lon, requested_variables):
    os.makedirs(os.path.dirname(cache_path(name)), exist_ok=True)
    response["_request"] = {"latitude": lat, "longitude": lon, "start_date": START_DATE,
                            "end_date": END_DATE, "timezone": TIMEZONE,
                            "requested_variables": requested_variables}
    with open(cache_path(name), "w", encoding="utf-8") as f:
        json.dump(response, f)


# ---------------------------------------------------------------- tables
def response_to_frame(response):
    """daily json -> dataframe, trailing all-null days trimmed."""
    df = pd.DataFrame(response["daily"]).rename(columns={"time": "date"})
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"] <= pd.Timestamp(END_DATE)]
    value_cols = [c for c in df.columns if c != "date"]
    has_data = df[value_cols].notna().any(axis=1)
    if has_data.any():
        df = df.loc[:has_data[has_data].index[-1]]  # cut everything after the last valid day
    else:
        df = df.iloc[0:0]
    return df.reset_index(drop=True)


def check_location(df):
    """gaps, duplicates and value-rule checks; returns a list of problem strings."""
    problems = []
    if df.empty:
        return ["no valid data returned"]
    expected = pd.date_range(START_DATE, df["date"].max(), freq="D")
    dup = int(df["date"].duplicated().sum())
    missing = expected.difference(pd.DatetimeIndex(df["date"]))
    if dup:
        problems.append(f"{dup} duplicate dates")
    if len(missing):
        problems.append(f"{len(missing)} missing dates, first {missing[0]:%Y-%m-%d}, "
                        f"last {missing[-1]:%Y-%m-%d}")
    for col in ("precipitation_sum", "rain_sum"):
        if col in df and (df[col] < 0).sum():
            problems.append(f"{int((df[col] < 0).sum())} rows with negative {col}")
    temps = ["temperature_2m_min", "temperature_2m_mean", "temperature_2m_max"]
    if set(temps) <= set(df.columns):
        bad = ~((df[temps[0]] <= df[temps[1]]) & (df[temps[1]] <= df[temps[2]]))
        bad &= df[temps].notna().all(axis=1)
        if bad.sum():
            problems.append(f"{int(bad.sum())} rows break tmin <= tmean <= tmax")
    return problems


def build_tables(frames, coords, originals):
    """long table (one row per date and location) and wide table (one row per date)."""
    long_parts, wide_parts = [], []
    for name, df in frames.items():
        lat, lon = coords[name]
        part = df.copy()
        part.insert(1, "location", name)
        part.insert(2, "district_original", originals[name])
        part.insert(3, "latitude", lat)
        part.insert(4, "longitude", lon)
        long_parts.append(part)
        wide = df.set_index("date")
        wide.columns = [f"{slug(name)}_{c}" for c in wide.columns]
        wide_parts.append(wide)
    long_df = pd.concat(long_parts, ignore_index=True).sort_values(["date", "location"])
    wide_df = pd.concat(wide_parts, axis=1).sort_index().reset_index()
    for table in (long_df, wide_df):
        table["date"] = table["date"].dt.strftime("%Y-%m-%d")
    return long_df, wide_df


# ---------------------------------------------------------------- summary
def build_summary(frames, grid, coords, dropped, failed, problems, match_table, not_found):
    lines = ["OPEN-METEO DAILY WEATHER SUMMARY", ""]
    lines.append(f"Locations (districts) with weather: {len(frames)}")
    lines.append(f"Districts without coordinates: {', '.join(not_found) or 'none'}")
    for name, why in failed.items():
        lines.append(f"  FAILED {name}: {why}")
    lines.append(f"Requested range: {START_DATE} to {END_DATE}")
    ends = {n: df["date"].max() for n, df in frames.items() if not df.empty}
    if ends:
        lines.append(f"Last valid date: {max(ends.values()):%Y-%m-%d}"
                     + ("" if len(set(ends.values())) == 1 else
                        " (differs by location: " + ", ".join(
                            f"{n} {d:%Y-%m-%d}" for n, d in ends.items()) + ")"))
    lines.append(f"Variables dropped (not supported): {', '.join(dropped) or 'none'}")
    lines += ["", "District to place matching", match_table]
    lines += ["", "Rows per location and grid cell used by the api"]
    for name, df in frames.items():
        g = grid.get(name, {})
        lines.append(f"  {name}: {len(df):,} rows, requested {coords[name]}, "
                     f"grid cell ({g.get('latitude')}, {g.get('longitude')}), "
                     f"elevation {g.get('elevation')} m")
    lines += ["", "Missing values per column per location (only locations with any)"]
    any_missing = False
    for name, df in frames.items():
        miss = df.drop(columns="date").isna().sum()
        miss = miss[miss > 0]
        if len(miss):
            any_missing = True
            lines.append(f"  {name}: " + ", ".join(f"{c}={int(n)}" for c, n in miss.items()))
    if not any_missing:
        lines.append("  none, every column is complete for every location")
    lines += ["", "Checks (rows are never deleted)"]
    for name in frames:
        lines.append(f"  {name}: " + ("; ".join(problems[name]) if problems[name] else "ok"))
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- main
def main():
    price_name = sys.argv[1] if len(sys.argv) > 1 else PRICE_FILE
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    log(f"range {START_DATE} to {END_DATE}, timezone {TIMEZONE}")

    # districts and coordinates
    districts = read_districts(price_name)
    log("finding coordinates ...")
    rows, coords, not_found = resolve_locations(districts)
    match_table = format_match_table(rows, not_found)
    log("")
    log(match_table)
    log("")
    if not coords:
        sys.exit("no district has coordinates, nothing to download")

    # work out which locations still need a download
    cached = {n: load_cache(n, *xy) for n, xy in coords.items()}
    variables, dropped = list(DAILY_VARIABLES), []
    need_download = [n for n, c in cached.items() if c is None]
    if need_download:
        log("checking which daily variables the api supports ...")
        variables, dropped = find_supported_variables(*coords[need_download[0]])
        if not variables:
            sys.exit("none of the variables are supported, nothing to download")
        log(f"  using: {', '.join(variables)}")

    frames, grid, failed = {}, {}, {}
    first_request = True
    for name, (lat, lon) in coords.items():
        try:
            response = cached[name]
            if response is not None:
                log(f"{name}: using cached response")
                kept_vars = [v for v in DAILY_VARIABLES if v in response["daily"]]
                dropped = [v for v in DAILY_VARIABLES if v not in kept_vars]
            else:
                if not first_request:
                    time.sleep(PAUSE_SECONDS)
                first_request = False
                log(f"{name}: downloading {START_DATE} to {END_DATE} ...")
                response = fetch_range(lat, lon, START_DATE, END_DATE, variables)
                save_cache(name, response, lat, lon, DAILY_VARIABLES)
            df = response_to_frame(response)
            frames[name] = df
            grid[name] = {k: response.get(k) for k in ("latitude", "longitude", "elevation")}
            if df.empty:
                log("  no valid rows")
            else:
                log(f"  {len(df):,} rows, last valid date {df['date'].max():%Y-%m-%d}")
        except Exception as exc:  # one bad location must not stop the run
            failed[name] = str(exc)
            log(f"  FAILED {name}: {exc}")

    if not frames:
        sys.exit("no location could be downloaded, nothing to write")

    # checks
    log("running checks ...")
    problems = {name: check_location(df) for name, df in frames.items()}
    for name, plist in problems.items():
        for p in plist:
            log(f"  PROBLEM {name}: {p}")
    if not any(problems.values()):
        log("  all checks passed")

    # write outputs
    log(f"writing outputs to '{OUTPUT_DIR}/' ...")
    long_df, wide_df = build_tables(frames, coords, districts)
    long_df.to_csv(os.path.join(OUTPUT_DIR, "weather_daily_long.csv"), index=False)
    wide_df.to_csv(os.path.join(OUTPUT_DIR, "weather_daily_wide.csv"), index=False)
    summary = build_summary(frames, grid, coords, dropped, failed, problems,
                            match_table, not_found)
    with open(os.path.join(OUTPUT_DIR, "weather_summary.txt"), "w", encoding="utf-8") as f:
        f.write(summary)
    log("")
    log(summary)
    log("done.")


if __name__ == "__main__":
    main()
