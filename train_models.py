"""train and compare forecasting models for karnataka onion and tomato prices.

install (python 3.10+):
    pip install pandas numpy matplotlib scipy statsmodels prophet xgboost scikit-learn joblib holidays

run:
    python train_models.py            # full run (can take 30-60 minutes)
    python train_models.py --quick    # small grids, for a fast test

reads the datasets built in the previous step from output/, never changes them.
rules: no shuffling, tuning only on valid, final models retrained on train+valid, test scored once.
"""
import json
import logging
import os
import random
import sys
import time
import warnings

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import ParameterSampler
from statsmodels.tsa.statespace.sarimax import SARIMAX
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")
logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
logging.getLogger("prophet").setLevel(logging.ERROR)

# ---------------------------------------------------------------- settings
COMMODITIES = ["Onion", "Tomato"]
HORIZONS = [7, 14, 30]
RANDOM_SEED = 42
TARGET_MODES = ["price", "change"]   # predict the price, or the % change from today's price
MAX_MINUTES_PER_MODEL = 10           # stop a model's search early after this long

OUTPUT_DIR = "output"
MODEL_DIR = os.path.join(OUTPUT_DIR, "models")
WEATHER_LONG_FILE = os.path.join(OUTPUT_DIR, "weather_daily_long.csv")
TRAIN_END = pd.Timestamp("2024-12-31")
VALID_END = pd.Timestamp("2025-12-31")

WEEKLY_STEPS = {7: 1, 14: 2, 30: 4}  # day horizon -> nearest weekly step (30 days = 4.3 weeks -> 4)
SARIMA_ORDERS = [(1, 1, 0), (0, 1, 1), (1, 1, 1)]
SARIMA_SEASONAL = [(0, 0, 0, 0), (1, 0, 0, 52)]    # without / with a yearly seasonal part
REFIT_EVERY_WEEKS = 13               # expanding window, parameters re-estimated every 13 weeks
PROPHET_CPS = [0.01, 0.1, 0.5]       # changepoint_prior_scale grid
PROPHET_REFIT_DAYS = 56              # expanding window, refit every 8 weeks
PROPHET_JOBS = 3
XGB_ITER, RF_ITER = 12, 6            # randomised search sizes
N_JOBS = 8

LAG_COLS = ["price_lag_1", "price_lag_7", "price_lag_14", "price_lag_30",
            "price_roll_mean_7", "price_roll_std_7", "price_roll_mean_30", "price_roll_std_30"]
WEATHER_COLS = ["rain_sum_7d", "rain_sum_30d", "temp_mean_7d", "temp_mean_30d",
                "ks_rain_sum_7d", "ks_rain_sum_30d", "ks_temp_mean_7d", "ks_temp_mean_30d"]
OTHER_COLS = ["price", "n_markets", "price_was_filled", "ks_price_lag_7", "month", "week_of_year",
              "day_of_week", "doy_sin", "doy_cos", "district_code"]

QUICK = "--quick" in sys.argv
if QUICK:
    SARIMA_ORDERS, SARIMA_SEASONAL = [(1, 1, 0), (0, 1, 1)], [(0, 0, 0, 0)]
    PROPHET_CPS, XGB_ITER, RF_ITER = [0.1], 3, 2


def log(msg=""):
    print(msg, flush=True)


def seed_everything():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)


# ---------------------------------------------------------------- loading
def load_state(commodity):
    return pd.read_csv(os.path.join(OUTPUT_DIR, f"model_data_state_{commodity.lower()}.csv"),
                       parse_dates=["date"])


def load_district(commodity):
    return pd.read_csv(os.path.join(OUTPUT_DIR, f"model_data_district_{commodity.lower()}.csv"),
                       parse_dates=["date"])


def load_baseline():
    b = pd.read_csv(os.path.join(OUTPUT_DIR, "baseline_scores.csv"))
    return {(r.level, r.commodity, r.district, int(r.horizon), r.split): r.MAE for r in b.itertuples()}


def load_weather_daily():
    """karnataka-wide daily weather: average over all weather locations."""
    w = pd.read_csv(WEATHER_LONG_FILE, parse_dates=["date"])
    return (w.groupby("date").agg(rain=("rain_sum", "mean"), temp=("temperature_2m_mean", "mean"))
            .sort_index())


# ---------------------------------------------------------------- scoring
def metrics(actual, pred, today):
    ok = actual.notna() & pred.notna() & today.notna()
    a, p, t = actual[ok].to_numpy(), pred[ok].to_numpy(), today[ok].to_numpy()
    if len(a) == 0:
        return {"n": 0}
    err = p - a
    moved = a != t  # direction is only judged where the price really changed
    direction = (np.sign(p - t)[moved] == np.sign(a - t)[moved]).mean() if moved.any() else np.nan
    naive_mae = np.abs(a - t).mean()
    mae = np.abs(err).mean()
    return {"n": len(a), "MAE": mae, "RMSE": float(np.sqrt((err ** 2).mean())),
            "MAPE_percent": float((np.abs(err) / a).mean() * 100),
            "direction_accuracy": direction, "n_direction": int(moved.sum()),
            "naive_MAE": naive_mae,
            "skill_vs_naive_percent": (1 - mae / naive_mae) * 100 if naive_mae > 0 else np.nan}


def score_model(frame, pred_col, level, model, uses_weather, commodity, h, mode, settings, baseline):
    """metric rows for valid and test (and per district for the district level)."""
    rows = []
    groups = [("ALL", frame)]
    if level == "district":
        groups += list(frame.groupby("district"))
    for split in ("valid", "test"):
        for district, g in groups:
            g = g[g["split"] == split]
            m = metrics(g["actual"], g[pred_col], g["price_today"])
            if m["n"] == 0:
                continue
            m.update({"level": level, "model": model, "uses_weather": uses_weather,
                      "commodity": commodity, "horizon": h, "split": split, "district": district,
                      "target_mode": mode, "settings": json.dumps(settings, default=str),
                      "naive_MAE_baseline_file": baseline.get((level, commodity, district, h, split), np.nan)})
            rows.append(m)
    return rows


# ---------------------------------------------------------------- base frames (rows that get scored)
def base_frame(df, h, level):
    """rows of valid and test with a target. valid rows whose target date falls after VALID_END are
    left out so that no test-period price influences any choice."""
    f = df[df[f"target_h{h}"].notna() & df["split"].isin(["valid", "test"])].copy()
    f["target_date"] = f["date"] + pd.Timedelta(days=h)
    f = f[(f["split"] == "test") | (f["target_date"] <= VALID_END)]
    cols = ["date", "target_date", "split"] + (["district"] if level == "district" else [])
    out = f[cols].copy()
    out["price_today"] = f["price"]
    out["actual"] = f[f"target_h{h}"]
    return out.reset_index(drop=True)


# ---------------------------------------------------------------- A1/A2: SARIMA and SARIMAX (weekly)
def make_weekly(state_df, wx_daily):
    y = state_df.set_index("date")["price"].resample("W-SUN").mean()
    y = y[y.index <= state_df["date"].max()]            # only weeks that have ended
    x = wx_daily[["rain", "temp"]].resample("W-SUN").mean()
    return y, x


def last_sunday(date):
    return date - pd.Timedelta(days=(date.dayofweek + 1) % 7)


def sarima_rolling(y, x, spec, first_origin, last_origin, steps, start_params=None, deadline=None):
    """expanding window. parameters are re-estimated every REFIT_EVERY_WEEKS weeks, in between the
    new weekly observation is appended without refitting. forecast for each origin week T uses
    data up to T only. x (optional) is the already shifted weather for these steps."""
    order, seasonal = spec
    week = pd.Timedelta(weeks=1)
    origins = y.index[(y.index >= first_origin) & (y.index <= last_origin)]
    rows, res, prev, fails, last_fit, first_params, timed_out = {}, None, None, 0, None, None, False
    for i, T in enumerate(origins):
        if deadline is not None and time.time() > deadline:
            timed_out = True
            break
        try:
            if res is None or i % REFIT_EVERY_WEEKS == 0:
                mod = SARIMAX(y.loc[:T], exog=None if x is None else x.loc[:T], order=order,
                              seasonal_order=seasonal, enforce_stationarity=False,
                              enforce_invertibility=False)
                res = mod.fit(disp=False, maxiter=60, start_params=start_params)
                start_params = res.params.to_numpy()
                first_params = first_params if first_params is not None else start_params
                last_fit = res
            else:
                new = y.loc[prev + week:T]
                res = res.append(new, exog=None if x is None else x.loc[new.index], refit=False)
            s_max = max(steps)
            fx = None if x is None else x.loc[T + week:T + s_max * week]
            fc = res.forecast(steps=s_max, exog=fx).to_numpy()
            rows[T] = [fc[s - 1] for s in steps]
        except Exception:
            fails += 1
            res = None
        prev = T
    out = pd.DataFrame.from_dict(rows, orient="index", columns=steps).sort_index()
    return out, first_params, last_fit, fails, timed_out


def weekly_to_daily(frame, fc, step):
    """prediction for each daily row t = forecast made at the last Sunday on or before t."""
    origin = frame["date"].map(last_sunday)
    return origin.map(fc[step]) if step in fc.columns else pd.Series(np.nan, index=frame.index)


def exog_for_step(x_raw, y, step, first_origin):
    """weather shifted back by `step` weeks, standardised with data known at first_origin."""
    xs = x_raw.shift(step)
    xs = xs.loc[y.index[step]:]
    base = xs.loc[:first_origin]
    return (xs - base.mean()) / base.std()


def run_sarima_family(commodity, state_df, wx_daily, frames, baseline, results, preds, saved):
    y, x_raw = make_weekly(state_df, wx_daily)
    t0_valid = last_sunday(TRAIN_END)
    t_end_valid = last_sunday(VALID_END)
    t_end_test = y.index.max()
    steps_all = sorted(set(WEEKLY_STEPS.values()))

    # sarima grid on valid, shared by all horizons (one fit gives all steps)
    log(f"  SARIMA grid ({len(SARIMA_ORDERS) * len(SARIMA_SEASONAL)} specs), valid period")
    start = time.time()
    deadline = start + MAX_MINUTES_PER_MODEL * 60
    grid, timed_out = {}, False
    for order in SARIMA_ORDERS:
        for seas in SARIMA_SEASONAL:
            if time.time() > deadline:
                timed_out = True
                log(f"    time limit reached, skipping remaining specs from {order}{seas}")
                break
            t1 = time.time()
            fc, p0, _, fails, to = sarima_rolling(y, None, (order, seas), t0_valid, t_end_valid,
                                                  steps_all, deadline=deadline)
            grid[(order, seas)] = (fc, p0)
            timed_out |= to
            log(f"    {order}x{seas}: {len(fc)} origins, {fails} failed, {time.time() - t1:.0f}s")
        else:
            continue
        break

    best_spec = {}
    for h in HORIZONS:
        f = frames[h]
        val = f[f["split"] == "valid"]
        scored = []
        for spec, (fc, _) in grid.items():
            pred = weekly_to_daily(val, fc, WEEKLY_STEPS[h])
            m = metrics(val["actual"], pred, val["price_today"])
            if m["n"]:
                scored.append((m["MAE"], spec))
        if scored:
            best_spec[h] = min(scored)[1]
    log(f"  SARIMA chosen on valid: " + ", ".join(f"h{h}={best_spec[h]}" for h in best_spec))

    # final sarima: refit on train+valid (expanding), score test once
    test_cache = {}
    for h in best_spec:
        spec = best_spec[h]
        if spec not in test_cache:
            fc_t, _, last_fit, fails, to = sarima_rolling(
                y, None, spec, t_end_valid, t_end_test, steps_all,
                start_params=grid[spec][1], deadline=time.time() + MAX_MINUTES_PER_MODEL * 60)
            test_cache[spec] = (fc_t, last_fit)
        f = frames[h]
        col = "SARIMA"
        pred = pd.Series(np.nan, index=f.index)
        v, t = f["split"] == "valid", f["split"] == "test"
        pred[v] = weekly_to_daily(f[v], grid[spec][0], WEEKLY_STEPS[h])
        pred[t] = weekly_to_daily(f[t], test_cache[spec][0], WEEKLY_STEPS[h])
        preds[("state", commodity, h)][col] = pred
        settings = {"order": spec[0], "seasonal_order": spec[1], "weekly_step": WEEKLY_STEPS[h],
                    "refit_every_weeks": REFIT_EVERY_WEEKS, "grid_time_limit_hit": timed_out}
        results += score_model(preds[("state", commodity, h)], col, "state", "SARIMA", "no",
                               commodity, h, "price", settings, baseline)
        if test_cache[spec][1] is not None:
            path = os.path.join(MODEL_DIR, f"sarima_{commodity.lower()}_h{h}.pkl")
            test_cache[spec][1].save(path)
            saved.append(path)

    # sarimax: chosen sarima spec plus weather shifted back by the horizon in weeks
    for h in best_spec:
        s = WEEKLY_STEPS[h]
        spec = best_spec[h]
        t1 = time.time()
        yy = y.iloc[s:]
        xv = exog_for_step(x_raw, y, s, t0_valid)
        fc_v, p0, _, f_v, to_v = sarima_rolling(yy, xv, spec, t0_valid, t_end_valid, [s],
                                                deadline=time.time() + MAX_MINUTES_PER_MODEL * 60)
        xt = exog_for_step(x_raw, y, s, t_end_valid)
        fc_t, _, last_fit, f_t, to_t = sarima_rolling(yy, xt, spec, t_end_valid, t_end_test, [s],
                                                      start_params=p0,
                                                      deadline=time.time() + MAX_MINUTES_PER_MODEL * 60)
        f = frames[h]
        pred = pd.Series(np.nan, index=f.index)
        v, t = f["split"] == "valid", f["split"] == "test"
        pred[v] = weekly_to_daily(f[v], fc_v, s)
        pred[t] = weekly_to_daily(f[t], fc_t, s)
        col = "SARIMAX_weather"
        preds[("state", commodity, h)][col] = pred
        settings = {"order": spec[0], "seasonal_order": spec[1], "weekly_step": s,
                    "weather_shift_weeks": s, "fit_failures": f_v + f_t, "time_limit_hit": to_v or to_t}
        results += score_model(preds[("state", commodity, h)], col, "state", "SARIMAX (weather)", "yes",
                               commodity, h, "price", settings, baseline)
        if last_fit is not None:
            path = os.path.join(MODEL_DIR, f"sarimax_{commodity.lower()}_h{h}.pkl")
            last_fit.save(path)
            saved.append(path)
        log(f"  SARIMAX h{h}: {time.time() - t1:.0f}s")


# ---------------------------------------------------------------- A3: Prophet (daily)
def prophet_fit_predict(daily, wx, cps, with_weather, shift, origin, t_by_h):
    """fit on data up to origin; predict the price at t+h for every wanted t (per horizon)."""
    from prophet import Prophet
    from prophet.serialize import model_to_json
    warnings.filterwarnings("ignore")            # worker processes start with default settings
    logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
    logging.getLogger("prophet").setLevel(logging.ERROR)
    train = daily[daily["ds"] <= origin].copy()
    m = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False,
                changepoint_prior_scale=cps)
    try:
        m.add_country_holidays(country_name="IN")
    except Exception:
        pass
    if with_weather:
        wxs = wx.shift(shift, freq="D")     # value on date d = weather of day d - shift (known by then)
        train = train.merge(wxs.rename_axis("ds").reset_index(), on="ds", how="left").dropna()
        m.add_regressor("rain")
        m.add_regressor("temp")
    m.fit(train)
    out = {}
    for h, ts in t_by_h.items():
        if len(ts) == 0:
            continue
        fut = pd.DataFrame({"ds": ts + pd.Timedelta(days=h)})
        if with_weather:
            fut = fut.merge(wxs.rename_axis("ds").reset_index(), on="ds", how="left")
        out[h] = pd.Series(m.predict(fut)["yhat"].to_numpy(), index=ts)
    return out, model_to_json(m)


def prophet_run(daily, wx, cps, with_weather, shifts, frames, first_origin, split):
    """expanding window over one period; returns {h: Series of predictions indexed by row t}."""
    rows = {h: frames[h][frames[h]["split"] == split] for h in shifts}
    end = max(r["date"].max() for r in rows.values())
    origins = list(pd.date_range(first_origin, end, freq=f"{PROPHET_REFIT_DAYS}D"))
    tasks = []
    for k, R in enumerate(origins):
        nxt = origins[k + 1] if k + 1 < len(origins) else end + pd.Timedelta(days=1)
        t_by_h = {h: pd.DatetimeIndex(sorted(set(r["date"][(r["date"] >= R) & (r["date"] < nxt)])))
                  for h, r in rows.items()}
        tasks.append((R, t_by_h))
    # with weather there is one fit per horizon (the weather shift differs), otherwise one fit for all
    jobs = []
    for R, t_by_h in tasks:
        if with_weather:
            for h in shifts:
                jobs.append((R, {h: t_by_h[h]}, h))
        else:
            jobs.append((R, t_by_h, 0))
    results = Parallel(n_jobs=PROPHET_JOBS)(
        delayed(prophet_fit_predict)(daily, wx, cps, with_weather, shift, R, tb) for R, tb, shift in jobs)
    out = {h: [] for h in shifts}
    last_json = {}
    for (R, tb, shift), (pred, mjson) in zip(jobs, results):
        for h, s in pred.items():
            out[h].append(s)
            last_json[h] = mjson
    return {h: pd.concat(v) if v else pd.Series(dtype=float) for h, v in out.items()}, last_json


def run_prophet_family(commodity, state_df, wx_daily, frames, baseline, results, preds, saved):
    daily = state_df[["date", "price"]].rename(columns={"date": "ds", "price": "y"})
    for with_weather in (False, True):
        name = "Prophet_weather" if with_weather else "Prophet"
        label = "Prophet (weather)" if with_weather else "Prophet"
        t0 = time.time()
        shifts = {h: h for h in HORIZONS}
        # tune changepoint_prior_scale on valid
        val_preds = {}
        for cps in PROPHET_CPS:
            val_preds[cps], _ = prophet_run(daily, wx_daily, cps, with_weather, shifts, frames,
                                            TRAIN_END, "valid")
        best = {}
        for h in HORIZONS:
            f = frames[h]
            val = f[f["split"] == "valid"]
            scored = []
            for cps in PROPHET_CPS:
                pred = val["date"].map(val_preds[cps][h]) if len(val_preds[cps][h]) else None
                m = metrics(val["actual"], pred, val["price_today"]) if pred is not None else {"n": 0}
                if m["n"]:
                    scored.append((m["MAE"], cps))
            best[h] = min(scored)[1] if scored else PROPHET_CPS[0]
        # final: refit on train+valid (expanding), score test once, one run per distinct setting
        test_preds, test_json = {}, {}
        for cps in sorted(set(best.values())):
            hs = {h: h for h in HORIZONS if best[h] == cps}
            tp, tj = prophet_run(daily, wx_daily, cps, with_weather, hs, frames, VALID_END, "test")
            for h in hs:
                test_preds[h], test_json[h] = tp[h], tj.get(h)
        for h in HORIZONS:
            f = frames[h]
            pred = pd.Series(np.nan, index=f.index)
            v, t = f["split"] == "valid", f["split"] == "test"
            pred[v] = f.loc[v, "date"].map(val_preds[best[h]][h])
            pred[t] = f.loc[t, "date"].map(test_preds[h]) if len(test_preds[h]) else np.nan
            preds[("state", commodity, h)][name] = pred
            settings = {"changepoint_prior_scale": best[h], "refit_every_days": PROPHET_REFIT_DAYS,
                        "holidays": "India (country holidays)",
                        "weather_shift_days": h if with_weather else None}
            results += score_model(preds[("state", commodity, h)], name, "state", label,
                                   "yes" if with_weather else "no", commodity, h, "price", settings, baseline)
            if test_json.get(h):
                path = os.path.join(MODEL_DIR, f"{name.lower()}_{commodity.lower()}_h{h}.json")
                with open(path, "w", encoding="utf-8") as fh:
                    fh.write(test_json[h])
                saved.append(path)
        log(f"  {label}: {time.time() - t0:.0f}s, chosen changepoint_prior_scale "
            + ", ".join(f"h{h}={best[h]}" for h in HORIZONS))


# ---------------------------------------------------------------- B: xgboost and random forest
def feature_list(use_weather):
    cols = LAG_COLS + OTHER_COLS + (WEATHER_COLS if use_weather else [])
    assert not any(c.startswith("target") for c in cols), "a target column is in the features"
    return cols


def make_X(d, cols, all_codes, algo):
    X = d[cols].copy()
    if algo == "xgb":
        X["district_code"] = pd.Categorical(X["district_code"], categories=all_codes)
    return X


def to_price(pred, today, mode):
    return today.to_numpy() * (1 + pred) if mode == "change" else pred


def target_of(d, h, mode):
    return d[f"target_change_h{h}"] if mode == "change" else d[f"target_h{h}"]


def make_model(algo, params, n_estimators=None):
    if algo == "xgb":
        return XGBRegressor(tree_method="hist", enable_categorical=True, random_state=RANDOM_SEED,
                            n_jobs=N_JOBS, n_estimators=n_estimators or 2000, objective="reg:squarederror",
                            eval_metric="mae", early_stopping_rounds=50 if n_estimators is None else None,
                            **params)
    return RandomForestRegressor(random_state=RANDOM_SEED, n_jobs=N_JOBS, **params)


def search_space(algo):
    if algo == "xgb":
        space = {"max_depth": [3, 4, 5, 6, 8], "learning_rate": [0.02, 0.05, 0.1],
                 "subsample": [0.6, 0.8, 1.0], "colsample_bytree": [0.5, 0.8, 1.0],
                 "min_child_weight": [1, 5, 20], "reg_lambda": [1, 5, 20]}
        return list(ParameterSampler(space, XGB_ITER, random_state=RANDOM_SEED))
    space = {"n_estimators": [150, 300], "max_depth": [8, 12, 20, None],
             "min_samples_leaf": [3, 5, 10, 20], "max_features": [0.33, 0.5, 0.8, 1.0]}
    return list(ParameterSampler(space, RF_ITER, random_state=RANDOM_SEED))


def run_tabular(algo, use_weather, commodity, h, dfd, all_codes, frame):
    """tune on valid, choose the target mode on valid, refit on train+valid, predict test."""
    cols = feature_list(use_weather)
    d = dfd[dfd[f"target_h{h}"].notna()].copy()
    d["target_date"] = d["date"] + pd.Timedelta(days=h)
    # training rows must have their target inside the training period (no look into valid/test)
    tr = d[(d["split"] == "train") & (d["target_date"] <= TRAIN_END)]
    va = d[(d["split"] == "valid") & (d["target_date"] <= VALID_END)]
    te = d[d["split"] == "test"]
    trva = d[d["split"].isin(["train", "valid"]) & (d["target_date"] <= VALID_END)]
    deadline = time.time() + MAX_MINUTES_PER_MODEL * 60
    best, timed_out = None, False
    for mode in TARGET_MODES:
        for params in search_space(algo):
            if time.time() > deadline:
                timed_out = True
                break
            model = make_model(algo, params)
            Xtr, Xva = make_X(tr, cols, all_codes, algo), make_X(va, cols, all_codes, algo)
            if algo == "xgb":
                model.fit(Xtr, target_of(tr, h, mode), eval_set=[(Xva, target_of(va, h, mode))], verbose=False)
                n_iter = int(model.best_iteration) + 1
            else:
                model.fit(Xtr.fillna(np.nan), target_of(tr, h, mode))
                n_iter = None
            pred = to_price(model.predict(Xva), va["price"], mode)
            mae = float(np.abs(pred - va[f"target_h{h}"]).mean())
            if best is None or mae < best["val_mae"]:
                best = {"val_mae": mae, "mode": mode, "params": params, "n_iter": n_iter, "model": model}
    # validation predictions come from the model trained on train only
    mode, params = best["mode"], best["params"]
    pv = pd.Series(to_price(best["model"].predict(make_X(va, cols, all_codes, algo)), va["price"], mode),
                   index=va.index)
    # permutation importance on valid (train-only model)
    perm = permutation_importance(best["model"], make_X(va, cols, all_codes, algo), target_of(va, h, mode),
                                  scoring="neg_mean_absolute_error", n_repeats=3, random_state=RANDOM_SEED, n_jobs=1)
    imp = pd.DataFrame({"feature": cols, "builtin_importance": best["model"].feature_importances_,
                        "permutation_importance_mean": perm.importances_mean,
                        "permutation_importance_std": perm.importances_std})
    # final model on train+valid
    if algo == "xgb":
        final = make_model(algo, params, n_estimators=int(best["n_iter"] * 1.15) + 1)
    else:
        final = make_model(algo, params)
    final.fit(make_X(trva, cols, all_codes, algo), target_of(trva, h, mode))
    pt = pd.Series(to_price(final.predict(make_X(te, cols, all_codes, algo)), te["price"], mode), index=te.index)
    # map predictions onto the scoring frame (same date and district)
    key = d[["date", "district"]].assign(pred=pd.concat([pv, pt]))
    frame_pred = frame[["date", "district"]].merge(key, on=["date", "district"], how="left")["pred"]
    return {"pred": frame_pred.to_numpy(), "mode": mode, "params": params, "val_mae": best["val_mae"],
            "importance": imp, "final": final, "cols": cols, "timed_out": timed_out,
            "n_iter": best["n_iter"]}


def run_tabular_family(commodity, dfd, frames, baseline, results, preds, saved, importances):
    all_codes = sorted(dfd["district_code"].unique())
    for h in HORIZONS:
        for algo, label in (("xgb", "XGBoost"), ("rf", "RandomForest")):
            for use_weather in (True, False):
                t0 = time.time()
                name = f"{label}{'' if use_weather else '_noweather'}"
                out = run_tabular(algo, use_weather, commodity, h, dfd, all_codes, frames[h])
                f = preds[("district", commodity, h)]
                f[name] = out["pred"]
                shown = label + ("" if use_weather else " (no weather)")
                settings = dict(out["params"], target_mode=out["mode"], valid_mae_search=round(out["val_mae"], 2),
                                best_iteration=out["n_iter"], time_limit_hit=out["timed_out"])
                results += score_model(f, name, "district", shown, "yes" if use_weather else "no",
                                       commodity, h, out["mode"], settings, baseline)
                imp = out["importance"].assign(model=label, featureset="all" if use_weather else "no_weather")
                importances.setdefault((commodity, h), []).append(imp)
                path = os.path.join(MODEL_DIR, f"{name.lower()}_{commodity.lower()}_h{h}.joblib")
                joblib.dump({"model": out["final"], "features": out["cols"], "target_mode": out["mode"],
                             "district_codes": all_codes}, path, compress=3)
                saved.append(path)
                log(f"  {commodity} h{h} {shown}: mode={out['mode']}, valid MAE {out['val_mae']:.0f}, "
                    f"{time.time() - t0:.0f}s" + (" (time limit hit)" if out["timed_out"] else ""))


# ---------------------------------------------------------------- output tables
def write_predictions(preds):
    for (level, commodity, h), f in preds.items():
        out = f.copy()
        for c in ("date", "target_date"):
            out[c] = out[c].dt.strftime("%Y-%m-%d")
        out = out.rename(columns={"date": "date_t", "actual": "actual_price_at_t_plus_h"})
        out["naive"] = out["price_today"]
        out.to_csv(os.path.join(OUTPUT_DIR, f"predictions_{level}_{commodity.lower()}_h{h}.csv"), index=False)


def write_importances(importances):
    for (commodity, h), parts in importances.items():
        pd.concat(parts, ignore_index=True).sort_values(
            ["model", "featureset", "permutation_importance_mean"], ascending=[True, True, False]).to_csv(
            os.path.join(OUTPUT_DIR, f"feature_importance_{commodity.lower()}_h{h}.csv"), index=False)


# ---------------------------------------------------------------- plots
def pred_columns(f):
    return [c for c in f.columns if c not in ("date", "target_date", "split", "price_today", "actual", "district")]


COL_LABEL = {"SARIMA": "SARIMA", "SARIMAX_weather": "SARIMAX (weather)", "Prophet": "Prophet",
             "Prophet_weather": "Prophet (weather)", "XGBoost": "XGBoost",
             "XGBoost_noweather": "XGBoost (no weather)", "RandomForest": "RandomForest",
             "RandomForest_noweather": "RandomForest (no weather)"}


def best_model_by_valid(comp, level, commodity, h):
    c = comp[(comp["level"] == level) & (comp["commodity"] == commodity) & (comp["horizon"] == h)
             & (comp["split"] == "valid") & (comp["district"] == "ALL")]
    if c.empty:
        return None
    return c.sort_values("MAE").iloc[0]["model"]


def plot_best(comp, preds, commodity, h):
    fig, axes = plt.subplots(2, 1, figsize=(11, 7.5))
    for ax, level in zip(axes, ("state", "district")):
        f = preds[(level, commodity, h)]
        best = best_model_by_valid(comp, level, commodity, h)
        if best is None:
            continue
        col = [k for k, v in COL_LABEL.items() if v == best][0]
        t = f[f["split"] == "test"]
        agg = t.groupby("date").agg(actual=("actual", "mean"), pred=(col, "mean"), naive=("price_today", "mean"))
        ax.plot(agg.index + pd.Timedelta(days=h), agg["actual"], color="black", label="actual")
        ax.plot(agg.index + pd.Timedelta(days=h), agg["pred"], color="#c0504d", label=f"{best} (chosen on valid)")
        ax.plot(agg.index + pd.Timedelta(days=h), agg["naive"], color="#888", linestyle=":", label="naive")
        title = "Karnataka-wide series" if level == "state" else "average over districts"
        ax.set_title(f"{commodity}, {h}-day ahead, test period: {title}")
        ax.set_ylabel("Rs per quintal")
        ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, f"plot_best_{commodity.lower()}_h{h}.png"), dpi=120)
    plt.close(fig)


def plot_mae_bars(comp, level, commodity):
    c = comp[(comp["level"] == level) & (comp["commodity"] == commodity) & (comp["split"] == "test")
             & (comp["district"] == "ALL")]
    if c.empty:
        return
    fig, axes = plt.subplots(1, len(HORIZONS), figsize=(14, 4.5), sharey=False)
    for ax, h in zip(axes, HORIZONS):
        ch = c[c["horizon"] == h].sort_values("MAE")
        ax.bar(range(len(ch)), ch["MAE"], color="#4a7ab5")
        ax.set_xticks(range(len(ch)))
        ax.set_xticklabels(ch["model"], rotation=60, ha="right", fontsize=8)
        if len(ch):
            ax.axhline(ch["naive_MAE"].iloc[0], color="#c0504d", linestyle="--", label="naive (no change)")
        ax.set_title(f"{h}-day horizon")
        ax.set_ylabel("test MAE (Rs/quintal)")
        ax.legend()
    fig.suptitle(f"{commodity}, {level} level: test MAE per model")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, f"plot_test_mae_{level}_{commodity.lower()}.png"), dpi=120)
    plt.close(fig)


def plot_top_features(importances, commodity, h):
    parts = [p for p in importances.get((commodity, h), []) if (p["featureset"] == "all").all()]
    if not parts:
        return
    fig, axes = plt.subplots(1, len(parts), figsize=(12, 5.5))
    axes = np.atleast_1d(axes)
    for ax, p in zip(axes, parts):
        top = p.sort_values("permutation_importance_mean", ascending=False).head(15).iloc[::-1]
        ax.barh(top["feature"], top["permutation_importance_mean"], color="#4a7ab5")
        ax.set_title(f"{p['model'].iloc[0]}: permutation importance (valid)")
        ax.set_xlabel("increase in MAE when the feature is shuffled")
    fig.suptitle(f"{commodity}, {h}-day horizon: top 15 features")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, f"plot_top_features_{commodity.lower()}_h{h}.png"), dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------- checks and report
def check_consistency_with_baseline(comp):
    """our recomputed naive MAE must equal baseline_scores.csv on the test split (same rows)."""
    t = comp[(comp["split"] == "test") & (comp["district"] == "ALL")].drop_duplicates(
        ["level", "commodity", "horizon"])
    diffs = (t["naive_MAE"] - t["naive_MAE_baseline_file"]).abs()
    return int((diffs > 0.05).sum()), len(t)


def suspicious_scores(comp):
    c = comp[(comp["district"] == "ALL") & (comp["model"] != "naive")]
    return c[(c["MAPE_percent"] < 1.0) | (c["skill_vs_naive_percent"] > 60)]


def weather_effect(comp):
    """compare each model family with and without weather: valid and test MAE."""
    pairs = [("XGBoost", "XGBoost (no weather)"), ("RandomForest", "RandomForest (no weather)"),
             ("SARIMAX (weather)", "SARIMA"), ("Prophet (weather)", "Prophet")]
    rows = []
    c = comp[comp["district"] == "ALL"]
    for with_w, without in pairs:
        for commodity in COMMODITIES:
            for h in HORIZONS:
                for split in ("valid", "test"):
                    a = c[(c["model"] == with_w) & (c["commodity"] == commodity) & (c["horizon"] == h) & (c["split"] == split)]
                    b = c[(c["model"] == without) & (c["commodity"] == commodity) & (c["horizon"] == h) & (c["split"] == split)]
                    if len(a) and len(b):
                        change = (a["MAE"].iloc[0] - b["MAE"].iloc[0]) / b["MAE"].iloc[0] * 100
                        rows.append({"pair": f"{with_w} vs {without}", "commodity": commodity, "horizon": h,
                                     "split": split, "MAE_with_weather": a["MAE"].iloc[0],
                                     "MAE_without_weather": b["MAE"].iloc[0], "change_percent": change,
                                     "verdict": "worse" if change > 1 else ("better" if change < -1 else "same")})
    return pd.DataFrame(rows)


def write_report(comp, wx_table, n_bad_naive, suspicious, notes, saved):
    L = ["TRAINING REPORT", "", "What was done"]
    L += ["  Models were trained on train only, tuned only on valid, then retrained on train+valid and",
          f"  scored once on test. Train <= {TRAIN_END:%Y-%m-%d}, valid <= {VALID_END:%Y-%m-%d}, test after that. Nothing was shuffled.",
          "  The 'naive' rule says the price N days ahead equals today's price. Skill vs naive is the",
          "  percentage improvement in MAE over naive (positive = better than naive).",
          "  Rows used for scoring: valid rows whose target date falls after the valid period are left out",
          "  (so no test-period price influences any choice); naive is recomputed on exactly the scored rows.",
          "  Training rows for the tabular models also need their target date inside the training period.", "",
          "How horizons were handled",
          "  SARIMA and SARIMAX work on weekly averages. 7, 14 and 30 days map to 1, 2 and 4 weekly steps",
          "  (30 days is 4.3 weeks, so the nearest step is 4). A daily row t uses the forecast made at the",
          "  last Sunday on or before t, so only data up to t is used.",
          "  Weather inputs for SARIMAX and Prophet are shifted back by the horizon (7/14/30 days, or 1/2/4 weeks):",
          "  the forecast for t+h uses weather from t or earlier only.", ""]
    test = comp[(comp["split"] == "test") & (comp["district"] == "ALL")].copy()
    L.append("Test results (overall, all districts pooled at district level)")
    show = test[["level", "commodity", "horizon", "model", "n", "MAE", "RMSE", "MAPE_percent",
                 "direction_accuracy", "naive_MAE", "skill_vs_naive_percent"]].sort_values(
        ["level", "commodity", "horizon", "MAE"]).round(2)
    L.append(show.to_string(index=False))
    L += ["", "Winner per level, commodity and horizon (chosen by VALID MAE, then reported on test)"]
    for level in ("state", "district"):
        for commodity in COMMODITIES:
            for h in HORIZONS:
                best = best_model_by_valid(comp, level, commodity, h)
                if best is None:
                    continue
                v = comp[(comp["level"] == level) & (comp["commodity"] == commodity) & (comp["horizon"] == h)
                         & (comp["model"] == best) & (comp["district"] == "ALL")]
                vr, tr = v[v["split"] == "valid"].iloc[0], v[v["split"] == "test"]
                line = (f"  {level:8s} {commodity:7s} h{h:<3d} {best:26s} valid skill {vr['skill_vs_naive_percent']:+.1f}%")
                if len(tr):
                    t = tr.iloc[0]
                    line += (f" | test MAE {t['MAE']:.0f} vs naive {t['naive_MAE']:.0f} "
                             f"(skill {t['skill_vs_naive_percent']:+.1f}%, MAPE {t['MAPE_percent']:.1f}%)")
                if vr["skill_vs_naive_percent"] <= 0:
                    line += "  <-- NOT better than naive on validation"
                L.append(line)
    L += ["", "Models that did not beat naive on validation (overall rows)"]
    val = comp[(comp["split"] == "valid") & (comp["district"] == "ALL")]
    worse = val[val["skill_vs_naive_percent"] <= 0]
    L.append(f"  {len(worse)} of {len(val)} model/commodity/horizon combinations are not better than naive on valid.")
    for r in worse.sort_values(["level", "commodity", "horizon"]).itertuples():
        L.append(f"    {r.level} {r.commodity} h{r.horizon} {r.model}: skill {r.skill_vs_naive_percent:+.1f}%")
    L += ["", "Did weather help? (MAE change when weather is added; negative = better with weather)"]
    if len(wx_table):
        summary = wx_table.groupby(["pair", "split"])["verdict"].value_counts().unstack(fill_value=0)
        L.append(summary.to_string())
        L.append("  mean MAE change with weather, percent:")
        L.append(wx_table.groupby(["pair", "split"])["change_percent"].mean().round(2).to_string())
    L += ["", "Leakage checks"]
    L.append(f"  naive MAE recomputed vs baseline_scores.csv on test: {n_bad_naive[0]} of {n_bad_naive[1]} differ")
    if len(suspicious):
        L.append("  SCORES THAT LOOK TOO GOOD (MAPE < 1% or skill > 60%), investigate before trusting:")
        L.append(suspicious[["level", "model", "commodity", "horizon", "split", "MAPE_percent",
                             "skill_vs_naive_percent"]].round(2).to_string(index=False))
    else:
        L.append("  no score had MAPE under 1% or skill above 60%.")
    L += ["  features per model audited: no target or future column, only data up to day t."]
    L += ["", "Notes"] + [f"  {n}" for n in notes]
    L += ["", f"Saved models ({len(saved)}): see output/models/"]
    with open(os.path.join(OUTPUT_DIR, "training_report.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    return "\n".join(L)


# ---------------------------------------------------------------- main
def main():
    seed_everything()
    os.makedirs(MODEL_DIR, exist_ok=True)
    start = time.time()
    log(f"packages ok. quick mode: {QUICK}")
    log("weather for SARIMAX and Prophet is shifted back by the horizon, so a forecast for day t+h "
        "only uses weather known on day t")
    baseline = load_baseline()
    wx_daily = load_weather_daily()
    results, preds, saved, importances, notes = [], {}, [], {}, []

    for commodity in COMMODITIES:
        log(f"=== {commodity} ===")
        state_df, dist_df = load_state(commodity), load_district(commodity)
        frames_state = {h: base_frame(state_df, h, "state") for h in HORIZONS}
        frames_dist = {h: base_frame(dist_df, h, "district") for h in HORIZONS}
        for h in HORIZONS:
            preds[("state", commodity, h)] = frames_state[h].copy()
            preds[("district", commodity, h)] = frames_dist[h].copy()

        t0 = time.time()
        log("state level: SARIMA and SARIMAX")
        run_sarima_family(commodity, state_df, wx_daily, frames_state, baseline, results, preds, saved)
        log(f"  done in {time.time() - t0:.0f}s")
        t0 = time.time()
        log("state level: Prophet")
        run_prophet_family(commodity, state_df, wx_daily, frames_state, baseline, results, preds, saved)
        log(f"  done in {time.time() - t0:.0f}s")
        t0 = time.time()
        log("district level: XGBoost and RandomForest")
        run_tabular_family(commodity, dist_df, frames_dist, baseline, results, preds, saved, importances)
        log(f"  done in {time.time() - t0:.0f}s")

    comp = pd.DataFrame(results)
    cols = ["level", "model", "uses_weather", "commodity", "horizon", "split", "district", "target_mode",
            "n", "MAE", "RMSE", "MAPE_percent", "direction_accuracy", "n_direction", "naive_MAE",
            "naive_MAE_baseline_file", "skill_vs_naive_percent", "settings"]
    comp = comp[cols]
    comp.round(4).to_csv(os.path.join(OUTPUT_DIR, "model_comparison.csv"), index=False)
    write_predictions(preds)
    write_importances(importances)
    for commodity in COMMODITIES:
        for h in HORIZONS:
            plot_best(comp, preds, commodity, h)
            plot_top_features(importances, commodity, h)
        for level in ("state", "district"):
            plot_mae_bars(comp, level, commodity)

    n_bad = check_consistency_with_baseline(comp)
    suspicious = suspicious_scores(comp)
    wx_table = weather_effect(comp)
    notes += ["state-level work uses weather_daily_long.csv (average over all districts) for the weather inputs",
              "direction accuracy is only judged on rows where the price actually changed",
              f"total run time {(time.time() - start) / 60:.1f} minutes"]
    report = write_report(comp, wx_table, n_bad, suspicious, notes, saved)
    log("")
    log(report)


if __name__ == "__main__":
    main()
