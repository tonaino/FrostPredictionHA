"""Ecowitt rolling-window Tmin regression model.

This is a shadow model inspired by Talsma et al.: it predicts the minimum
temperature from the evening through the following morning using local
high-frequency samples. It is deliberately kept separate from the existing
FAO/classifier path until it wins a clean time-held-out comparison.
"""
import datetime as dt
import json
import math
import os

import numpy as np

import db

MODEL_DIR = os.environ.get("FROST_MODEL_DIR") or os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "models"
)
MODEL_PATH = os.path.join(MODEL_DIR, "rolling_tmin_rf.joblib")
META_PATH = os.path.join(MODEL_DIR, "rolling_tmin_rf_meta.json")
FEATURE_NAMES = [
    "evening_temp", "evening_dewpoint", "temp_min_6h", "temp_max_6h",
    "temp_mean_6h", "dewpoint_min_6h", "humidity_mean_6h", "wind_max_6h",
    "gust_max_6h", "pressure_change_6h", "radiation_mean_6h",
]


def _samples(db_path=None):
    rows = db.get_conn(db_path).execute(
        "SELECT * FROM ecowitt_samples ORDER BY timestamp"
    ).fetchall()
    out = {}
    for r in rows:
        try:
            timestamp = dt.datetime.fromisoformat(r["timestamp"])
        except (TypeError, ValueError):
            continue
        out.setdefault(timestamp.date().isoformat(), []).append((timestamp, r))
    return out


def _stats(day_samples, start_hour, end_hour):
    selected = [r for t, r in day_samples if start_hour <= t.hour <= end_hour and r["temp"] is not None]
    if not selected:
        return {}
    def vals(k):
        return [float(r[k]) for r in selected if r[k] is not None]
    temps = vals("temp")
    dew = vals("dewpoint")
    humidity = vals("humidity")
    wind = vals("wind")
    gust = vals("gust")
    pressure = vals("pressure")
    radiation = vals("radiation")
    result = {"temp_min_6h": min(temps), "temp_max_6h": max(temps),
              "temp_mean_6h": sum(temps) / len(temps)}
    for key, values, fn in [
        ("dewpoint_min_6h", dew, min), ("humidity_mean_6h", humidity, statistics_mean),
        ("wind_max_6h", wind, max), ("gust_max_6h", gust, max),
        ("radiation_mean_6h", radiation, statistics_mean),
    ]:
        if values:
            result[key] = fn(values)
    if len(pressure) >= 2:
        result["pressure_change_6h"] = pressure[-1] - pressure[0]
    return result


def statistics_mean(values):
    return sum(values) / len(values)


def build_training_rows(db_path=None):
    samples = _samples(db_path)
    conn = db.get_conn(db_path)
    rows = conn.execute(
        """SELECT o.*, l.temp_min AS next_temp_min
           FROM observations o JOIN observations l
             ON l.station_id=o.station_id AND l.date=date(o.date,'+1 day')
           WHERE o.temp_sunset IS NOT NULL AND o.dewpoint_sunset IS NOT NULL
             AND o.temp_sunset_source IN ('ha','ecowitt')
             AND o.dewpoint_sunset_source IN ('ha','ecowitt')
             AND l.temp_min IS NOT NULL AND l.temp_min_source IN ('ha','ecowitt')
           ORDER BY o.date"""
    ).fetchall()
    result = []
    for row in rows:
        day = row["date"]
        evening = _stats(samples.get(day, []), 17, 23)
        next_day = (dt.date.fromisoformat(day) + dt.timedelta(days=1)).isoformat()
        overnight = [float(r["temp"]) for t, r in samples.get(next_day, [])
                     if 0 <= t.hour <= 9 and r["temp"] is not None]
        if not evening or not overnight:
            continue
        features = {name: None for name in FEATURE_NAMES}
        features.update(evening)
        features["evening_temp"] = row["temp_sunset"]
        features["evening_dewpoint"] = row["dewpoint_sunset"]
        result.append((features, min(overnight)))
    return result


def train(db_path=None, random_state=42):
    import joblib
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    from sklearn.model_selection import TimeSeriesSplit

    rows = build_training_rows(db_path)
    if len(rows) < 60:
        raise RuntimeError(f"Not enough rolling-window rows ({len(rows)})")
    X = np.array([[r[0].get(k) for k in FEATURE_NAMES] for r in rows], dtype=float)
    y = np.array([r[1] for r in rows], dtype=float)
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    X = imputer.fit_transform(X)
    cv = []
    for tr, te in TimeSeriesSplit(n_splits=5).split(X):
        rf = RandomForestRegressor(n_estimators=400, min_samples_leaf=3,
                                   max_features="sqrt", random_state=random_state,
                                   n_jobs=-1)
        rf.fit(X[tr], y[tr]); pred = rf.predict(X[te])
        cv.append({"mae": float(mean_absolute_error(y[te], pred)),
                   "rmse": float(math.sqrt(mean_squared_error(y[te], pred)))})
    rf = RandomForestRegressor(n_estimators=500, min_samples_leaf=3,
                               max_features="sqrt", random_state=random_state,
                               n_jobs=-1)
    rf.fit(X, y)
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump({"imputer": imputer, "model": rf}, MODEL_PATH)
    meta = {"trained_at": dt.datetime.now().isoformat(timespec="seconds"),
            "n_samples": len(y), "features": FEATURE_NAMES, "cv": cv,
            "cv_mean": {k: float(np.mean([x[k] for x in cv])) for k in cv[0]}}
    with open(META_PATH, "w") as f: json.dump(meta, f, indent=2)
    return meta


def load():
    import joblib
    if not os.path.exists(MODEL_PATH): return None, None
    with open(META_PATH) as f: meta = json.load(f)
    return joblib.load(MODEL_PATH), meta


def predict_for_observation(row, bundle, db_path=None):
    """Return a shadow RF estimate for an evening observation, if possible."""
    if bundle is None or row is None:
        return None
    samples = _samples(db_path)
    features = {name: None for name in FEATURE_NAMES}
    features.update(_stats(samples.get(row["date"], []), 17, 23))
    features["evening_temp"] = row["temp_sunset"]
    features["evening_dewpoint"] = row["dewpoint_sunset"]
    x = np.array([[features.get(k) for k in FEATURE_NAMES]], dtype=float)
    return float(bundle["model"].predict(bundle["imputer"].transform(x))[0])
