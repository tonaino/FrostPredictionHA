"""Hybrid frost prediction model.

Layer 1 — FAO empirical radiation-frost model (Snyder & de Melo-Abreu 2005):
    Tmin = a * T_sunset + b * Td_sunset + c
  coefficients fitted locally on observed (evening -> next Tmin) pairs.
  Valid for radiative nights (low wind, no clouds); blended by a wind/cloud
  correction factor.

Layer 2 — Gradient-boosted trees / Random-Forest classifier trained on
  engineered features -> P(frost next morning). Combines physics prior with
  ML pattern learning, following Talsma et al. (2023, Frontiers in AI) and
  Eccel et al. (2007).
"""
import json
import math
import os

import numpy as np

import db
import features as F

MODEL_DIR = os.environ.get("FROST_MODEL_DIR") or os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "models"
)
MODEL_PATH = os.path.join(MODEL_DIR, "frost_model.joblib")
META_PATH = os.path.join(MODEL_DIR, "frost_model_meta.json")
REG_MODEL_PATH = os.path.join(MODEL_DIR, "frost_tmin_regressor.joblib")
REG_META_PATH = os.path.join(MODEL_DIR, "frost_tmin_regressor_meta.json")

LOCAL_SOURCES = ("ha", "ecowitt")
UNAVAILABLE_AT_EVENING = (
    "temp_max", "temp_mean", "temp_min", "dewpoint_min",
    "humidity_mean", "pressure_change",
)


# ------------------------------------------------------------------ FAO layer

def fit_fao(rows):
    """Least squares fit of Tmin_next ~ a*T_sunset + b*Td_sunset + c using
    radiative nights only (wind < 2 m/s-ish, i.e. low wind_max). Returns (a, b, c)."""
    X, y = [], []
    for r in rows:
        t = r["temp_sunset"]
        td = r["dewpoint_sunset"]
        tmin = r["next_temp_min"]
        if t is None or td is None or tmin is None:
            continue
        wind = r["wind_max"] if r["wind_max"] is not None else 0.0
        cloud = r["cloud_mean"] if r["cloud_mean"] is not None else 50.0
        if wind <= 2.0 and cloud <= 40.0:  # radiative nights
            X.append([t, td, 1.0])
            y.append(tmin)
    if len(y) < 10:
        # fall back to all nights
        for r in rows:
            t, td, tmin = r["temp_sunset"], r["dewpoint_sunset"], r["next_temp_min"]
            if t is None or td is None or tmin is None:
                continue
            X.append([t, td, 1.0])
            y.append(tmin)
    if len(y) < 10:
        # literature default coefficients (FAO)
        return (0.285, 0.328, -0.5)
    A = np.array(X, dtype=float)
    b = np.array(y, dtype=float)
    coef, *_ = np.linalg.lstsq(A, b, rcond=None)
    return tuple(float(c) for c in coef)


def fao_predict(t_sunset, td_sunset, coef):
    a, b, c = coef
    return a * t_sunset + b * td_sunset + c


def local_temperature_bias(db_path=None, limit=14, cap=5.0):
    """Recent local forecast error, in °C, for online bias correction.

    Positive values mean recent observations were warmer than predicted;
    negative values mean the local station was colder.  A short, capped,
    recency-weighted correction adapts to local cold-air pooling without
    allowing one bad observation to produce an unbounded forecast shift.
    """
    conn = db.get_conn(db_path)
    rows = conn.execute(
        """SELECT observed_tmin, tmin_empirical
           FROM predictions
           WHERE observed_tmin IS NOT NULL AND tmin_empirical IS NOT NULL
           ORDER BY target_date DESC, prediction_id DESC
           LIMIT ?""",
        (limit,),
    ).fetchall()
    if not rows:
        return 0.0
    # Newest observation has weight 1, then 0.9, 0.8, ...
    weights = [max(0.1, 1.0 - i * 0.1) for i in range(len(rows))]
    bias = sum(w * (r["observed_tmin"] - r["tmin_empirical"])
               for w, r in zip(weights, rows)) / sum(weights)
    return max(-cap, min(cap, float(bias)))


def _forecast_safe_features(feats):
    """Remove same-day values that are unavailable at evening forecast time."""
    safe = dict(feats)
    for key in UNAVAILABLE_AT_EVENING:
        safe[key] = None
    return safe


def train_tmin_regressor(db_path=None, random_state=42):
    """Train a local-station Tmin regressor using forecast-safe features."""
    import joblib
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.model_selection import TimeSeriesSplit

    conn = db.get_conn(db_path)
    rows = conn.execute(
        """SELECT o.*, l.temp_min AS next_temp_min,
                  l.temp_min_source AS next_temp_min_source
           FROM observations o
           JOIN observations l ON l.station_id = o.station_id
            AND l.date = date(o.date, '+1 day')
           WHERE o.temp_sunset IS NOT NULL
             AND o.dewpoint_sunset IS NOT NULL
             AND o.temp_sunset_source IN ('ha', 'ecowitt')
             AND o.dewpoint_sunset_source IN ('ha', 'ecowitt')
             AND l.temp_min IS NOT NULL
             AND l.temp_min_source IN ('ha', 'ecowitt')
           ORDER BY o.date"""
    ).fetchall()
    if len(rows) < 60:
        raise RuntimeError(f"Not enough local rows for Tmin regression ({len(rows)})")
    clim_rows = conn.execute("SELECT * FROM climate_history").fetchall()
    clim = F.day_of_year_frost_climatology(clim_rows, rows[0]["station_id"])
    X, y = [], []
    for row in rows:
        feats = _forecast_safe_features(F.build_features(row, clim))
        X.append(F.features_to_vector(feats))
        y.append(row["next_temp_min"])
    X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    X_imp = imputer.fit_transform(X)
    reg = RandomForestRegressor(
        n_estimators=500, min_samples_leaf=4, max_features="sqrt",
        random_state=random_state, n_jobs=-1,
    )
    cv = []
    for tr, te in TimeSeriesSplit(n_splits=5).split(X_imp):
        fold = RandomForestRegressor(
            n_estimators=300, min_samples_leaf=4, max_features="sqrt",
            random_state=random_state, n_jobs=-1,
        )
        fold.fit(X_imp[tr], y[tr])
        pred = fold.predict(X_imp[te])
        cv.append({"mae": float(mean_absolute_error(y[te], pred)),
                   "rmse": float(np.sqrt(mean_squared_error(y[te], pred))),
                   "r2": float(r2_score(y[te], pred))})
    reg.fit(X_imp, y)
    joblib.dump({"imputer": imputer, "model": reg}, REG_MODEL_PATH)
    meta = {
        "trained_at": __import__("datetime").datetime.now().isoformat(timespec="seconds"),
        "n_samples": len(y), "cv": cv,
        "cv_mean": {k: float(np.mean([x[k] for x in cv])) for k in cv[0]},
        "feature_names": F.FEATURE_NAMES,
        "source_priority": "ha/ecowitt only",
    }
    with open(REG_META_PATH, "w") as f:
        json.dump(meta, f, indent=2)
    return meta


def load_tmin_regressor():
    import joblib
    if not os.path.exists(REG_MODEL_PATH):
        return None, None
    with open(REG_META_PATH) as f:
        meta = json.load(f)
    return joblib.load(REG_MODEL_PATH), meta


def radiative_night_factor(wind_max, cloud_mean):
    """0 (advection/mixed) .. 1 (pure radiation night). FAO validity gate."""
    if wind_max is None:
        wind_max = 0.0
    if cloud_mean is None:
        cloud_mean = 50.0
    f_wind = max(0.0, 1.0 - wind_max / 4.0)
    f_cloud = max(0.0, 1.0 - cloud_mean / 80.0)
    return f_wind * f_cloud


# ------------------------------------------------------------------ ML layer

def train(db_path=None, random_state=42):
    import joblib
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score, brier_score_loss, f1_score
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestClassifier, StackingClassifier

    os.makedirs(MODEL_DIR, exist_ok=True)
    rows = db.get_training_data(db_path)
    if len(rows) < 60:
        raise RuntimeError(
            f"Not enough labeled observations to train ({len(rows)} rows). "
            "Need at least 60 days of collected data. Run the collector daily."
        )

    clim_rows = db.get_conn(db_path).execute("SELECT * FROM climate_history").fetchall()
    climatology = F.day_of_year_frost_climatology(clim_rows, rows[0]["station_id"])

    X, y, dates = [], [], []
    for r in rows:
        feats = F.build_features(r, climatology)
        X.append(F.features_to_vector(feats))
        y.append(1 if (r["next_temp_min"] is not None and r["next_temp_min"] <= 0.0) else 0)
        dates.append(r["date"])

    X = np.array(X, dtype=float)
    y = np.array(y)
    n_frost = int(y.sum())

    base_rf = RandomForestClassifier(
        n_estimators=400, min_samples_leaf=3, max_features="sqrt",
        class_weight="balanced_subsample", random_state=random_state, n_jobs=-1,
    )
    base_gb = GradientBoostingClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=3,
        subsample=0.9, random_state=random_state,
    )
    stack = StackingClassifier(
        estimators=[("rf", base_rf), ("gb", base_gb)],
        final_estimator=LogisticRegression(max_iter=1000, class_weight="balanced"),
        stack_method="predict_proba", cv=3, n_jobs=-1,
    )
    pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("model", stack),
    ])

    # time-series CV (2 folds minimum)
    n_splits = min(5, max(2, len(y) // 60))
    cv_auc, cv_f1, cv_brier = [], [], []
    if n_splits >= 2 and n_frost > 0:
        tscv = TimeSeriesSplit(n_splits=n_splits)
        for tr, te in tscv.split(X):
            if len(np.unique(y[tr])) < 2 or len(te) == 0:
                continue
            pipe.fit(X[tr], y[tr])
            p = pipe.predict_proba(X[te])[:, 1]
            if len(np.unique(y[te])) == 2:
                cv_auc.append(roc_auc_score(y[te], p))
            cv_brier.append(brier_score_loss(y[te], p))
            cv_f1.append(f1_score(y[te], (p >= 0.5).astype(int), zero_division=0))

    pipe.fit(X, y)

    fao_coef = fit_fao(rows)

    meta = {
        "trained_at": __import__("datetime").datetime.now().isoformat(timespec="seconds"),
        "feature_names": F.FEATURE_NAMES,
        "fao_coef": list(fao_coef),
        "climatology_snapshot": {str(k): v for k, v in climatology.items()},
        "n_samples": len(y),
        "n_frost": n_frost,
        "cv": {
            "auc": float(np.mean(cv_auc)) if cv_auc else None,
            "brier": float(np.mean(cv_brier)) if cv_brier else None,
            "f1": float(np.mean(cv_f1)) if cv_f1 else None,
        },
    }
    joblib.dump(pipe, MODEL_PATH)
    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)

    version_id = db.save_model_version(
        trained_at=meta["trained_at"],
        algorithm="stacked_rf_gb + fao_hybrid",
        n_samples=len(y),
        n_frost=n_frost,
        metrics=json.dumps(meta["cv"]),
        model_path=MODEL_PATH,
        db_path=db_path,
    )
    return meta, version_id


def load_model():
    import joblib
    if not os.path.exists(MODEL_PATH):
        return None, None
    pipe = joblib.load(MODEL_PATH)
    with open(META_PATH) as f:
        meta = json.load(f)
    return pipe, meta


# ------------------------------------------------------------------ inference

def predict_frost_probability(feats, pipe, meta, db_path=None, regressor=None):
    """Returns (probability, tmin_ml_estimate, tmin_fao).

    tmin_ml_estimate: regression-style estimate via blended physics prior when
    the classifier is not yet trained on enough frost events; when the model
    was trained on real frost events, probability comes from the classifier.
    """
    fao_coef = tuple(meta.get("fao_coef", (0.285, 0.328, -0.5)))
    t = feats.get("temp_sunset")
    td = feats.get("dewpoint_sunset")
    if t is None or td is None:
        raise ValueError("temp_sunset and dewpoint_sunset are required")
    tmin_fao = fao_predict(t, td, fao_coef)
    tmin_raw = tmin_fao
    if regressor is not None:
        safe = _forecast_safe_features(feats)
        x = np.asarray([F.features_to_vector(safe)], dtype=float)
        tmin_raw = float(regressor["model"].predict(regressor["imputer"].transform(x))[0])
    bias_c = local_temperature_bias(db_path)
    tmin_estimate = tmin_raw + bias_c
    rad = radiative_night_factor(feats.get("wind_max"), feats.get("cloud_mean"))

    # physics-based probability: logistic mapping on predicted Tmin margin
    # P ~ sigmoid(-(Tmin - 0)/width): Tmin below 0 -> P > 0.5
    width = 1.5  # C
    p_phys = 1.0 / (1.0 + math.exp((tmin_estimate + 0.5) / width))
    # blend toward climatology for advection nights where FAO is unreliable
    p_clim = feats.get("clim_frost", 0.0)
    p_blend = rad * p_phys + (1.0 - rad) * max(p_phys, p_clim)

    x = np.array([F.features_to_vector(feats)], dtype=float)
    p_ml = None
    tmin_ml = None
    if pipe is not None:
        p_ml = float(pipe.predict_proba(x)[0, 1])
        # combined: geometric blend weighted by ML trust (more frost samples = more trust)
        trust = min(1.0, meta.get("n_frost", 0) / 30.0)
        p_final = (p_ml ** trust) * (p_blend ** (1.0 - trust))
        if trust == 0:
            p_final = p_blend
        tmin_ml = tmin_fao  # regression head can be added later
    else:
        p_final, p_ml = p_blend, None

    return float(p_final), tmin_ml, tmin_estimate, {"p_ml": p_ml, "p_phys": p_phys,
                                                "p_clim": p_clim, "rad_factor": rad,
                                                "tmin_raw": tmin_raw, "tmin_fao": tmin_fao,
                                                "local_bias_c": bias_c}
