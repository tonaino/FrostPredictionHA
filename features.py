"""Feature engineering for frost prediction.

Design based on literature (Snyder & de Melo-Abreu, FAO 2005; Talsma et al. 2023,
Frontiers in AI; Eccel et al. 2007):
  - Evening temperature + dewpoint (2h after sunset) drive radiation-frost physics
    (FAO: Tmin = a*T + b*Td + c)
  - Humidity proxies radiative cooling and condensation heat release
  - Wind suppresses nocturnal inversion (advection mixing)
  - Cloud cover blocks radiative loss
  - Soil moisture increases thermal inertia of the surface
  - Dewpoint spread (T - Td) large => dry air => strong radiative cooling
  - Climatology (doy frost climatology) gives the seasonal prior
"""
import math

FEATURE_NAMES = [
    "temp_sunset", "dewpoint_sunset", "dewpoint_spread_sunset",
    "temp_max", "temp_mean", "temp_min",
    "dewpoint_min", "humidity_min", "humidity_mean",
    "wind_max", "gust_max", "pressure_change", "cloud_mean", "cloud_min",
    "precipitation", "soil_moisture",
    "doy_sin", "doy_cos", "clim_frost",
]


# ---------------------------------------------------------------- climatology

def day_of_year_frost_climatology(climate_rows, station_id, window=7):
    """Per-day-of-year frost probability from the 10y climate_history table,
    smoothed with +/- window days. Returns {doy: p}."""
    import datetime as dt
    counts = {d: [0, 0] for d in range(1, 368)}  # doy -> [frost, total]
    for r in climate_rows:
        if r["station_id"] != station_id or r["temp_min"] is None:
            continue
        doy = dt.date.fromisoformat(r["date"]).timetuple().tm_yday
        counts[doy][1] += 1
        if r["temp_min"] <= 0.0:
            counts[doy][0] += 1
    smooth = {}
    for d in range(1, 367):
        f = t = 0
        for off in range(-window, window + 1):
            dd = ((d - 1 + off) % 366) + 1
            f += counts[dd][0]
            t += counts[dd][1]
        smooth[d] = (f + 1.0) / (t + 2.0) if t > 0 else 0.0
    return smooth


# ---------------------------------------------------------------- features

def build_features(row, climatology):
    """row: observations row (dict/Row) for day D; prediction target = night D->D+1."""
    import datetime as dt

    def g(k, default=None):
        try:
            v = row[k]
        except (KeyError, IndexError):
            return default
        return v if v is not None else default

    date = dt.date.fromisoformat(row["date"])
    doy = date.timetuple().tm_yday
    t_sunset = g("temp_sunset")
    td_sunset = g("dewpoint_sunset")
    spread = (t_sunset - td_sunset) if (t_sunset is not None and td_sunset is not None) else None

    feats = {
        "temp_sunset": t_sunset,
        "dewpoint_sunset": td_sunset,
        "dewpoint_spread_sunset": spread,
        "temp_max": g("temp_max"),
        "temp_mean": g("temp_mean"),
        "temp_min": g("temp_min"),
        "dewpoint_min": g("dewpoint_min"),
        "humidity_min": g("humidity_min"),
        "humidity_mean": g("humidity_mean"),
        "wind_max": g("wind_max"),
        "gust_max": g("gust_max"),
        "pressure_change": g("pressure_change"),
        "cloud_mean": g("cloud_mean"),
        "cloud_min": g("cloud_min"),
        "precipitation": g("precipitation"),
        "soil_moisture": g("soil_moisture"),
        "doy_sin": math.sin(2 * math.pi * doy / 365.25),
        "doy_cos": math.cos(2 * math.pi * doy / 365.25),
        "clim_frost": climatology.get(doy, 0.0),
    }
    return feats


def features_to_vector(feats):
    return [feats.get(k) for k in FEATURE_NAMES]
