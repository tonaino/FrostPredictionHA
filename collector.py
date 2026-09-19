"""Data collector for the frost-forecast system.

Sources:
  1. Open-Meteo archive (10y daily climate history) -> climatology + bootstrap
  2. Open-Meteo forecast/archive API (hourly) -> daily aggregates incl. the
     evening snapshot taken exactly 2h after sunset (FAO measurement time)
  3. Home Assistant weather station (optional token) -> station-specific data
  4. ha_snapshot.json written via MCP -> station evening snapshot import

Usage:
  python collector.py backfill          # 10y Open-Meteo climate history
  python collector.py pull-history      # last 60d daily aggregates (hourly-based)
  python collector.py bootstrap         # seed observations from climate history
  python collector.py snapshot          # evening snapshot from Open-Meteo (sunset+2h)
  python collector.py import-ha         # import ha_snapshot.json (Bernacca live values)
  python collector.py outcomes          # backfill observed next-morning Tmin + scoring
"""
import json
import os
import sys
from datetime import date, datetime, timedelta

import requests

import db

HA_URL = os.environ.get("HA_URL", "http://192.168.31.200:8123")
HA_TOKEN = os.environ.get("HA_TOKEN", "")
HA_TOKEN_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ha_token.txt")
HA_SNAPSHOT_FILE = os.environ.get("HA_SNAPSHOT_FILE") or os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "ha_snapshot.json"
)

STATION_ID = "gorna_malina"
STATION_NAME = "Gorna Malina / MeteoCasa"
LAT, LON = 42.25, 23.85
TZ = "Europe/Sofia"

UA = {"User-Agent": "frost-collector/1.0"}

# HA long-term statistics entity ids (weather station, unit-aware)
HA_SENSORS = {
    "sensor.bernacca_outdoor_temperature": "temperature",
    "sensor.bernacca_dewpoint": "dewpoint",
    "sensor.bernacca_humidity": "humidity",
    "sensor.bernacca_wind_speed": "wind",
    "sensor.bernacca_relative_pressure": "pressure",
    "sensor.bernacca_solar_radiation": "radiation",
}


# --------------------------------------------------------------- open-meteo

def backfill_climate(db_path=None):
    """10 years of daily climate history from Open-Meteo archive."""
    print("Fetching Open-Meteo archive (2016-09-18 .. yesterday)...")
    start = (date.today() - timedelta(days=3650)).isoformat()
    end = (date.today() - timedelta(days=1)).isoformat()
    url = (
        "https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={LAT}&longitude={LON}"
        f"&start_date={start}&end_date={end}"
        "&daily=temperature_2m_min,temperature_2m_max,dew_point_2m_min,"
        "precipitation_sum,windspeed_10m_max,cloudcover_mean"
        f"&timezone={TZ}"
    )
    r = requests.get(url, timeout=120, headers=UA)
    r.raise_for_status()
    daily = r.json()["daily"]
    rows = []
    for i, d in enumerate(daily["time"]):
        rows.append({
            "date": d,
            "temp_min": daily["temperature_2m_min"][i],
            "temp_max": daily["temperature_2m_max"][i],
            "dewpoint_min": daily["dew_point_2m_min"][i],
            "precipitation": daily["precipitation_sum"][i],
            "wind_max": daily["windspeed_10m_max"][i],
            "cloud_mean": daily["cloudcover_mean"][i],
        })
    db.upsert_station(STATION_ID, STATION_NAME, LAT, LON, TZ, db_path)
    db.upsert_climate_rows(STATION_ID, rows, db_path)
    print(f"Imported {len(rows)} climate-history rows.")


def _hourly_daily(days_back=60, db_path=None):
    """Fetch hourly data for the last N days and derive daily aggregates,
    including evening values 2h after sunset (FAO measurement time)."""
    end = date.today().isoformat()
    start = (date.today() - timedelta(days=days_back)).isoformat()
    url = (
        "https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={LAT}&longitude={LON}"
        f"&start_date={start}&end_date={end}"
        "&hourly=temperature_2m,dew_point_2m,relative_humidity_2m,"
        "windspeed_10m,windgusts_10m,cloudcover,surface_pressure,shortwave_radiation"
        "&daily=temperature_2m_min,temperature_2m_max,temperature_2m_mean,"
        "dew_point_2m_min,relative_humidity_2m_mean,relative_humidity_2m_min,"
        "windspeed_10m_max,windgusts_10m_max,cloudcover_mean,cloudcover_min,"
        "precipitation_sum,sunrise,sunset"
        f"&timezone={TZ}"
    )
    r = requests.get(url, timeout=120, headers=UA)
    r.raise_for_status()
    j = r.json()
    hourly, daily = j["hourly"], j["daily"]

    h_by_day = {}
    for i, ts in enumerate(hourly["time"]):
        h_by_day.setdefault(ts[:10], []).append(i)

    db.upsert_station(STATION_ID, STATION_NAME, LAT, LON, TZ, db_path)
    n = 0
    for i, d in enumerate(daily["time"]):
        idx = h_by_day.get(d, [])
        if not idx:
            continue
        # evening = 2h after sunset
        sunset_h = int(daily["sunset"][i][11:13])
        ev1, ev2 = sunset_h + 2, (sunset_h + 3) % 24
        if ev1 < len(hourly["time"]) and hourly["time"][ev1][:10] == d:
            pass
        # find evening indices for this day
        ev_idx = [k for k in idx if int(hourly["time"][k][11:13]) in (sunset_h + 2, sunset_h + 3)]
        t_sunset = (sum(hourly["temperature_2m"][k] for k in ev_idx) / len(ev_idx)) if ev_idx else None
        td_sunset = (sum(hourly["dew_point_2m"][k] for k in ev_idx) / len(ev_idx)) if ev_idx else None
        fields = {
            "temp_min": daily["temperature_2m_min"][i],
            "temp_max": daily["temperature_2m_max"][i],
            "temp_mean": daily["temperature_2m_mean"][i],
            "dewpoint_min": daily["dew_point_2m_min"][i],
            "humidity_mean": daily["relative_humidity_2m_mean"][i],
            "humidity_min": daily["relative_humidity_2m_min"][i],
            "wind_max": daily["windspeed_10m_max"][i],
            "gust_max": daily["windgusts_10m_max"][i],
            "cloud_mean": daily["cloudcover_mean"][i],
            "cloud_min": daily["cloudcover_min"][i],
            "precipitation": daily["precipitation_sum"][i],
            "temp_sunset": t_sunset,
            "dewpoint_sunset": td_sunset,
            "radiation_max": max((hourly["shortwave_radiation"][k] for k in idx), default=None),
        }
        db.upsert_observation(STATION_ID, d, fields, source="openmeteo", db_path=db_path)
        n += 1
    print(f"Upserted {n} observation rows from Open-Meteo hourly.")


def pull_history(db_path=None, days=60):
    _hourly_daily(days_back=days, db_path=db_path)


def bootstrap(db_path=None, years=3):
    """Seed observation rows from climate_history (last `years` years) with
    estimated evening fields so the model can train before real data accrues.
    T_sunset ~ Tmax - 5.0, Td_sunset ~ Td_min + 1.0 (semi-continental heuristics)."""
    conn = db.get_conn(db_path)
    rows = conn.execute(
        """SELECT date, temp_min, temp_max, dewpoint_min, precipitation,
                  wind_max, cloud_mean FROM climate_history
           WHERE station_id = ? AND date >= date('now', ?)
           ORDER BY date""",
        (STATION_ID, f"-{years * 365} days"),
    ).fetchall()
    for r in rows:
        if r["temp_min"] is None or r["temp_max"] is None:
            continue
        fields = {
            "temp_min": r["temp_min"],
            "temp_max": r["temp_max"],
            "temp_mean": (r["temp_min"] + r["temp_max"]) / 2.0,
            "dewpoint_min": r["dewpoint_min"],
            "dewpoint_sunset": (r["dewpoint_min"] + 1.0) if r["dewpoint_min"] is not None else None,
            "temp_sunset": r["temp_max"] - 5.0,
            "wind_max": r["wind_max"],
            "cloud_mean": r["cloud_mean"],
            "precipitation": r["precipitation"],
        }
        db.upsert_observation(STATION_ID, r["date"], fields, source="openmeteo", db_path=db_path)
    print(f"Bootstrapped {len(rows)} observation rows from climate history.")


# --------------------------------------------------------------- HA helpers

def has_token() -> bool:
    """True if a HA token is available (env var or file)."""
    if HA_TOKEN:
        return True
    return os.path.exists(HA_TOKEN_FILE) and os.path.getsize(HA_TOKEN_FILE) > 0


def _ha_headers():
    token = HA_TOKEN
    if not token and os.path.exists(HA_TOKEN_FILE):
        with open(HA_TOKEN_FILE) as f:
            token = f.read().strip()
    return {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}


def import_ha_snapshot(db_path=None):
    """Import ha_snapshot.json (written via MCP from Bernacca live values)."""
    today = date.today().isoformat()
    with open(HA_SNAPSHOT_FILE) as f:
        snap = json.load(f)
    db.upsert_station(STATION_ID, STATION_NAME, LAT, LON, TZ, db_path)
    db.upsert_observation(STATION_ID, today, snap, source="ha", db_path=db_path)
    print(f"HA snapshot imported for {today}: {snap}")


def pull_ha_longterm_stats(db_path=None, days=90):
    """Optional: pull per-day aggregates from HA long-term statistics."""
    db.upsert_station(STATION_ID, STATION_NAME, LAT, LON, TZ, db_path)
    headers = _ha_headers()
    today = date.today()
    for entity_id, kind in HA_SENSORS.items():
        payload = {
            "start_time": f"{(today - timedelta(days=days)).isoformat()}T00:00:00+03:00",
            "end_time": f"{today.isoformat()}T23:59:59+03:00",
            "statistic_ids": [entity_id],
            "period": "day",
            "types": ["min", "max", "mean"],
        }
        try:
            r = requests.post(f"{HA_URL}/api/statistics/period", headers=headers,
                              json=payload, timeout=60)
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            print(f"  !! {entity_id}: {e}")
            continue
        if not data:
            print(f"  -- {entity_id}: no statistics")
            continue
        day_min, day_max, day_mean = {}, {}, {}
        for row in data:
            st = row["start"][:10]
            day_min.setdefault(st, {})[entity_id] = row.get("min")
            day_max.setdefault(st, {})[entity_id] = row.get("max")
            day_mean.setdefault(st, {})[entity_id] = row.get("mean")
        for d in sorted(day_mean.keys()):
            mn, mx, me = day_min[d], day_max[d], day_mean[d]
            if kind == "temperature":
                fields = {"temp_min": mn.get(entity_id), "temp_max": mx.get(entity_id),
                          "temp_mean": me.get(entity_id)}
            elif kind == "dewpoint":
                fields = {"dewpoint_min": mn.get(entity_id)}
            elif kind == "humidity":
                fields = {"humidity_min": mn.get(entity_id), "humidity_mean": me.get(entity_id)}
            elif kind == "wind":
                fields = {"wind_mean": me.get(entity_id), "wind_max": mx.get(entity_id)}
            elif kind == "pressure":
                fields = {"pressure_min": mn.get(entity_id)}
            else:
                fields = {"radiation_max": mx.get(entity_id)}
            db.upsert_observation(STATION_ID, d, fields, source="ha", db_path=db_path)
        print(f"  ok {entity_id}: {len(day_mean)} days")


def _om_cloud_fill(day_iso):
    """Cloud cover for `day_iso` from Open-Meteo (Bernacca has no cloud sensor)."""
    url = (
        "https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={LAT}&longitude={LON}"
        f"&start_date={day_iso}&end_date={day_iso}"
        "&daily=cloudcover_mean,cloudcover_min"
        f"&timezone={TZ}"
    )
    try:
        r = requests.get(url, timeout=30, headers=UA)
        r.raise_for_status()
        d = r.json()["daily"]
        return {"cloud_mean": d["cloudcover_mean"][0], "cloud_min": d["cloudcover_min"][0]}
    except Exception:
        return {}


def snapshot_evening_ha(db_path=None):
    """Evening snapshot from the Bernacca station via the HA REST API.
    Requires ha_token.txt. Cloud cover (not measured by the station) is
    filled from Open-Meteo. Run 2h after sunset."""
    import datetime as dt
    headers = _ha_headers()
    day = dt.date.today().isoformat()

    def ha_state(entity_id):
        r = requests.get(f"{HA_URL}/api/states/{entity_id}", headers=headers, timeout=30)
        r.raise_for_status()
        s = r.json()
        try:
            return float(s["state"])
        except (ValueError, TypeError):
            return None

    fields = {
        "temp_sunset": ha_state("sensor.bernacca_outdoor_temperature"),
        "dewpoint_sunset": ha_state("sensor.bernacca_dewpoint"),
        "humidity_min": ha_state("sensor.bernacca_humidity"),
        "wind_max": ha_state("sensor.bernacca_wind_speed"),
        "gust_max": ha_state("sensor.bernacca_wind_gust"),
        "soil_moisture": ha_state("sensor.bernacca_soil_moisture_1"),
        "precipitation": ha_state("sensor.bernacca_daily_rain_rate"),
        "pressure_min": ha_state("sensor.bernacca_relative_pressure"),
    }
    fields.update(_om_cloud_fill(day))
    db.upsert_station(STATION_ID, STATION_NAME, LAT, LON, TZ, db_path)
    db.upsert_observation(STATION_ID, day, fields, source="ha", db_path=db_path)
    print(f"HA evening snapshot saved for {day}: {fields}")


def snapshot_evening(db_path=None):
    """Evening snapshot 2h after sunset today (or yesterday, if not yet reached)
    from Open-Meteo hourly — fully autonomous, no HA token required."""
    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={LAT}&longitude={LON}"
        "&hourly=temperature_2m,dew_point_2m,relative_humidity_2m,wind_speed_10m,"
        "wind_gusts_10m,cloud_cover"
        "&daily=sunrise,sunset,temperature_2m_min,dew_point_2m_min"
        "&past_days=1&forecast_days=2"
        f"&timezone={TZ}"
    )
    r = requests.get(url, timeout=60, headers=UA)
    r.raise_for_status()
    j = r.json()
    hourly, daily = j["hourly"], j["daily"]

    target_day = date.today().isoformat()
    # if now < sunset+2h of today, use yesterday
    now_local = datetime.now().strftime("%Y-%m-%d %H:%M")
    today_sunset = daily["sunset"][-2]
    if now_local < daily["sunset"][1]:
        pass
    # pick the day whose sunset+2h has already passed
    idx_by_day = {}
    for i, ts in enumerate(hourly["time"]):
        idx_by_day.setdefault(ts[:10], []).append(i)
    chosen = None
    for i in range(len(daily["time"]) - 1, -1, -1):
        d = daily["time"][i]
        sunset_h = int(daily["sunset"][i][11:13])
        ev_ok = any(int(hourly["time"][k][11:13]) >= sunset_h + 2 for k in idx_by_day.get(d, []))
        if ev_ok and datetime.strptime(d, "%Y-%m-%d") <= datetime.now():
            chosen = (d, sunset_h)
            target_day = d
            break
    if chosen is None:
        print("No completed evening window yet.")
        return
    d, sunset_h = chosen
    ev_idx = [k for k in idx_by_day[d] if int(hourly["time"][k][11:13]) in (sunset_h + 2, sunset_h + 3)]
    fields = {
        "temp_sunset": sum(hourly["temperature_2m"][k] for k in ev_idx) / len(ev_idx),
        "dewpoint_sunset": sum(hourly["dew_point_2m"][k] for k in ev_idx) / len(ev_idx),
        "humidity_min": min(hourly["relative_humidity_2m"][k] for k in idx_by_day[d]),
        "wind_max": max(hourly["wind_speed_10m"][k] for k in idx_by_day[d]),
        "gust_max": max(hourly["wind_gusts_10m"][k] for k in idx_by_day[d]),
        "cloud_mean": sum(hourly["cloud_cover"][k] for k in idx_by_day[d]) / len(idx_by_day[d]),
        "cloud_min": min(hourly["cloud_cover"][k] for k in idx_by_day[d]),
    }
    db.upsert_station(STATION_ID, STATION_NAME, LAT, LON, TZ, db_path)
    db.upsert_observation(STATION_ID, d, fields, source="openmeteo", db_path=db_path)
    print(f"Evening snapshot for {d} (sunset {daily['sunset'][idx_by_day and 0 or 0] if False else sunset_h}h+2): {fields}")


def backfill_outcomes(db_path=None):
    """Backfill observed next-morning Tmin into outcomes and score predictions."""
    conn = db.get_conn(db_path)
    rows = conn.execute(
        """SELECT date, temp_min FROM observations
           WHERE station_id = ? AND temp_min IS NOT NULL""",
        (STATION_ID,),
    ).fetchall()
    for r in rows:
        db.backfill_outcome(STATION_ID, r["date"], r["temp_min"], db_path)
    print(f"Backfilled {len(rows)} outcomes.")
    print(f"Scoring: {db.model_stats(db_path)}")


if __name__ == "__main__":
    db.init_db()
    cmd = sys.argv[1] if len(sys.argv) > 1 else "help"
    if cmd == "backfill":
        backfill_climate()
    elif cmd == "pull-history":
        pull_history(days=60)
    elif cmd == "bootstrap":
        bootstrap(years=3)
    elif cmd == "snapshot":
        snapshot_evening()
    elif cmd == "snapshot-ha":
        snapshot_evening_ha()
    elif cmd == "pull-ha":
        pull_ha_longterm_stats()
    elif cmd == "import-ha":
        import_ha_snapshot()
    elif cmd == "outcomes":
        backfill_outcomes()
    else:
        print(__doc__)
