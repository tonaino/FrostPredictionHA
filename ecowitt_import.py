"""Import historical Ecowitt Cloud data into Frost Detector.

Credentials are read from environment variables:
  ECOWITT_API_KEY
  ECOWITT_APPLICATION_KEY
  ECOWITT_MAC

Example:
  ECOWITT_API_KEY=... ECOWITT_APPLICATION_KEY=... ECOWITT_MAC=... \
    python ecowitt_import.py --start 2023-01-01 --end 2026-09-24

The importer requests five-minute outdoor data, derives daily aggregates, and
uses the Ecowitt reading nearest sunset+2h for the evening features. Existing
HA/Bernacca fields win through db.upsert_observation's source precedence.
"""
import argparse
import os
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import requests

import db

LAT = float(os.environ.get("FROST_LAT", "0"))
LON = float(os.environ.get("FROST_LON", "0"))
TZ_NAME = os.environ.get("FROST_TZ", "UTC")
TZ = ZoneInfo(TZ_NAME)
STATION_ID = os.environ.get("FROST_STATION_ID", "local_station")
API_URL = "https://api.ecowitt.net/api/v3/device/history"


def _credentials():
    names = ("ECOWITT_API_KEY", "ECOWITT_APPLICATION_KEY", "ECOWITT_MAC")
    values = [os.environ.get(name, "").strip() for name in names]
    missing = [name for name, value in zip(names, values) if not value]
    if missing:
        raise RuntimeError("Missing Ecowitt credentials: " + ", ".join(missing))
    return dict(zip(names, values))


def _sunsets(start, end):
    today = date.today()
    result = {}

    # Archive handles completed historical days. The forecast endpoint handles
    # the current/future boundary; archive rejects ranges that cross it.
    archive_end = min(end, today - timedelta(days=5))
    if start < archive_end:
        r = requests.get(
            "https://archive-api.open-meteo.com/v1/archive",
            params={"latitude": LAT, "longitude": LON, "daily": "sunset",
                    "start_date": start.isoformat(),
                    "end_date": (archive_end - timedelta(days=1)).isoformat(),
                    "timezone": TZ_NAME}, timeout=60,
        )
        r.raise_for_status()
        result.update(dict(zip(r.json()["daily"]["time"], r.json()["daily"]["sunset"])))

    forecast_start = max(start, today - timedelta(days=4))
    if forecast_start < end:
        r = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={"latitude": LAT, "longitude": LON, "daily": "sunset",
                    "past_days": max(0, (today - forecast_start).days),
                    "forecast_days": max(1, (end - today).days + 1),
                    "timezone": TZ_NAME}, timeout=60,
        )
        r.raise_for_status()
        result.update(dict(zip(r.json()["daily"]["time"], r.json()["daily"]["sunset"])))

    return {
        day: datetime.fromisoformat(value).replace(tzinfo=TZ) + timedelta(hours=2)
        for day, value in result.items()
    }


def _series(payload, *path):
    value = payload
    for key in path:
        value = value.get(key, {}) if isinstance(value, dict) else {}
    return value.get("list", {}) if isinstance(value, dict) else {}


def _timestamp(value):
    try:
        return datetime.fromtimestamp(float(value), timezone.utc).astimezone(TZ)
    except (TypeError, ValueError, OSError):
        return datetime.fromisoformat(str(value)).replace(tzinfo=TZ)


def _fetch(start, end, credentials):
    params = {
        "application_key": credentials["ECOWITT_APPLICATION_KEY"],
        "api_key": credentials["ECOWITT_API_KEY"],
        "mac": credentials["ECOWITT_MAC"],
        "start_date": start.strftime("%Y-%m-%d %H:%M:%S"),
        "end_date": end.strftime("%Y-%m-%d %H:%M:%S"),
        "call_back": "outdoor,wind,pressure,solar_and_uvi",
        "cycle_type": "5min",
        "temp_unitid": "1",  # Celsius
        "wind_speed_unitid": "6",  # m/s
        "pressure_unitid": "3",  # hPa
    }
    r = requests.get(API_URL, params=params, timeout=90)
    r.raise_for_status()
    payload = r.json()
    if str(payload.get("code", "0")) not in ("0", "200") and payload.get("msg") != "success":
        raise RuntimeError(f"Ecowitt API error: {payload}")
    return payload.get("data", payload)


def _import_values(values, sunsets, db_path=None):
    imported = 0
    for day, series in sorted(values.items()):
        temps = series.get("temperature", [])
        dews = series.get("dewpoint", [])
        if not temps:
            continue
        target = sunsets.get(day)
        evening = min(temps, key=lambda item: abs(item[0] - target)) if target else None
        fields = {
            "temp_min": min(value for _, value in temps),
            "temp_max": max(value for _, value in temps),
            "temp_mean": sum(value for _, value in temps) / len(temps),
        }
        if dews:
            fields["dewpoint_min"] = min(value for _, value in dews)
        if evening:
            fields["temp_sunset"] = evening[1]
            matching_dew = min(dews, key=lambda item: abs(item[0] - evening[0])) if dews else None
            if matching_dew:
                fields["dewpoint_sunset"] = matching_dew[1]
        db.upsert_observation(STATION_ID, day, fields, source="ecowitt", db_path=db_path)
        imported += 1
    return imported


def import_range(start, end, db_path=None, chunk_days=7):
    credentials = _credentials()
    db.init_db(db_path)
    sunsets = _sunsets(start, end)
    cursor = start
    while cursor < end:
        chunk_end = min(cursor + timedelta(days=chunk_days), end)
        payload = _fetch(cursor, chunk_end, credentials)
        values = {}
        series_map = [
            (("outdoor", "temperature"), "temperature", "temp"),
            (("outdoor", "dew_point"), "dewpoint", "dewpoint"),
            (("outdoor", "humidity"), "humidity", "humidity"),
            (("wind", "wind_speed"), "wind", "wind"),
            (("wind", "wind_gust"), "gust", "gust"),
            (("pressure", "relative"), "pressure", "pressure"),
            (("solar_and_uvi", "solar"), "radiation", "radiation"),
        ]
        samples = {}
        for path, field, sample_field in series_map:
            for raw_ts, raw_value in _series(payload, *path).items():
                try:
                    timestamp = _timestamp(raw_ts)
                    value = float(raw_value)
                except (TypeError, ValueError, OSError):
                    continue
                day = timestamp.date().isoformat()
                values.setdefault(day, {}).setdefault(field, []).append((timestamp, value))
                samples.setdefault(timestamp.isoformat(), {
                    "timestamp": timestamp.isoformat(), "source": "ecowitt"
                })[sample_field] = value
        db.upsert_ecowitt_samples(list(samples.values()), db_path=db_path)
        print(f"Fetched Ecowitt {cursor} .. {chunk_end}", flush=True)
        print(f"Imported {_import_values(values, sunsets, db_path)} Ecowitt days.", flush=True)
        cursor = chunk_end


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", required=True, type=date.fromisoformat)
    parser.add_argument("--end", required=True, type=date.fromisoformat,
                        help="Inclusive end date")
    parser.add_argument("--db", default=None, help="Override Frost DB path")
    parser.add_argument("--chunk-days", type=int, default=7,
                        help="Ecowitt request size (default: 7 days)")
    args = parser.parse_args()
    import_range(args.start, args.end + timedelta(days=1), args.db, args.chunk_days)


if __name__ == "__main__":
    main()
