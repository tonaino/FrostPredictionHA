"""In-container scheduler for frost-forecast.

Runs the daily pipeline at 21:45 local time (container TZ) and the weekly
refresh+retrain on Sundays. Replaces cron/launchd when deployed in Docker —
a single long-running process, logs to stdout, honors the same logic as
daily.sh. On startup it backfills any missed run for today.

Usage:  python scheduler.py            # run scheduler loop (container entrypoint)
        python scheduler.py --now      # run pipeline once and exit (debug)
"""
import subprocess
import sys
import time
import os
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import requests

TZ = ZoneInfo(os.environ.get("FROST_TZ", "UTC"))
LAT = float(os.environ.get("FROST_LAT", "0"))
LON = float(os.environ.get("FROST_LON", "0"))
FALLBACK_HOUR, FALLBACK_MINUTE = 21, 45


def _run(label, cmd):
    print(f"[scheduler] {label}: {' '.join(cmd)}", flush=True)
    r = subprocess.run([sys.executable, *cmd], capture_output=True, text=True)
    out = (r.stdout or "").strip()
    if out:
        print(out, flush=True)
    if r.returncode != 0:
        print(f"[scheduler] !! {label} failed rc={r.returncode}\n{r.stderr}", flush=True)
    return r.returncode == 0


def has_token():
    import collector
    return collector.has_token()


def run_pipeline():
    """Same steps as daily.sh."""
    import db
    db.init_db()
    _run("outcomes", ["-m", "collector", "outcomes"])
    if has_token():
        # HA station mode: evening snapshot + mid-week stats refresh
        _run("snapshot-ha", ["-m", "collector", "snapshot-ha"])
        if datetime.now(TZ).weekday() == 2:  # Wednesday
            _run("pull-ha", ["-m", "collector", "pull-ha"])
    else:
        _run("snapshot", ["-m", "collector", "snapshot"])
    if datetime.now(TZ).weekday() == 6:  # Sunday
        _run("pull-history", ["-m", "collector", "pull-history"])
        _run("train", ["-m", "predict", "train"])
    _run("tonight", ["-m", "predict", "tonight"])


def _missing_todays_prediction():
    """True if we haven't predicted for tomorrow yet today (catch-up check)."""
    import db
    import os
    path = os.environ.get("FROST_DB_PATH") or db.DB_PATH
    conn = db.get_conn(path)
    row = conn.execute(
        """SELECT COUNT(*) AS n FROM predictions
           WHERE target_date = date('now', 'localtime', '+1 day')"""
    ).fetchone()
    return row["n"] == 0


def next_snapshot_time(now=None):
    """Return the next local sunset+2h run time.

    The station snapshot must be taken after the evening radiative window
    begins, not at a fixed clock time. Keep the old time as a safe fallback
    if the astronomy API is unavailable.
    """
    now = now or datetime.now(TZ)
    try:
        r = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": LAT, "longitude": LON,
                "daily": "sunset", "timezone": os.environ.get("FROST_TZ", "UTC"),
                "forecast_days": 2,
            }, timeout=15,
        )
        r.raise_for_status()
        sunsets = r.json()["daily"]["sunset"]
        for value in sunsets:
            target = datetime.fromisoformat(value).replace(tzinfo=TZ) + timedelta(hours=2)
            if target > now:
                return target
    except Exception as e:
        print(f"[scheduler] sunset lookup failed, using fallback: {e}", flush=True)
    target = now.replace(hour=FALLBACK_HOUR, minute=FALLBACK_MINUTE,
                         second=0, microsecond=0)
    if target <= now:
        target += timedelta(days=1)
    return target


def today_snapshot_time(now=None):
    """Today's sunset+2h time, used for startup catch-up."""
    now = now or datetime.now(TZ)
    try:
        r = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": LAT, "longitude": LON,
                "daily": "sunset", "timezone": os.environ.get("FROST_TZ", "UTC"),
                "forecast_days": 1,
            }, timeout=15,
        )
        r.raise_for_status()
        value = r.json()["daily"]["sunset"][0]
        return datetime.fromisoformat(value).replace(tzinfo=TZ) + timedelta(hours=2)
    except Exception as e:
        print(f"[scheduler] today's sunset lookup failed, using fallback: {e}", flush=True)
        return now.replace(hour=FALLBACK_HOUR, minute=FALLBACK_MINUTE,
                           second=0, microsecond=0)


def seconds_until_next_run():
    return max(1.0, (next_snapshot_time() - datetime.now(TZ)).total_seconds())


def loop():
    while True:
        wait = seconds_until_next_run()
        target = next_snapshot_time()
        print(f"[scheduler] next Bernacca snapshot at {target.isoformat()} "
              f"(in {wait/3600:.1f}h)", flush=True)
        time.sleep(max(wait, 1))
        try:
            run_pipeline()
        except Exception as e:  # keep the scheduler alive
            print(f"[scheduler] pipeline error: {e}", flush=True)


if __name__ == "__main__":
    if "--now" in sys.argv:
        run_pipeline()
        sys.exit(0)
    # catch-up: if started after 21:45 and today's prediction is missing, run now
    try:
        if datetime.now(TZ) >= today_snapshot_time() and _missing_todays_prediction():
            print("[scheduler] catch-up run on startup", flush=True)
            run_pipeline()
    except Exception as e:
        print(f"[scheduler] startup catch-up skipped: {e}", flush=True)
    loop()
