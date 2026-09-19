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
from datetime import datetime, timedelta

TZ_OFFSET_OK = True
RUN_HOUR, RUN_MINUTE = 21, 45


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
        if datetime.now().weekday() == 2:  # Wednesday
            _run("pull-ha", ["-m", "collector", "pull-ha"])
    else:
        _run("snapshot", ["-m", "collector", "snapshot"])
    if datetime.now().weekday() == 6:  # Sunday
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


def seconds_until_next_run():
    now = datetime.now()
    target = now.replace(hour=RUN_HOUR, minute=RUN_MINUTE, second=0, microsecond=0)
    if target <= now:
        target += timedelta(days=1)
    return (target - now).total_seconds()


def loop():
    while True:
        wait = seconds_until_next_run()
        print(f"[scheduler] next run in {wait/3600:.1f}h", flush=True)
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
        if datetime.now().hour >= RUN_HOUR and _missing_todays_prediction():
            print("[scheduler] catch-up run on startup", flush=True)
            run_pipeline()
    except Exception as e:
        print(f"[scheduler] startup catch-up skipped: {e}", flush=True)
    loop()
