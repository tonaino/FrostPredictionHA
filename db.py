"""SQLite database for frost-forecast: stations, daily aggregates, historical climate, predictions."""
import os
import sqlite3
import threading
from contextlib import contextmanager

DB_PATH = os.environ.get("FROST_DB_PATH") or os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "frost.db"
)

SOURCE_PRIORITY = {"openmeteo": 1, "ecowitt": 2, "ha": 3, "manual": 4}
SOURCE_COLUMNS = {
    "temp_min": "temp_min_source",
    "temp_max": "temp_max_source",
    "temp_mean": "temp_mean_source",
    "temp_sunset": "temp_sunset_source",
    "dewpoint_sunset": "dewpoint_sunset_source",
}

_local = threading.local()


def get_conn(db_path: str = None) -> sqlite3.Connection:
    conn = getattr(_local, "conn", None)
    path = db_path or DB_PATH
    if conn is None or getattr(_local, "path", None) != path:
        conn = sqlite3.connect(path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        _local.conn = conn
        _local.path = path
    return conn


@contextmanager
def tx(db_path: str = None):
    conn = get_conn(db_path)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise


SCHEMA = """
CREATE TABLE IF NOT EXISTS stations (
    station_id   TEXT PRIMARY KEY,
    name         TEXT NOT NULL,
    latitude     REAL,
    longitude    REAL,
    timezone     TEXT DEFAULT 'UTC',
    is_active    INTEGER DEFAULT 1,
    created_at   TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS observations (
    station_id   TEXT NOT NULL REFERENCES stations(station_id),
    date         TEXT NOT NULL,              -- local date YYYY-MM-DD
    temp_min     REAL,                       -- daily min air temp C
    temp_max     REAL,
    temp_mean    REAL,
    temp_sunset  REAL,                       -- temp ~2h after sunset C
    dewpoint_sunset REAL,
    dewpoint_min REAL,
    humidity_min REAL,
    humidity_mean REAL,
    wind_mean    REAL,
    wind_max     REAL,
    gust_max     REAL,
    pressure_min REAL,                       -- hPa (sea-level adjusted)
    pressure_change REAL,                    -- pressure drop from previous day
    cloud_mean   REAL,                       -- % (0-100)
    cloud_min    REAL,
    radiation_max REAL,                      -- W/m2
    radiation_total REAL,                    -- MJ/m2/day
    precipitation REAL,                      -- mm (total of previous 24-48h)
    soil_temp    REAL,
    soil_moisture REAL,
    night_hours_subzero REAL,                -- hours with T < 0 during night
    source       TEXT,                       -- 'ha' | 'openmeteo' | 'manual'
    updated_at   TEXT DEFAULT (datetime('now')),
    PRIMARY KEY (station_id, date)
);

CREATE TABLE IF NOT EXISTS climate_history (
    station_id   TEXT NOT NULL REFERENCES stations(station_id),
    date         TEXT NOT NULL,
    temp_min     REAL,
    temp_max     REAL,
    dewpoint_min REAL,
    precipitation REAL,
    wind_max     REAL,
    cloud_mean   REAL,
    PRIMARY KEY (station_id, date)
);

CREATE TABLE IF NOT EXISTS model_versions (
    version_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    trained_at   TEXT NOT NULL,
    algorithm    TEXT NOT NULL,
    n_samples    INTEGER,
    n_frost      INTEGER,
    metrics      TEXT,                       -- JSON
    model_path   TEXT,
    is_active    INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS predictions (
    prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
    station_id   TEXT NOT NULL REFERENCES stations(station_id),
    target_date  TEXT NOT NULL,              -- date of the predicted morning
    run_at       TEXT NOT NULL DEFAULT (datetime('now')),
    model_version_id INTEGER REFERENCES model_versions(version_id),
    frost_probability REAL NOT NULL,         -- 0..1
    tmin_predicted    REAL,                  -- predicted min temp
    tmin_empirical    REAL,                  -- FAO-based estimate
    tmin_rolling_rf   REAL,                  -- shadow local rolling RF estimate
    features_json     TEXT,                  -- JSON of input features
    observed_frost    INTEGER,               -- 1/0 backfilled after target date
    observed_tmin     REAL,
    hit               INTEGER                -- 1 if prediction correct at threshold
);
CREATE INDEX IF NOT EXISTS idx_predictions_target ON predictions(station_id, target_date);

CREATE TABLE IF NOT EXISTS outcomes (
    station_id   TEXT NOT NULL REFERENCES stations(station_id),
    date         TEXT NOT NULL,
    tmin_observed REAL,
    frost        INTEGER,                    -- 1 if frost observed
    backfilled_at TEXT DEFAULT (datetime('now')),
    PRIMARY KEY (station_id, date)
);

CREATE TABLE IF NOT EXISTS ecowitt_samples (
    timestamp TEXT PRIMARY KEY,
    temp REAL,
    dewpoint REAL,
    humidity REAL,
    wind REAL,
    gust REAL,
    pressure REAL,
    radiation REAL,
    source TEXT NOT NULL,
    updated_at TEXT DEFAULT (datetime('now'))
);
"""


def init_db(db_path: str = None):
    with tx(db_path) as conn:
        conn.executescript(SCHEMA)
        try:
            conn.execute("ALTER TABLE predictions ADD COLUMN tmin_rolling_rf REAL")
        except sqlite3.OperationalError:
            pass
        # Field-level provenance was added after the original single `source`
        # column.  Preserve existing data while making future imports obey
        # HA/Bernacca > Ecowitt > Open-Meteo precedence.
        for col in SOURCE_COLUMNS.values():
            try:
                conn.execute(f"ALTER TABLE observations ADD COLUMN {col} TEXT")
            except sqlite3.OperationalError:
                pass
        for field, source_col in SOURCE_COLUMNS.items():
            conn.execute(
                f"UPDATE observations SET {source_col}=source "
                f"WHERE {field} IS NOT NULL AND {source_col} IS NULL"
            )


def upsert_station(station_id, name, lat, lon, timezone="UTC", db_path=None):
    with tx(db_path) as conn:
        conn.execute(
            """INSERT INTO stations (station_id, name, latitude, longitude, timezone, is_active)
               VALUES (?, ?, ?, ?, ?, 1)
               ON CONFLICT(station_id) DO UPDATE SET
                 name=excluded.name, latitude=excluded.latitude,
                 longitude=excluded.longitude, timezone=excluded.timezone""",
            (station_id, name, lat, lon, timezone),
        )


def upsert_observation(station_id: str, date: str, fields: dict, source: str, db_path=None):
    """Upsert one daily aggregate row. Only provided fields are written (None-aware)."""
    allowed = [
        "temp_min", "temp_max", "temp_mean", "temp_sunset", "dewpoint_sunset",
        "dewpoint_min", "humidity_min", "humidity_mean", "wind_mean", "wind_max",
        "gust_max", "pressure_min", "pressure_change", "cloud_mean", "cloud_min",
        "radiation_max", "radiation_total", "precipitation", "soil_temp",
        "soil_moisture", "night_hours_subzero",
    ]
    cols = ["station_id", "date"]
    vals = [station_id, date]
    sets = ["updated_at=datetime('now')"]
    conn = get_conn(db_path)
    existing = conn.execute(
        "SELECT * FROM observations WHERE station_id = ? AND date = ?",
        (station_id, date),
    ).fetchone()
    incoming_rank = SOURCE_PRIORITY.get(source, 0)
    accepted = []
    for k in allowed:
        if k in fields and fields[k] is not None:
            source_col = SOURCE_COLUMNS.get(k)
            if existing is not None and source_col:
                old_source = existing[source_col]
                if old_source and SOURCE_PRIORITY.get(old_source, 0) > incoming_rank:
                    continue
            cols.append(k)
            vals.append(fields[k])
            sets.append(f"{k}=excluded.{k}")
            if source_col:
                cols.append(source_col)
                vals.append(source)
                sets.append(f"{source_col}=excluded.{source_col}")
            accepted.append(k)
    cols.append("source")
    vals.append(source)
    # Keep the legacy row-level source useful as the highest source accepted
    # by this write, without allowing a lower-priority write to relabel data.
    old_row_source = existing["source"] if existing is not None else None
    row_source = source if accepted or existing is None else old_row_source
    vals[-1] = row_source
    sets.append("source=excluded.source")
    sql = (f"INSERT INTO observations ({', '.join(cols)}) VALUES "
           f"({', '.join('?' * len(vals))}) ON CONFLICT(station_id, date) DO UPDATE SET "
           f"{', '.join(sets)}")
    with tx(db_path) as conn:
        conn.execute(sql, vals)


def upsert_ecowitt_samples(samples: list, db_path=None):
    """Store high-frequency Ecowitt samples, deduplicated by timestamp."""
    conn = get_conn(db_path)
    for sample in samples:
        timestamp = sample.get("timestamp")
        if not timestamp:
            continue
        existing = conn.execute(
            "SELECT source FROM ecowitt_samples WHERE timestamp = ?", (timestamp,)
        ).fetchone()
        incoming = SOURCE_PRIORITY.get(sample.get("source", "ecowitt"), 0)
        if existing and SOURCE_PRIORITY.get(existing["source"], 0) > incoming:
            continue
        conn.execute(
            """INSERT INTO ecowitt_samples
               (timestamp,temp,dewpoint,humidity,wind,gust,pressure,radiation,source)
               VALUES (?,?,?,?,?,?,?,?,?)
               ON CONFLICT(timestamp) DO UPDATE SET
                 temp=excluded.temp, dewpoint=excluded.dewpoint,
                 humidity=excluded.humidity, wind=excluded.wind,
                 gust=excluded.gust, pressure=excluded.pressure,
                 radiation=excluded.radiation, source=excluded.source,
                 updated_at=datetime('now')""",
            (timestamp, sample.get("temp"), sample.get("dewpoint"),
             sample.get("humidity"), sample.get("wind"), sample.get("gust"),
             sample.get("pressure"), sample.get("radiation"),
             sample.get("source", "ecowitt")),
        )
    conn.commit()


def upsert_climate_rows(station_id: str, rows: list, db_path=None):
    with tx(db_path) as conn:
        conn.executemany(
            """INSERT INTO climate_history (station_id, date, temp_min, temp_max, dewpoint_min,
                                            precipitation, wind_max, cloud_mean)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(station_id, date) DO UPDATE SET
                 temp_min=excluded.temp_min, temp_max=excluded.temp_max,
                 dewpoint_min=excluded.dewpoint_min, precipitation=excluded.precipitation,
                 wind_max=excluded.wind_max, cloud_mean=excluded.cloud_mean""",
            [
                (station_id, r["date"], r.get("temp_min"), r.get("temp_max"),
                 r.get("dewpoint_min"), r.get("precipitation"), r.get("wind_max"),
                 r.get("cloud_mean"))
                for r in rows
            ],
        )


def save_prediction(station_id, target_date, frost_probability, tmin_predicted,
                    tmin_empirical, features, model_version_id=None,
                    tmin_rolling_rf=None, db_path=None):
    with tx(db_path) as conn:
        cur = conn.execute(
            """INSERT INTO predictions (station_id, target_date, frost_probability,
                                        tmin_predicted, tmin_empirical, tmin_rolling_rf,
                                        features_json, model_version_id)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (station_id, target_date, frost_probability, tmin_predicted, tmin_empirical,
             tmin_rolling_rf, features, model_version_id),
        )
        return cur.lastrowid


def save_model_version(trained_at, algorithm, n_samples, n_frost, metrics, model_path, db_path=None):
    with tx(db_path) as conn:
        conn.execute("UPDATE model_versions SET is_active=0")
        cur = conn.execute(
            """INSERT INTO model_versions (trained_at, algorithm, n_samples, n_frost,
                                           metrics, model_path, is_active)
               VALUES (?, ?, ?, ?, ?, ?, 1)""",
            (trained_at, algorithm, n_samples, n_frost, metrics, model_path),
        )
        return cur.lastrowid


def get_training_data(db_path=None):
    """Rows joined with next-day observed outcome (frost if temp_min <= 0)."""
    conn = get_conn(db_path)
    return conn.execute(
        """SELECT o.date, o.*, l.temp_min AS next_temp_min
           FROM observations o
           JOIN observations l
             ON l.station_id = o.station_id
            AND l.date = date(o.date, '+1 day')
           WHERE o.temp_sunset IS NOT NULL
           ORDER BY o.date"""
    ).fetchall()


def get_pending_predictions(db_path=None):
    """Predictions whose target date has passed and still lack observed outcomes."""
    conn = get_conn(db_path)
    return conn.execute(
        """SELECT p.prediction_id, p.station_id, p.target_date
           FROM predictions p
           LEFT JOIN outcomes o ON o.station_id = p.station_id AND o.date = p.target_date
           WHERE o.date IS NULL AND p.target_date < date('now', 'localtime')"""
    ).fetchall()


def backfill_outcome(station_id, date, tmin, db_path=None):
    frost = 1 if (tmin is not None and tmin <= 0.0) else 0
    with tx(db_path) as conn:
        conn.execute(
            """INSERT INTO outcomes (station_id, date, tmin_observed, frost)
               VALUES (?, ?, ?, ?)
               ON CONFLICT(station_id, date) DO UPDATE SET
                 tmin_observed=excluded.tmin_observed, frost=excluded.frost""",
            (station_id, date, tmin, frost),
        )
        conn.execute(
            """UPDATE predictions
               SET observed_tmin = ?, observed_frost = ?,
                   hit = CASE WHEN (frost_probability >= 0.5) = (? = 1) THEN 1 ELSE 0 END
               WHERE station_id = ? AND target_date = ?""",
            (tmin, frost, frost, station_id, date),
        )


def model_stats(db_path=None):
    conn = get_conn(db_path)
    total = conn.execute("SELECT COUNT(*) AS n FROM predictions").fetchone()["n"]
    scored = conn.execute(
        "SELECT COUNT(*) AS n FROM predictions WHERE hit IS NOT NULL"
    ).fetchone()["n"]
    hits = conn.execute(
        "SELECT COUNT(*) AS n FROM predictions WHERE hit = 1"
    ).fetchone()["n"]
    return {"predictions": total, "scored": scored, "hits": hits}


def recent_predictions(station_id, limit=10, db_path=None):
    conn = get_conn(db_path)
    return conn.execute(
        """SELECT * FROM predictions WHERE station_id = ?
           ORDER BY run_at DESC LIMIT ?""",
        (station_id, limit),
    ).fetchall()


if __name__ == "__main__":
    init_db()
    print(f"Database initialized at {DB_PATH}")
