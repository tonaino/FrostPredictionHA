"""Frost predictor CLI.

Usage:
  python predict.py tonight          # predict frost probability for next morning
  python predict.py train            # (re)train model on collected data
  python predict.py report           # show recent predictions + model accuracy
"""
import sys
import json
import datetime as dt

import numpy as np

import db
import features as F
import model
import rolling_model


def _get_evening_row(station_id, date, db_path=None):
    conn = db.get_conn(db_path)
    return conn.execute(
        "SELECT * FROM observations WHERE station_id = ? AND date = ?",
        (station_id, date),
    ).fetchone()


def push_mqtt(prob, tmin_fao, level, target, tmin_rf=None):
    """Publish prediction via MQTT with HA discovery (independent of HA REST API).
     Broker: MQTT_BROKER env or mqtt:1883. Retained messages keep values
    across restarts; discovery makes the sensors appear automatically.
    Returns True if published."""
    import os
    import json
    import threading
    import paho.mqtt.client as mqtt

    host = os.environ.get("MQTT_BROKER", "mqtt")
    port = int(os.environ.get("MQTT_PORT", 1883))
    user = os.environ.get("MQTT_USER") or None
    pw = os.environ.get("MQTT_PASS") or None
    base = os.environ.get("MQTT_TOPIC", "frost_forecast")

    device = {
        "identifiers": ["frost_forecast"],
        "name": "Frost Forecast",
        "manufacturer": "frost-forecast",
        "model": "ML frost predictor",
    }
    entities = [
        ("probability", "Frost Probability", "%", {  # state_topic per entity below
            "state_topic": f"{base}/probability",
            "unique_id": "frost_forecast_probability",
            "unit_of_measurement": "%",
            "value_template": "{{ value_json.p }}",
        }),
        ("tmin", "Frost Tmin Estimate", "°C", {
            "state_topic": f"{base}/tmin",
            "unique_id": "frost_forecast_tmin",
            "unit_of_measurement": "°C",
            "value_template": "{{ value_json.tmin }}",
        }),
        ("tmin_rf", "Frost Tmin Estimate (Rolling RF)", "°C", {
            "state_topic": f"{base}/tmin_rf",
            "unique_id": "frost_forecast_tmin_rf",
            "object_id": "frost_tmin_rf",
            "unit_of_measurement": "°C",
            "value_template": "{{ value_json.tmin_rf }}",
        }),
        ("risk", "Frost Risk Level", None, {
            "state_topic": f"{base}/risk",
            "unique_id": "frost_forecast_risk",
        }),
        ("cold_warning", "Cold Warning (<3°C)", None, {
            "state_topic": f"{base}/cold_warning",
            "unique_id": "frost_forecast_cold_warning",
            "icon": "mdi:thermometer-alert",
        }),
    ]
    vals = {"probability": {"p": round(prob * 100, 1)},
            "tmin": {"tmin": round(float(tmin_fao), 1)},
            "tmin_rf": {"tmin_rf": round(float(tmin_rf), 1) if tmin_rf is not None else None},
            "risk": {"risk": level},
            "cold_warning": {"cold_warning": tmin_fao < 3.0}}

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="frost_forecast")
    if user:
        client.username_pw_set(user, pw)
    connected = threading.Event()
    connect_rc = {"rc": None}

    def _on_connect(cl, u, flags, reason_code, properties=None):
        # reason_code name is set for v3.1.1 as well in VERSION2 callback api
        connect_rc["rc"] = str(reason_code)
        if str(reason_code) in ("Success", "Connected"):
            connected.set()

    client.on_connect = _on_connect
    client.on_disconnect = lambda cl, u, flags, reason_code, properties=None: None
    ok = True
    try:
        client.connect(host, port, keepalive=30)
        client.loop_start()
        if not connected.wait(timeout=10):
            code = connect_rc["rc"]
            raise ConnectionError(
                f"broker {host}:{port} refused (reason={code}); set MQTT_USER/MQTT_PASS")
        for key, name, _unit, cfg in entities:
            cfg.update({"device": device, "name": name,
                        "json_attributes_topic": f"{base}/{key}_attrs"})
            client.publish(f"homeassistant/sensor/frost_forecast_{key}/config",
                           json.dumps(cfg), retain=True).wait_for_publish()
            # risk is published as plain text (no json template on its config)
            if key == "risk":
                payload = level
            elif key == "cold_warning":
                payload = "ON" if vals[key]["cold_warning"] else "OFF"
            elif key == "tmin_rf" and vals[key]["tmin_rf"] is None:
                payload = "unknown"
            else:
                payload = json.dumps(vals[key])
            client.publish(f"{base}/{key}", payload, retain=True).wait_for_publish()
            client.publish(f"{base}/{key}_attrs",
                           json.dumps({"forecast_morning": target.isoformat(),
                                       "updated": dt.datetime.now().isoformat(timespec="seconds")}),
                           retain=True).wait_for_publish()
        client.loop_stop()
    except Exception as e:
        ok = False
        print(f"MQTT push failed: {e}")
    return ok


def predict_tonight(db_path=None):
    station_id = collector.STATION_ID
    today = dt.date.today()

    # use today's evening snapshot if present, else yesterday's
    row = _get_evening_row(station_id, today.isoformat(), db_path)
    if row is None or row["temp_sunset"] is None:
        row = _get_evening_row(station_id, (today - dt.timedelta(days=1)).isoformat(), db_path)
        if row is None or row["temp_sunset"] is None:
            print("No evening snapshot yet. Run: python collector.py snapshot")
            return
        target = today
    else:
        target = today + dt.timedelta(days=1)

    clim_rows = db.get_conn(db_path).execute("SELECT * FROM climate_history").fetchall()
    climatology = F.day_of_year_frost_climatology(clim_rows, station_id)

    feats = F.build_features(row, climatology)
    pipe, meta = model.load_model()
    # Keep the paper-inspired regressor opt-in until its walk-forward error
    # beats the calibrated FAO/local-bias path on the local dataset.
    use_regressor = __import__("os").environ.get("FROST_USE_TMIN_REGRESSOR", "0").lower() in (
        "1", "true", "yes"
    )
    regressor, reg_meta = model.load_tmin_regressor() if use_regressor else (None, None)
    prob, tmin_ml, tmin_fao, detail = model.predict_frost_probability(
        feats, pipe, meta, db_path, regressor=regressor)
    rolling_bundle, rolling_meta = rolling_model.load()
    rolling_tmin = rolling_model.predict_for_observation(row, rolling_bundle, db_path)

    version_id = meta.get("version_id") if meta else None
    if version_id is None:
        conn = db.get_conn(db_path)
        active = conn.execute(
            "SELECT version_id FROM model_versions WHERE is_active = 1"
        ).fetchone()
        version_id = active["version_id"] if active else None

    pid = db.save_prediction(
        station_id=station_id,
        target_date=target.isoformat(),
        frost_probability=round(prob, 4),
        tmin_predicted=tmin_ml,
        tmin_empirical=round(tmin_fao, 2) if tmin_fao is not None else None,
        features=json.dumps(feats),
        model_version_id=version_id,
        tmin_rolling_rf=rolling_tmin,
        db_path=db_path,
    )

    level = ("HIGH" if prob >= 0.6 else
             "MODERATE" if prob >= 0.3 else
             "LOW" if prob >= 0.1 else "VERY LOW")
    pushed = push_mqtt(prob, tmin_fao, level, target, rolling_tmin)
    print(f"Frost forecast for morning of {target}")
    print(f"  probability : {prob*100:.0f}%  [{level}]" + ("  -> pushed to HA" if pushed else ""))
    print(f"  Tmin (FAO)  : {tmin_fao:.1f} C")
    print(f"  model detail: {detail}")
    if rolling_tmin is not None:
        print(f"  RF rolling Tmin (shadow): {rolling_tmin:.1f} C")
    print(f"  prediction id: {pid}")


def train(db_path=None):
    meta, version_id = model.train(db_path)
    reg_meta = model.train_tmin_regressor(db_path)
    rolling_meta = rolling_model.train(db_path)
    print("Model trained.")
    print(f"  samples: {meta['n_samples']}  frost events: {meta['n_frost']}")
    print(f"  CV: {meta['cv']}")
    print(f"  FAO coefficients (a,b,c): {meta['fao_coef']}")
    print(f"  version id: {version_id}")
    print(f"  local Tmin regressor: {reg_meta['n_samples']} samples")
    print(f"  regressor CV: {reg_meta['cv_mean']}")
    print(f"  rolling RF: {rolling_meta['n_samples']} samples, CV: {rolling_meta['cv_mean']}")


def report(db_path=None):
    stats = db.model_stats(db_path)
    print(f"Predictions: {stats}")
    for p in db.recent_predictions(collector.STATION_ID, 10, db_path):
        obs = ("-> frost" if p["observed_frost"] else
               "-> no frost") if p["observed_frost"] is not None else ""
        print(f"  {p['target_date']}  P={p['frost_probability']*100:5.1f}%  "
              f"Tmin_fao={p['tmin_empirical']}  {obs}  hit={p['hit']}")


if __name__ == "__main__":
    import collector
    db.init_db()
    cmd = sys.argv[1] if len(sys.argv) > 1 else "tonight"
    if cmd == "tonight":
        predict_tonight()
    elif cmd == "train":
        train()
    elif cmd == "report":
        report()
    else:
        print(__doc__)
