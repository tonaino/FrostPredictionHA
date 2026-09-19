# Frost Forecast — Gorna Malina

Morning-frost probability model for the garden/orchard, built from the Home
Assistant weather station ("Bernacca" / MeteoCasa) plus Open-Meteo data.

## What it does

Every evening it evaluates **"what is the % chance of frost tomorrow morning?"**
and stores every prediction in a SQLite database, scoring itself against what
actually happened so accuracy improves over time.

Hybrid model (following FAO *Frost Protection* (Snyder & de Melo-Abreu 2005),
Talsma et al. 2023 and Eccel et al. 2007):

1. **FAO empirical physics layer** — `Tmin = a·T_sunset + b·Td_sunset + c`
   measured 2h after sunset (fitted locally on radiative nights), corrected by
   a wind/cloud "radiative night factor".
2. **Machine-learning layer** — stacked Random Forest + Gradient Boosting on
   19 engineered features (evening T/Td + dewpoint spread, humidity, wind,
   gusts, pressure change, cloud cover, precipitation, soil moisture, day-of-year
   cyclic encoding, 10-year day-of-year frost climatology).
3. **Blended probability** — ML trust grows with the number of real frost
   events collected; physics + climatology dominate early on.

## Current state

- Database: `frost.db` — 10 years (3652 days) climate history, 3 years of
  bootstrap observations + recent daily aggregates, prediction log with
  observed-outcome backfill and hit/miss scoring.
- Trained model: 1095 samples, 162 frost events, **CV AUC 0.951**, Brier 0.079.
- Daily automation: `launchctl` job `com.frostforecast.daily` runs `daily.sh`
  at **21:45** (after sunset + 2h, the FAO measurement time).
- HA automation: `automation.frost_alert_evening_forecast_check` — 20:00
  forecast check, phone + persistent notification when forecast overnight low
  <= 2 °C.

## HA entity export (MQTT)

Every night after the prediction, the model publishes to the MQTT broker with
**HA discovery** — entities `sensor.frost_probability`, `sensor.frost_tmin_estimate`
and `sensor.frost_risk_level` appear automatically under the "Frost Forecast"
device, with retained state (values survive HA restarts without re-publish).
No HA REST token is needed for the export; only MQTT broker credentials:

| Env | Default | Meaning |
|---|---|---|
| `MQTT_BROKER` | 192.168.31.200 | broker host |
| `MQTT_PORT` | 1883 | broker port |
| `MQTT_USER`/`MQTT_PASS` | — | broker credentials (broker rejects anonymous) |
| `MQTT_TOPIC` | frost_forecast | topic base |

Dashboard tile: `sensor.frost_probability`; automations can key on
`sensor.frost_risk_level` (HIGH triggers `frost_alert_ml_model_high`).

Two frost automations exist (independent layers):
1. `frost_alert_evening_forecast_check` — 20:00 forecast-based (weather entity)
2. `frost_alert_ml_model_high` — ML-model-driven, fires on HIGH push

## Docker deployment (server)

The container runs its own scheduler (`scheduler.py`) — 21:45 daily pipeline,
Wed 03:45 mid-week HA stats refresh, Sunday refresh+retrain, plus a catch-up
run if started late. Data lives in `./data/` (SQLite + models), so the
container can be rebuilt or upgraded without losing history.

```bash
# on the server (Docker + compose required)
scp -r frost-forecast/ server:~/            # or git clone
cd ~/frost-forecast

# optional: enable Bernacca station collection in-container
echo 'HA_TOKEN=eyJ...your-long-lived-token...' > .env

docker compose up -d --build
docker logs -f frost-forecast               # scheduler + pipeline output
```

One-off commands inside the container:

```bash
docker compose exec frost-forecast python -m predict tonight   # frost % now
docker compose exec frost-forecast python -m predict report    # accuracy log
docker compose exec frost-forecast python -m collector snapshot-ha
```

Portability notes: `HA_URL` (default `http://192.168.31.200:8123`) is set in
`docker-compose.yml`; no ports are published (the container is an HTTP
*client* of HA only); run `docker compose restart` after switching tokens.

When moving off the Mac, unload the local scheduler:
`launchctl bootout gui/$(id -u)/com.frostforecast.daily`

## Commands

```
cd ~/frost-forecast
source .venv/bin/activate          # or use ./.venv/bin/python

./.venv/bin/python collector.py snapshot     # evening snapshot (sunset+2h)
./.venv/bin/python collector.py outcomes     # backfill observed Tmin + scoring
./.venv/bin/python predict.py tonight        # frost % for next morning
./.venv/bin/python predict.py report         # recent predictions + accuracy
./.venv/bin/python predict.py train          # retrain (auto every Sunday)
./.venv/bin/python collector.py pull-history # refresh daily aggregates
./.venv/bin/python collector.py backfill     # 10y climate history (already done)
```

Manual one-off test of the full cycle: `./daily.sh`

## Data sources

| Source | What | Used for |
|---|---|---|
| Open-Meteo archive API | 10y daily T/Td/precip/wind/cloud | climatology + model training |
| Open-Meteo forecast API | hourly data incl. sunset+2h values | evening snapshot, daily aggregates |
| HA Bernacca station (optional) | real on-site T/Td/humidity/wind/soil | richer features via token |

## Optional: connect the real weather station

The standalone scripts can't authenticate to HA without a token. To feed the
model with on-site Bernacca data instead of (or in addition to) Open-Meteo:

1. HA → Profile → Security → **Long-Lived Access Tokens** → Create
2. Save it: `echo "eyJ..." > ~/frost-forecast/ha_token.txt`
3. Then run `./.venv/bin/python collector.py pull-ha` (add `pull_ha_longterm_stats()`
   to `daily.sh`) — `import-ha` + `ha_snapshot.json` also work for one-off
   station snapshots.

## Weather metrics in use

From **Bernacca station**: outdoor temperature, dewpoint, humidity, wind
speed/gust/direction, relative+absolute pressure, solar radiation/lux, UV,
rain rates, soil moisture, VPD, windchill/feels-like.
From **MeteoCasa (weather entity)**: condition, temperature, wind.
From **Open-Meteo**: 10-year history + hourly evening values, cloud cover.

## Model improvement loop

- 21:45 daily: snapshot → prediction stored → next day observed Tmin backfills
  the outcome and scores the prediction (`hit`).
- Sundays: aggregates refreshed + model retrained on all labeled data.
- `predict.py report` shows running accuracy; the ML layer's trust in itself
  grows automatically as real frost events accumulate.
