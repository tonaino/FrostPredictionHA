#!/usr/bin/env bash
# Daily frost-forecast pipeline: collect -> backfill outcomes -> predict (-> train weekly)
# Uses the Bernacca HA station when a HA token exists, Open-Meteo otherwise.
# (On a server, prefer Docker: docker compose up -d — scheduler.py replaces this script.)
set -e
cd /Users/antonio/frost-forecast
[ -f .env ] && set -a && source .env && set +a
PY=./.venv/bin/python

$PY collector.py outcomes
if [ -f ha_token.txt ] && [ -s ha_token.txt ]; then
  $PY collector.py snapshot-ha
  # station history refresh: 7 days back daily keeps aggregates fresh
  if [ "$(date +%u)" = "3" ]; then
    $PY collector.py pull-ha || true
  fi
else
  $PY collector.py snapshot
fi
# weekly aggregates refresh + retrain (Open-Meteo based, always)
if [ "$(date +%u)" = "7" ]; then
  $PY collector.py pull-history
  $PY predict.py train
fi
$PY predict.py tonight
