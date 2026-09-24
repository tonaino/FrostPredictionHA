FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TZ=Europe/Sofia

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY db.py features.py model.py collector.py predict.py scheduler.py ecowitt_import.py rolling_model.py ./

# data volume: database + trained models survive container updates
RUN mkdir -p /app/data && chown -R 1000:1000 /app
ENV FROST_DB_PATH=/app/data/frost.db \
    FROST_MODEL_DIR=/app/data/models
VOLUME ["/app/data"]

USER 1000

CMD ["python", "scheduler.py"]
