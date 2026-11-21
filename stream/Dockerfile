# SENTINEL stream image: Kafka producer + real-time scorer + analytics API + animated flow page.
#   docker build -t sentinel-stream stream
#   docker run -p 8003:8003 -e KAFKA_BOOTSTRAP=kafka:9092 -e XGB_URL=http://xgb:8001 sentinel-stream            # scorer + API
#   docker run -e KAFKA_BOOTSTRAP=kafka:9092 sentinel-stream python -m sentinel_stream.producer               # producer
FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PIP_DEFAULT_TIMEOUT=300 STORE_PATH=/data/sentinel.db STORE_DIR=/data
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip pip install --retries 10 --timeout 300 -r requirements.txt
COPY sentinel_stream ./sentinel_stream
RUN mkdir -p /data
EXPOSE 8003
HEALTHCHECK --interval=30s --timeout=5s CMD curl -fs http://localhost:8003/live/health || exit 1
CMD ["uvicorn", "sentinel_stream.api:app", "--host", "0.0.0.0", "--port", "8003"]
