# Serving image: only app/ and its runtime dependencies. The model is NOT baked in:
# it is downloaded from the MLflow registry at startup (alias -> concrete version).
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /srv

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN useradd --create-home --uid 10001 appuser
COPY --chown=appuser:appuser app/ ./app/
USER appuser

# Commit the image is built from (docker build --build-arg GIT_SHA=...), shown by /health.
# Declared last so changing it does not invalidate the dependency layers.
ARG GIT_SHA=unknown
ENV GIT_SHA=${GIT_SHA}

EXPOSE 8000
# exec: uvicorn becomes PID 1 and receives SIGTERM, so the lifespan shutdown runs.
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
