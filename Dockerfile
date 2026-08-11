# ============================================================
# API + pipeline
# ============================================================
# Dvě fáze: první přeloží kolečka (scipy, neurokit2 umí táhnout kompilátor),
# druhá si odnese jen hotové virtualenv – výsledný obraz je bez build-essential.

FROM python:3.12-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install -r requirements.txt


FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    TZ=Europe/Prague

# libpq pro psycopg, curl pro healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
        libpq5 curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 garmin

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
COPY --chown=garmin:garmin alembic.ini pyproject.toml ./
COPY --chown=garmin:garmin alembic ./alembic
COPY --chown=garmin:garmin config ./config
COPY --chown=garmin:garmin src ./src
COPY --chown=garmin:garmin scripts ./scripts

# Sem se mountují volumes: FIT soubory, CSV exporty, tokeny Garminu, logy.
RUN mkdir -p /app/data /app/logs /app/.garth /app/.garminconnect \
    && chown -R garmin:garmin /app/data /app/logs /app/.garth /app/.garminconnect

USER garmin
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
