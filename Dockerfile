FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /build

COPY requirements-build.lock ./

RUN python -m pip install --requirement requirements-build.lock

COPY pyproject.toml README.md LICENSE alembic.ini ./
COPY app ./app
COPY alembic ./alembic
COPY config ./config
COPY docs/personal-humanizer ./docs/personal-humanizer
COPY docs/analyze-personal-voice ./docs/analyze-personal-voice
COPY scripts/run_external_semantic_holdout_v5.py ./scripts/run_external_semantic_holdout_v5.py

RUN python -m build --wheel --no-isolation --outdir /wheel .


FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    ALEMBIC_CONFIG=/app/alembic.ini \
    DATA_DIR=/app/data \
    DRAFTS_DIR=/app/drafts \
    LOGS_DIR=/app/logs \
    CONFIG_DIR=/app/config \
    AUTO_PUBLISH=false

WORKDIR /app

RUN groupadd --gid 10001 vouch \
    && useradd \
        --uid 10001 \
        --gid vouch \
        --create-home \
        --shell /usr/sbin/nologin \
        vouch

COPY requirements.lock ./
COPY --from=builder /wheel/vouch-*.whl /tmp/wheels/

RUN python -m pip install --requirement requirements.lock \
    && python -m pip install --no-deps /tmp/wheels/vouch-*.whl \
    && rm -rf /tmp/wheels

COPY alembic.ini ./
COPY alembic ./alembic
COPY config ./config
COPY scripts/docker_entrypoint.py ./scripts/docker_entrypoint.py

RUN mkdir -p \
        /app/data \
        /app/drafts \
        /app/logs \
        /app/media \
    && chown -R vouch:vouch \
        /app/data \
        /app/drafts \
        /app/logs \
        /app/media

USER vouch

EXPOSE 8000

ENTRYPOINT ["python", "scripts/docker_entrypoint.py"]
CMD ["python", "-m", "app.cli", "serve"]