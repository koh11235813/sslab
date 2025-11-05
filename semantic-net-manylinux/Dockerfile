FROM python:3.11-slim AS base

ARG UV_EXTRA=cpu
ENV UV_EXTRA=${UV_EXTRA}
ENV UV_PROJECT_ENV=/opt/semantic/.venv
ENV PATH="${UV_PROJECT_ENV}/bin:${PATH}"
ENV PYTHONUNBUFFERED=1

WORKDIR /opt/semantic

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock README.md ./
COPY src ./src
COPY scripts ./scripts
COPY configs ./configs

RUN uv sync --extra "${UV_EXTRA}" --frozen

ENTRYPOINT ["uv", "run"]
CMD ["python", "run_single.py", "--help"]
