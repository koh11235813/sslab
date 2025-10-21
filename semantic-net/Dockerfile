# syntax=docker/dockerfile:1

FROM nvcr.io/nvidia/l4t-base:r36.4.0

ARG DEBIAN_FRONTEND=noninteractive

ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PYTHONUNBUFFERED=1

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-venv \
        python3-dev \
        build-essential \
        libopenblas-base \
        libomp5 \
        libjpeg-dev \
        zlib1g-dev \
        curl \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh -s -- --install-dir /usr/local/bin

WORKDIR /workspace/semantic-net

COPY pyproject.toml README.md uv.lock ./
COPY src ./src
COPY configs ./configs
COPY wheels ./wheels

RUN uv sync --extra jetson && \
    .venv/bin/python -m pip cache purge

ENV VIRTUAL_ENV=/workspace/semantic-net/.venv
ENV PATH="/workspace/semantic-net/.venv/bin:$PATH"

CMD ["bash"]
