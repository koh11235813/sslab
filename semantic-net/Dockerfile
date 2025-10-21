# syntax=docker/dockerfile:1

FROM nvcr.io/nvidia/l4t-base:r36.4.0

ARG DEBIAN_FRONTEND=noninteractive

ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

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
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --upgrade pip setuptools wheel

ARG TORCH_WHEEL_URL=https://developer.download.nvidia.com/compute/redist/jp/v61/pytorch/torch-2.5.0a0+872d972e41.nv24.08.17622132-cp310-cp310-linux_aarch64.whl
ARG TORCHVISION_WHEEL=torchvision-0.20.0a0+afc54f7-cp310-cp310-linux_aarch64.whl

COPY wheels/${TORCHVISION_WHEEL} /tmp/torchvision.whl

RUN pip install --no-cache-dir "${TORCH_WHEEL_URL}" && \
    pip install --no-cache-dir /tmp/torchvision.whl && \
    rm /tmp/torchvision.whl

WORKDIR /workspace/semantic-net

COPY pyproject.toml README.md uv.lock ./
COPY src ./src
COPY configs ./configs
COPY wheels ./wheels

RUN pip install --no-cache-dir -e .

CMD ["bash"]
