FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg \
    PYTHONPATH=/workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    build-essential \
    ca-certificates \
    curl \
    git \
    gosu \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements-dev.txt /tmp/stad/
COPY docker/entrypoint.sh /usr/local/bin/stad-entrypoint.sh

RUN python -m pip install --upgrade pip setuptools wheel && \
    python -m pip install -r /tmp/stad/requirements.txt && \
    python -m pip install -r /tmp/stad/requirements-dev.txt

RUN chmod +x /usr/local/bin/stad-entrypoint.sh

WORKDIR /workspace
ENTRYPOINT ["/usr/local/bin/stad-entrypoint.sh"]

CMD ["bash"]
