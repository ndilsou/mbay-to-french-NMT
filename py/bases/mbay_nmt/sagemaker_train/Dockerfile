FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3-pip curl && \
    rm -rf /var/lib/apt/lists/*

# Install uv for fast dep resolution.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /opt/ml/code

# Copy project files.
COPY py/pyproject.toml py/uv.lock ./
COPY py/components/ components/
COPY py/bases/ bases/

# Install deps (locked).
RUN uv sync --frozen --no-dev

# SageMaker convention: SAGEMAKER_PROGRAM points to entrypoint.
ENV SAGEMAKER_PROGRAM=entrypoint.py

ENTRYPOINT ["uv", "run", "python", "-m", "mbay_nmt.sagemaker_train.entrypoint"]
