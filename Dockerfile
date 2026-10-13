FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc g++ \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    "numpy>=1.24" \
    "scipy>=1.11" \
    "nibabel>=5.0" \
    "dipy>=1.12"

WORKDIR /opt/app
