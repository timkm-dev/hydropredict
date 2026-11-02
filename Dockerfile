# Hosted dashboard image. Place the runtime package (scripts/package_runtime.py)
# at ./runtime.zip before building; it is verified and unpacked during the build.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
# XGBoost needs the OpenMP runtime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
# Editable install keeps the package ROOT at /app, where frontend/ models/ data/ live.
RUN pip install "torch>=2.6,<3" --index-url https://download.pytorch.org/whl/cpu \
    && pip install -e ".[lstm,bearing]"

COPY scripts ./scripts
COPY frontend ./frontend
COPY data/cwru_manifest.csv ./data/
RUN --mount=type=bind,source=runtime.zip,target=/tmp/runtime.zip \
    python -m scripts.package_runtime --restore /tmp/runtime.zip

COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh
EXPOSE 8000
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
