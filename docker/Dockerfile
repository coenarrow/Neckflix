# syntax=docker/dockerfile:1
ARG PYTHON_VERSION=3.12

# --- Stage 1: build the ECF codec plugin from vendored sources ---
FROM python:${PYTHON_VERSION}-slim-bookworm AS plugin_builder
RUN apt-get update && \
    apt-get install -qqy --no-install-recommends \
        build-essential cmake libhdf5-dev && \
    rm -rf /var/lib/apt/lists/*
WORKDIR /build
COPY plugin ./plugin
COPY build_scripts/build_plugin.sh ./build_scripts/
RUN bash build_scripts/build_plugin.sh /opt/hdf5_plugin

# --- Stage 2: runtime ---
FROM python:${PYTHON_VERSION}-slim-bookworm AS runtime
ENV HDF5_PLUGIN_PATH=/opt/hdf5_plugin \
    PATH="/neckflix/.venv/bin:$PATH" \
    UV_LINK_MODE=copy
# libhdf5 runtime: the plugin links the system libhdf5
RUN apt-get update && \
    apt-get install -qqy --no-install-recommends libhdf5-dev && \
    rm -rf /var/lib/apt/lists/*
COPY --from=plugin_builder /opt/hdf5_plugin /opt/hdf5_plugin
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
WORKDIR /neckflix
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv sync --frozen --no-dev
ENTRYPOINT ["neckflix-preprocess"]
CMD ["--help"]
