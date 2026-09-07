# Obico ML (Hailo) Home Assistant addon — single, self-contained build.
#
# Starts from a stock python:3.11-slim-bookworm and provisions everything the
# failure detector needs:
#   - HailoRT (hailo_platform + hailort runtime .deb) — runs obico_part1.hef and
#     obico_part2.hef on the Hailo-8/8L device
#   - ONNX Runtime — runs decode.onnx (host-side YOLOv5 head)
#   - OpenCV headless — 416-crop + BGR2RGB preprocessing
#   - s6-overlay + bashio + tempio — Home Assistant addon machinery
#
# HailoRT is provisioned from hailo_assets/ (offline) or downloaded from
# https://dev-public.hailo.ai/<RELEASE>/ at build time if the binaries are
# missing. The file names are derived from BUILD_ARCH: aarch64 → arm64.deb +
# linux_aarch64 wheel, amd64 → amd64.deb + linux_x86_64 wheel. See
# hailo_assets/README.md.

# ---------------------------------------------------------------------------
# HailoRT provisioning params (used with the stock python base below)
# ---------------------------------------------------------------------------
ARG HAILORT_VERSION=4.21.0
ARG HAILORT_RELEASE=2025_04
ARG HAILORT_BASE_URL=https://dev-public.hailo.ai/${HAILORT_RELEASE}

# HA addon machinery versions / base
ARG BUILD_FROM=python:3.11-slim-bookworm
ARG BUILD_ARCH=aarch64
# hadolint ignore=DL3006
FROM ${BUILD_FROM}

ENV \
    CARGO_NET_GIT_FETCH_WITH_CLI=true \
    DEBIAN_FRONTEND="noninteractive" \
    HOME="/root" \
    LANG="C.UTF-8" \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_PREFER_BINARY=1 \
    PS1="$(whoami)@$(hostname):$(pwd)$ " \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    S6_BEHAVIOUR_IF_STAGE2_FAILS=2 \
    S6_CMD_WAIT_FOR_SERVICES_MAXTIME=0 \
    S6_CMD_WAIT_FOR_SERVICES=1 \
    YARN_HTTP_TIMEOUT=1000000 \
    TERM="xterm-256color"

# Set shell
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# ---------------------------------------------------------------------------
# Hailo runtime media: system libs for the format detector.
#   libgomp1            OpenMP runtime required by ONNX Runtime
#   libglib2.0-0        shared lib required by opencv-python-headless
#   libssl3 libudev1    HailoRT runtime libs
#   gcc python3-dev     to build netifaces (hailort 4.21 runtime dep)
# ---------------------------------------------------------------------------
RUN apt-get update \
    && apt-get install --no-install-recommends --assume-yes \
        ca-certificates curl jq tzdata xz-utils \
        libgomp1 libglib2.0-0 libssl3 libudev1 \
        gcc python3-dev wget \
    && rm -rf /var/lib/apt/lists/*

# The hailo_assets/ dir always exists in the context (kept, possibly empty).
COPY hailo_assets/ /hailo_assets_local/

# Re-declare the HailoRT ARGs after the FROM (they are build-scoped).
ARG HAILORT_VERSION=4.21.0
ARG HAILORT_RELEASE=2025_04
ARG HAILORT_BASE_URL=https://dev-public.hailo.ai/${HAILORT_RELEASE}
ARG BUILD_ARCH=aarch64

# Install HailoRT runtime (.deb → lib/usr/bin) — offline from hailo_assets/ if
# present, otherwise downloaded from the public Hailo release server.
RUN \
    HAILORT_DEB_ARCH="arm64" \
    && if [ "${BUILD_ARCH}" = "amd64" ]; then HAILORT_DEB_ARCH="amd64"; fi \
    && if [ -f "/hailo_assets_local/hailort_${HAILORT_VERSION}_${HAILORT_DEB_ARCH}.deb" ]; then \
         HAILORT_DEB="/hailo_assets_local/hailort_${HAILORT_VERSION}_${HAILORT_DEB_ARCH}.deb"; \
       else \
         curl --fail --silent --show-error --location \
           --output /tmp/hailort.deb \
           "${HAILORT_BASE_URL}/hailort_${HAILORT_VERSION}_${HAILORT_DEB_ARCH}.deb"; \
         HAILORT_DEB="/tmp/hailort.deb"; \
       fi \
    && dpkg-deb -x "${HAILORT_DEB}" /tmp/hailort \
    && cp -a /tmp/hailort/usr/lib/. /usr/lib/ \
    && cp -a /tmp/hailort/usr/bin/. /usr/bin/ \
    && rm -rf /tmp/hailort /tmp/hailort.deb

# Install HailoRT python bindings + ONNX Runtime + OpenCV headless.
RUN pip install --no-cache-dir --upgrade pip \
    && HAILORT_WHEEL_TAG="cp311-cp311-linux_aarch64" \
    && if [ "${BUILD_ARCH}" = "amd64" ]; then HAILORT_WHEEL_TAG="cp311-cp311-linux_x86_64"; fi \
    && if [ -f "/hailo_assets_local/hailort-${HAILORT_VERSION}-${HAILORT_WHEEL_TAG}.whl" ]; then \
         HAILORT_WHEEL="/hailo_assets_local/hailort-${HAILORT_VERSION}-${HAILORT_WHEEL_TAG}.whl"; \
       else \
         HAILORT_WHEEL="${HAILORT_BASE_URL}/hailort-${HAILORT_VERSION}-${HAILORT_WHEEL_TAG}.whl"; \
       fi \
    && pip install --no-cache-dir \
        onnxruntime==1.29.0 \
        opencv-python-headless==4.11.0.86 \
        "${HAILORT_WHEEL}"

# Ship the HailoRT redistribution licenses with the binaries (required to
# redistribute HailoRT): MIT (libhailort, pyhailort, hailortcli) and
# LGPL-2.1-or-later (hailonet GStreamer plugin).
COPY hailo_assets/licenses/ /usr/share/licenses/hailort/

# ---------------------------------------------------------------------------
# Home Assistant Community Add-on base (s6-overlay + bashio + tempio)
# ---------------------------------------------------------------------------
ARG BASHIO_VERSION="v0.16.2"
ARG S6_OVERLAY_VERSION="3.2.0.0"
ARG TEMPIO_VERSION="2021.09.0"
ARG BUILD_ARCH=aarch64

RUN \
    S6_ARCH="${BUILD_ARCH}" \
    && if [ "${BUILD_ARCH}" = "i386" ]; then S6_ARCH="i686"; \
    elif [ "${BUILD_ARCH}" = "amd64" ]; then S6_ARCH="x86_64"; \
    elif [ "${BUILD_ARCH}" = "armv7" ]; then S6_ARCH="arm"; fi \
    \
    && curl -L -s "https://github.com/just-containers/s6-overlay/releases/download/v${S6_OVERLAY_VERSION}/s6-overlay-noarch.tar.xz" \
        | tar -C / -Jxpf - \
    \
    && curl -L -s "https://github.com/just-containers/s6-overlay/releases/download/v${S6_OVERLAY_VERSION}/s6-overlay-${S6_ARCH}.tar.xz" \
        | tar -C / -Jxpf - \
    \
    && curl -L -s "https://github.com/just-containers/s6-overlay/releases/download/v${S6_OVERLAY_VERSION}/s6-overlay-symlinks-noarch.tar.xz" \
        | tar -C / -Jxpf - \
    \
    && curl -L -s "https://github.com/just-containers/s6-overlay/releases/download/v${S6_OVERLAY_VERSION}/s6-overlay-symlinks-arch.tar.xz" \
        | tar -C / -Jxpf - \
    \
    && mkdir -p /etc/fix-attrs.d \
    && mkdir -p /etc/services.d \
    \
    && curl -J -L -o /tmp/bashio.tar.gz \
        "https://github.com/hassio-addons/bashio/archive/${BASHIO_VERSION}.tar.gz" \
    && mkdir /tmp/bashio \
    && tar zxvf \
        /tmp/bashio.tar.gz \
        --strip 1 -C /tmp/bashio \
    \
    && mv /tmp/bashio/lib /usr/lib/bashio \
    && ln -s /usr/lib/bashio/bashio /usr/bin/bashio \
    \
    && curl -L -s -o /usr/bin/tempio \
        "https://github.com/home-assistant/tempio/releases/download/${TEMPIO_VERSION}/tempio_${BUILD_ARCH}" \
    && chmod a+x /usr/bin/tempio \
    \
    && apt-get purge -y --auto-remove \
        xz-utils \
    && apt-get clean \
    && rm -fr \
        /tmp/* \
        /var/{cache,log}/* \
        /var/lib/apt/lists/*

# Copy root filesystem + s6-overlay adjustments
COPY rootfs /
COPY s6-overlay /package/admin/s6-overlay-${S6_OVERLAY_VERSION}/

# Belt and braces: even though rootfs no longer ships the nginx service or its
# user-bundle marker, never let a stale nginx service tree leak into s6-rc.d
# (s6-rc-compile is fatal if the "user" bundle references a missing service).
RUN rm -rf /etc/s6-overlay/s6-rc.d/nginx /etc/s6-overlay/s6-rc.d/user/contents.d/nginx

# ---------------------------------------------------------------------------
# Obico ML API
# ---------------------------------------------------------------------------
EXPOSE 3333

# Health check
HEALTHCHECK \
    CMD curl --fail http://0.0.0.0:3333/hc || exit 1

WORKDIR /app
ADD ./app /app
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Entrypoint & CMD
ENTRYPOINT [ "/init" ]

# Build arugments
ARG BUILD_DATE
ARG BUILD_REF
ARG BUILD_VERSION
ARG BUILD_REPOSITORY

# Labels
LABEL \
    io.hass.name="Obico ML (Hailo)" \
    io.hass.description="Home Assistant Add-on: Obico ML failure detection on Hailo 8/8L" \
    io.hass.arch="${BUILD_ARCH}" \
    io.hass.type="addon" \
    io.hass.version=${BUILD_VERSION} \
    io.hass.base.version=${BUILD_VERSION} \
    io.hass.base.name="obico-ml-hailo" \
    io.hass.base.image="obico-ml-hailo" \
    maintainer="mpeex" \
    org.opencontainers.image.title="Obico ML (Hailo)" \
    org.opencontainers.image.description="Home Assistant Add-on: Obico ML failure detection on Hailo 8/8L" \
    org.opencontainers.image.vendor="mpeex" \
    org.opencontainers.image.authors="mpeex" \
    org.opencontainers.image.licenses="AGPL-3.0" \
    org.opencontainers.image.url="https://github.com/mpeex/ha-obico-hailo" \
    org.opencontainers.image.source="https://github.com/mpeex/ha-obico-hailo" \
    org.opencontainers.image.documentation="https://github.com/mpeex/ha-obico-hailo/blob/main/README.md" \
    org.opencontainers.image.created=${BUILD_DATE} \
    org.opencontainers.image.revision=${BUILD_REF} \
    org.opencontainers.image.version=${BUILD_VERSION}
