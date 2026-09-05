#!/usr/bin/env bash
# ==============================================================================
# Cross-compile and publish the Obico ML (Hailo) addon image to GHCR.
#
# Supports cross-building for linux/arm64 (and linux/amd64) from any host using
# Docker buildx. On non-native hosts buildx boots a QEMU-backed builder.
#
# Usage:
#   ./build-push.sh               # linux/arm64, tag 0.1
#   ./build-push.sh 0.1 linux/amd64
#
# The image is named after the addon image from config.yaml, with {arch}
# replaced by the target arch (ghcr.io/mpeex/obico_ml_hailo_addon-aarch64:0.1).
#
# Prerequisites:
#   - docker with buildx (Docker Desktop / buildx plugin)
#   - docker login ghcr.io with a token that has write:packages
#
# Buildkit skips the Docker Desktop credstore on --push, which results in an
# anonymous 403 from GHCR. The helper below exports the saved ghcr.io creds as
# DOCKER_AUTH_CONFIG so the push authenticates (unless already set).
# ==============================================================================
set -euo pipefail

TAG="${1:-0.5}"
PLATFORM="${2:-linux/arm64}"

if [ -z "${DOCKER_AUTH_CONFIG:-}" ]; then
  stored_creds="$(printf '%s\n' 'ghcr.io' | docker-credential-desktop get 2>/dev/null || true)"
  if [ -n "${stored_creds}" ]; then
    stored_user="$(printf '%s' "${stored_creds}" | jq -r .Username)"
    stored_secret="$(printf '%s' "${stored_creds}" | jq -r .Secret)"
    if [ -n "${stored_user}" ] && [ -n "${stored_secret}" ]; then
      export DOCKER_AUTH_CONFIG="${DOCKER_AUTH_CONFIG:-$(printf '{"auths":{"ghcr.io":{"auth":"%s"}}}' "$(printf '%s:%s' "${stored_user}" "${stored_secret}" | base64 | tr -d '\n')")}"
    fi
  fi
fi

case "${PLATFORM}" in
  linux/arm64) BUILD_ARCH="aarch64" ;;
  linux/amd64) BUILD_ARCH="amd64" ;;
  *) echo "Unsupported platform: ${PLATFORM}"; exit 1 ;;
esac

REPO="ghcr.io/mpeex/obico_ml_hailo_addon-${BUILD_ARCH}"
IMAGE="${REPO}:${TAG}"

echo "► Building ${IMAGE} for ${PLATFORM} (BUILD_ARCH=${BUILD_ARCH})"

docker buildx create --name obico-builder --use --bootstrap 2>/dev/null || true

docker buildx build \
  --platform "${PLATFORM}" \
  --build-arg BUILD_ARCH="${BUILD_ARCH}" \
  --build-arg BUILD_VERSION="${TAG}" \
  --build-arg BUILD_REF="$(git rev-parse HEAD)" \
  --build-arg BUILD_DATE="$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  --tag "${IMAGE}" \
  --push \
  .

echo "✔ Published ${IMAGE}"