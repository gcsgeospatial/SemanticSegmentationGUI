#!/usr/bin/env bash
# Pull the trainer-local-* images on a machine that won't build them.
# Usage:  bash docker/pull_all.sh   (TT_REGISTRY=ghcr.io/gcsgeospatial by default)
set -euo pipefail
: "${TT_REGISTRY:=ghcr.io/gcsgeospatial}"
for key in ptv3 ptv3_hag randlanet randlanet_hag kpconvx_cold kpconvx_cold_hag; do
  docker pull "$TT_REGISTRY/trainer-local-$key:latest"
done
