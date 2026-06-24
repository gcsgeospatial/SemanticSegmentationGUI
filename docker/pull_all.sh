#!/usr/bin/env bash
# Pull the trainer-local-* images on a machine that won't build them.
# Usage:  TT_REGISTRY=ghcr.io/<you> bash docker/pull_all.sh
set -euo pipefail
: "${TT_REGISTRY:?set TT_REGISTRY=ghcr.io/<your-user-or-org> (docker login first)}"
for key in ptv3 ptv3_hag randlanet randlanet_hag kpconvx_cold kpconvx_cold_hag; do
  docker pull "$TT_REGISTRY/trainer-local-$key:latest"
done
