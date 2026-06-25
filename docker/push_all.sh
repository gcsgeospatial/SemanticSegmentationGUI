#!/usr/bin/env bash
# Tag + push the locally-built trainer-local-* images to a registry so other machines can pull them (no model repos needed — the code is baked in).
# Usage:  bash docker/push_all.sh   (TT_REGISTRY=ghcr.io/gcsgeospatial by default; docker login first)
set -euo pipefail
: "${TT_REGISTRY:=ghcr.io/gcsgeospatial}"
for key in ptv3 ptv3_hag randlanet randlanet_hag kpconvx_cold kpconvx_cold_hag; do
  docker tag "trainer-local-$key" "$TT_REGISTRY/trainer-local-$key:latest"
  docker push "$TT_REGISTRY/trainer-local-$key:latest"
done
echo "pushed to $TT_REGISTRY — set TT_REGISTRY (or local_config['registry']) there too."
