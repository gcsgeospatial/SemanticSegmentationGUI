#!/usr/bin/env bash
# Pull the trainer-local-* images on a machine that won't build them.
# Usage:  bash docker/pull_all.sh   (TT_REGISTRY=ghcr.io/gcsgeospatial by default)
set -euo pipefail
: "${TT_REGISTRY:=ghcr.io/gcsgeospatial}"
for key in ptv3 randlanet kpconvx_cold kpconv concerto sonata utonia; do
  docker pull "$TT_REGISTRY/trainer-local-$key:latest"
done
