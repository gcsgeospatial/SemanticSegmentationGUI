#!/usr/bin/env bash
# Tag + push the locally-built trainer-local-* images to a registry so other machines can pull them (no model repos needed — the code is baked in).
# Usage:  TT_REGISTRY=ghcr.io/<you> bash docker/push_all.sh   (after: docker login <registry>)
set -euo pipefail
: "${TT_REGISTRY:?set TT_REGISTRY=ghcr.io/<your-user-or-org> (docker login first)}"
for key in ptv3 ptv3_hag randlanet randlanet_hag kpconvx_cold kpconvx_cold_hag; do
  docker tag "trainer-local-$key" "$TT_REGISTRY/trainer-local-$key:latest"
  docker push "$TT_REGISTRY/trainer-local-$key:latest"
done
echo "pushed to $TT_REGISTRY — set TT_REGISTRY (or local_config['registry']) there too."
