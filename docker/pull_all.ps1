# Pull the trainer-local-* images on a machine that won't build them.
if (-not $env:TT_REGISTRY) { $env:TT_REGISTRY = "ghcr.io/gcsgeospatial" }
foreach ($key in @("ptv3","ptv3_hag","randlanet","randlanet_hag","kpconvx_cold","kpconvx_cold_hag")) {
  docker pull "$env:TT_REGISTRY/trainer-local-$key:latest"
}
