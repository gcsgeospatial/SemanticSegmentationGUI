# Pull the trainer-local-* images on a machine that won't build them.
if (-not $env:TT_REGISTRY) { $env:TT_REGISTRY = "ghcr.io/gcsgeospatial" }
foreach ($key in @("ptv3","randlanet","kpconvx_cold","kpconv","concerto","sonata","utonia")) {
  docker pull "$env:TT_REGISTRY/trainer-local-$key:latest"
}
