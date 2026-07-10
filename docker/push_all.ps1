# Tag + push the locally-built trainer-local-* images to a registry so other machines can pull them (no model repos needed — the code is baked in).
if (-not $env:TT_REGISTRY) { $env:TT_REGISTRY = "ghcr.io/gcsgeospatial" }
foreach ($key in @("ptv3","ptv3_hag","randlanet","randlanet_hag","kpconvx_cold","kpconvx_cold_hag","kpconv","kpconv_hag")) {
  docker tag "trainer-local-$key" "$env:TT_REGISTRY/trainer-local-$key:latest"
  docker push "$env:TT_REGISTRY/trainer-local-$key:latest"
}
