"""LEGACY weight-era compatibility for --mode infer. DELETE BEFORE PRODUCTION.

Makes every pre-current era of trained weights runnable at inference:
  Era 0  IEEE run_config.json runs (display-name backbones, no features spec,
         kp checkpoints trained with a per-tile z-min channel)
  Era 1  *_hag twin scripts (rgb_r/rgb_g/rgb_b + uppercase HAG tokens)
  Era 2  wrapper era (hag_source-driven manifests)
  Era 3  FEAT_CHANNELS era (extinct channel names, feat_ aliases)

Rules: channels that can't be produced anymore ride as zeros; the one surviving
computation is the z-min column era-0/3 kp checkpoints were trained with (kept
HERE only - never in FEAT_VOCAB or any UI). CRS needs nothing weight-side:
inputs re-convert through the current CRS pipeline (declare an EPSG as usual).

Removal: delete this file and grep 'LEGACY' for the handful of call sites
(local_train_kpconv / kpconvx_cold / randlanet / ptv3 / concerto, infer_page).
"""

import json
import os

# the era-0/3 z-min channel token; only translate/spec functions below emit it
LEGACY_ZMIN = "legacy_zmin"

BACKBONE_KEYS = {"RandLA-Net": "randlanet", "KPConvX-L": "kpconvx_cold",
                 "PTv3": "ptv3", "KPConv": "kpconv"}

_ALIAS = {"feat_intensity": "intensity", "feat_return_number": "return_number",
          "feat_returnnumber": "return_number", "feat_ret_num": "return_number",
          "ret_num": "return_number", "hag": "feat_hag"}


def translate_features(feats, hag_source=None):
    """Era-1/3 feature tokens -> the modern spec: rgb_r/g/b collapse to rgb,
    HAG -> feat_hag, aliases canonicalize, the removed z-derived channel maps
    to feat_hag when the run actually trained on HAG else to LEGACY_ZMIN."""
    out = []
    for n in feats or []:
        s = str(n).strip()
        low = s.lower()
        if low in ("rgb_r", "rgb_g", "rgb_b"):
            s = "rgb"
        elif low in _ALIAS:
            s = _ALIAS[low]
        elif low == "height":
            s = "feat_hag" if hag_source else LEGACY_ZMIN
        if s not in out:
            out.append(s)
    return out


def _raw_manifest(weights_path):
    d = os.path.dirname(weights_path)
    if os.path.basename(d) == "checkpoints":
        d = os.path.dirname(d)
    for fn in ("run.json", "run_config.json"):
        p = os.path.join(d, fn)
        if os.path.exists(p):
            try:
                with open(p, encoding="utf-8") as f:
                    return json.load(f)
            except (OSError, ValueError):
                return None
    return None


def kp_legacy_spec(weights_path):
    """(FEAT_SPEC, note) for a kp-family checkpoint's manifest, or (None, None)
    when nothing legacy applies (caller falls back to its default)."""
    m = _raw_manifest(weights_path)
    if not m:
        return None, None
    feats = m.get("features")
    if feats:
        out = translate_features(feats, hag_source=m.get("hag_source"))
        note = (None if out == list(feats)
                else f"features translated: {list(feats)} -> {out}")
        return out, note
    if m.get("hag_source") or "hag" in str(m.get("feature_mode") or ""):
        return (["intensity", "return_number", "feat_hag"],
                "spec-less HAG-era manifest -> [intensity, return_number, feat_hag]")
    if m.get("feature_mode") or m.get("feature_recipe") or m.get("input_channels"):
        return (["intensity", "return_number", LEGACY_ZMIN],
                "era-0 native manifest -> [intensity, return_number, z-min]")
    return None, None


def randlanet_spec_from_width(in_dim):
    """(FEAT_SPEC, note) from the checkpoint's fc0 input width, for spec-less
    era-0 manifests; (None, None) for widths the eras never produced."""
    if in_dim == 3:
        return ["x", "y", "z"], "era-0 warm-start checkpoint (fc0 width 3) -> [x, y, z]"
    if in_dim == 5:
        return (["x", "y", "z", "intensity", "return_number"],
                "era-0 checkpoint (fc0 width 5) -> [x, y, z, intensity, return_number]")
    return None, None


def wrap_build_feat(bf):
    """Supply columns the core build_feat no longer knows: LEGACY_ZMIN gets the
    real per-tile z-min the checkpoint trained on; any other extinct non-feat_*
    token is fed as zeros. Identity for modern specs."""
    import numpy as np
    known = {"x", "y", "z", "intensity", "return_number"}
    dead = [n for n in bf.spec if n not in known and not n.startswith("feat_")]
    if not dead:
        return bf

    def wrapped(xyz, intensity, ret_num, drop=(), extras=None):
        ex = dict(extras or {})
        for n in dead:
            ex[n] = ((xyz[:, 2] - xyz[:, 2].min()).astype(np.float32)
                     if n == LEGACY_ZMIN else np.zeros(len(xyz), np.float32))
        return bf(xyz, intensity, ret_num, drop=drop, extras=ex)

    wrapped.spec = bf.spec
    return wrapped


if __name__ == "__main__":
    import numpy as np
    assert translate_features(["x", "y", "z", "rgb_r", "rgb_g", "rgb_b", "HAG"]) \
        == ["x", "y", "z", "rgb", "feat_hag"]
    assert translate_features(["intensity", "return_number", "height"]) \
        == ["intensity", "return_number", LEGACY_ZMIN]
    assert translate_features(["height"], hag_source="hag_delaunay+labels") == ["feat_hag"]
    assert translate_features(["feat_intensity", "feat_geo_sphericity"]) \
        == ["intensity", "feat_geo_sphericity"]
    assert randlanet_spec_from_width(3)[0] == ["x", "y", "z"]
    assert randlanet_spec_from_width(5)[0] == ["x", "y", "z", "intensity", "return_number"]
    assert randlanet_spec_from_width(4) == (None, None)

    class _BF:                       # mimic kp_make_build_feat's contract
        spec = ["intensity", "return_number", LEGACY_ZMIN, "gone"]

        def __call__(self, xyz, intensity, ret_num, drop=(), extras=None):
            return extras
    ex = wrap_build_feat(_BF())(np.array([[0., 0., 1.], [0., 0., 3.]], np.float32),
                                None, None)
    assert np.allclose(ex[LEGACY_ZMIN], [0.0, 2.0]) and np.all(ex["gone"] == 0.0)
    import tempfile
    d = tempfile.mkdtemp()
    json.dump({"feature_mode": "native", "input_channels": 4},
              open(os.path.join(d, "run_config.json"), "w"))
    spec, _ = kp_legacy_spec(os.path.join(d, "final_model.pth"))
    assert spec == ["intensity", "return_number", LEGACY_ZMIN]
    print("ok")
