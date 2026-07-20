"""Drift checker: envs/pixi.toml + conda-recipes/ vs the modal.Image recipes.

The modal_train_*.py scripts stay the single source of truth for training
deps. This tool imports each under the offline `_modal_shim` (exactly like
gen_dockerfiles.py did), re-derives per backbone:

  - the pip package pins + torch index / find-links URLs
  - the pinned model-source git SHA

and diffs them against the pixi feature's pypi-dependencies / pypi-options
and the trainer-src recipe's context sha. Run by the smoke test; exits
non-zero listing every mismatch on drift.

    python tools/check_env_sync.py
"""

from __future__ import annotations

import os
import re
import sys
import tomllib

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts", "modal"))
sys.path.insert(0, os.path.join(REPO, "scripts", "helper"))

import _modal_shim  # noqa: E402

# backbone key -> (modal script, pixi feature, trainer-src recipe dir)
BACKBONES = {
    "ptv3": ("modal_train_ptv3.py", "ptv3", "trainer-src-ptv3"),
    "randlanet": ("modal_train_randlanet.py", "randlanet", "trainer-src-randlanet"),
    "kpconvx_cold": ("modal_train_kpconvx_cold.py", "kpconvx-cold", "trainer-src-kpconvx"),
    "kpconv": ("modal_train_kpconv.py", "kpconv", "trainer-src-kpconv"),
    "concerto": ("modal_train_concerto.py", "concerto", "trainer-src-concerto"),
    "sonata": ("modal_train_sonata.py", "sonata", "trainer-src-sonata"),
    "utonia": ("modal_train_utonia.py", "utonia", "trainer-src-utonia"),
}

# pip deps that are Modal-shell plumbing, not part of the training env contract
_IGNORE = {"modal"}


def _norm(name: str) -> str:
    return name.lower().replace("_", "-")


def _split(spec: str) -> tuple[str, str]:
    """'numpy<2.0' -> ('numpy', '<2.0'); 'scipy' -> ('scipy', '')."""
    m = re.match(r"([A-Za-z0-9._-]+)\s*(.*)", spec.strip())
    return _norm(m.group(1)), m.group(2).replace(" ", "")


def _base_ver(constraint: str) -> str:
    """Strip a +local tag so '==2.1.0+cu118' compares equal to '==2.1.0'."""
    return constraint.split("+", 1)[0]


def _recipe(script: str):
    """Import the modal script under the shim and return its recorded Image."""
    import importlib
    _modal_shim.install()
    stem = os.path.splitext(script)[0]
    for m in list(sys.modules):
        if m.startswith("modal_train_"):
            sys.modules.pop(m, None)
    mod = importlib.import_module(stem)
    img = _modal_shim.App.last.image or getattr(mod, "image", None)
    if img is None or not getattr(img, "steps", None):
        raise RuntimeError(f"{stem}: no recorded Image recipe")
    return img


def modal_spec(script: str) -> dict:
    """{pins: {name: constraint}, index_url, find_links, sha} from the recipe."""
    img = _recipe(script)
    pins, index_url, find_links, sha = {}, None, set(), None
    for kind, p in img.steps:
        if kind == "pip":
            for pkg in p["pkgs"]:
                name, con = _split(pkg)
                if name not in _IGNORE:
                    pins[name] = con
            index_url = index_url or p.get("index_url")
            if p.get("find_links"):
                find_links.add(p["find_links"])
        elif kind == "run":
            for cmd in p:
                m = re.search(r"checkout --detach ([0-9a-f]{7,40})", cmd)
                if m:
                    sha = m.group(1)
    return {"pins": pins, "index_url": index_url,
            "find_links": find_links, "sha": sha}


def pixi_spec(feature: str, manifest: dict) -> dict:
    feat = manifest["feature"][feature]
    pins = {}
    for name, spec in feat.get("pypi-dependencies", {}).items():
        con = spec.get("version", "*") if isinstance(spec, dict) else spec
        pins[_norm(name)] = "" if con == "*" else con.replace(" ", "")
    opts = feat.get("pypi-options", {})
    fl = {e.get("url") for e in opts.get("find-links", []) if e.get("url")}
    return {"pins": pins, "index_url": opts.get("index-url"), "find_links": fl}


def recipe_sha(recipe_dir: str) -> str | None:
    path = os.path.join(REPO, "conda-recipes", recipe_dir, "recipe.yaml")
    try:
        with open(path, encoding="utf-8") as f:
            m = re.search(r'sha:\s*"([0-9a-f]{7,40})"', f.read())
        return m.group(1) if m else None
    except OSError:
        return None


def compare(key: str, modal: dict, pixi: dict, rsha: str | None) -> list[str]:
    errs = []
    for name, con in modal["pins"].items():
        if name not in pixi["pins"]:
            errs.append(f"{key}: pip dep '{name}' in modal recipe but not in pixi feature")
        elif _base_ver(pixi["pins"][name]) != _base_ver(con):
            errs.append(f"{key}: '{name}' pin drift — modal '{con}' vs pixi "
                        f"'{pixi['pins'][name]}'")
    for name in pixi["pins"]:
        if name not in modal["pins"]:
            errs.append(f"{key}: pip dep '{name}' in pixi feature but not in modal recipe")
    if modal["index_url"] and pixi["index_url"] != modal["index_url"]:
        errs.append(f"{key}: index-url drift — modal '{modal['index_url']}' vs pixi "
                    f"'{pixi['index_url']}'")
    for fl in modal["find_links"]:
        if fl not in pixi["find_links"]:
            errs.append(f"{key}: find-links '{fl}' in modal recipe but not in pixi feature")
    if modal["sha"] and rsha != modal["sha"]:
        errs.append(f"{key}: model-source SHA drift — modal '{modal['sha']}' vs "
                    f"recipe '{rsha}'")
    return errs


def check_all() -> list[str]:
    with open(os.path.join(REPO, "envs", "pixi.toml"), "rb") as f:
        manifest = tomllib.load(f)
    errs = []
    for key, (script, feature, recipe_dir) in BACKBONES.items():
        errs += compare(key, modal_spec(script), pixi_spec(feature, manifest),
                        recipe_sha(recipe_dir))
    return errs


def main():
    errs = check_all()
    for e in errs:
        print(f"  DRIFT  {e}")
    if errs:
        sys.exit(f"{len(errs)} drift(s) between modal recipes and pixi/conda specs — "
                 "fix envs/pixi.toml / conda-recipes to mirror scripts/modal.")
    print(f"env sync OK: {len(BACKBONES)} backbones match their modal recipes")


if __name__ == "__main__":
    main()
