#!/usr/bin/env python3
"""Package a finished training run's weights as a conda package.

    pixi run --manifest-path envs/pixi.toml -e pkg package-weights <run_dir>

Reads <run_dir>/run.json (backbone + dataset), stages final_model.pth +
run.json, generates a rattler-build recipe, and builds
trainer-weights-<dataset>-<key> into conda-recipes/output/; upload with the
`upload` pixi task, and installed packages land in
$PREFIX/share/trainer-weights/<name>/ where the GUI's Infer page finds them.

Runs on any platform rattler-build supports (noarch package, no shell script:
the recipe uses python for the copy so Windows works too).
"""
import argparse
import datetime
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT_DIR = REPO / "conda-recipes" / "output"

RECIPE = """\
package:
  name: {name}
  version: "{version}"

source:
  - path: {staged}

build:
  number: 0
  noarch: generic
  script:
    interpreter: python
    content: |
      import os, shutil
      dst = os.path.join(os.environ["PREFIX"], "share", "trainer-weights", "{name}")
      os.makedirs(dst, exist_ok=True)
      for f in ("final_model.pth", "run.json"):
          shutil.copy2(os.path.join(os.environ["SRC_DIR"], f), dst)

about:
  summary: "{summary}"
"""


def slug(s):
    return re.sub(r"[^a-z0-9-]+", "-", str(s).lower()).strip("-")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", help="a finished runs/<id> dir (has final_model.pth + run.json)")
    ap.add_argument("--version", default=None,
                    help="package version (default: YYYY.M.D of today)")
    ap.add_argument("--output-dir", default=str(OUT_DIR),
                    help="channel dir to build into (default: conda-recipes/output)")
    args = ap.parse_args(argv)

    run = Path(args.run_dir)
    weights, manifest = run / "final_model.pth", run / "run.json"
    for f in (weights, manifest):
        if not f.exists():
            sys.exit(f"error: {f} not found. Package a FINISHED run")
    rc = json.loads(manifest.read_text(encoding="utf-8-sig"))
    backbone = rc.get("backbone") or "model"
    dataset = rc.get("dataset") or "custom"
    name = f"trainer-weights-{slug(dataset)}-{slug(backbone)}"
    today = datetime.date.today()
    version = args.version or f"{today.year}.{today.month}.{today.day}"

    with tempfile.TemporaryDirectory() as td:
        staged = Path(td) / "staged"
        staged.mkdir()
        for f in (weights, manifest):
            staged.joinpath(f.name).write_bytes(f.read_bytes())
        recipe = Path(td) / "recipe.yaml"
        recipe.write_text(RECIPE.format(
            name=name, version=version, staged=staged.as_posix(),
            summary=f"{backbone} weights trained on {dataset} ({run.name})"),
            encoding="utf-8")
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"[package-weights] {name} {version} <- {run}")
        # --no-include-recipe: without it the .conda embeds a second copy of the
        # staged source under info/recipe/, doubling multi-hundred-MB weights
        r = subprocess.run(["rattler-build", "build", "-r", str(recipe),
                            "--no-include-recipe", "--output-dir", args.output_dir])
        if r.returncode:
            sys.exit(r.returncode)
    built = sorted(out_dir.glob(f"noarch/{name}-{version}-*.conda"))
    print(f"[package-weights] built: {built[-1] if built else out_dir}")
    print("[package-weights] publish with: pixi run -e pkg upload")


if __name__ == "__main__":
    main()
