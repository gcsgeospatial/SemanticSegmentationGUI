"""find_latest_unfinished_run recipe matching (group A fix).

Run: python tests/test_fix_a_train_common.py
"""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "scripts", "helper"))

with tempfile.TemporaryDirectory() as tmp:
    os.environ["TT_OUTPUTS_ROOT"] = tmp
    import train_common as tc

    cfg = {"grid_size": 0.1, "chunk_xy": 40.0, "n_epochs": 50,
           "num_classes": 3, "class_names": ["a", "b", "c"],
           "features": ("x", "y", "z", "rgb")}

    def make_run(run_id, rc, done=False):
        rd = f"{tmp}/runs/{run_id}"
        os.makedirs(f"{rd}/checkpoints")
        open(f"{rd}/checkpoints/ep007.pth", "w").close()
        with open(f"{rd}/run.json", "w") as f:
            json.dump(rc, f)
        if done:
            open(f"{rd}/DONE", "w").close()
        return run_id

    def found(suffix, cfg=None):    # (run_id, epoch) or None; glob may use \
        r = tc.find_latest_unfinished_run(suffix, cfg) if cfg is not None \
            else tc.find_latest_unfinished_run(suffix)
        return None if r is None else (os.path.basename(r[0]), r[2])

    # same recipe (tuple vs json list must still match) -> resume
    rid = make_run("20260101_000000_ds_ptv3", dict(cfg, features=list(cfg["features"])))
    assert found("ds_ptv3", cfg) == (rid, 7)

    # newer run with a reordered class list -> skipped, older one resumed
    make_run("20260102_000000_ds_ptv3", dict(cfg, class_names=["b", "a", "c"]))
    assert found("ds_ptv3", cfg) == (rid, 7)

    # every candidate mismatched -> fresh run
    assert found("ds_ptv3", dict(cfg, chunk_xy=80.0)) is None

    # DONE and suffix rules still hold; no cfg = legacy unchecked behaviour
    make_run("20260103_000000_ds_ptv3", dict(cfg), done=True)
    assert found("ds_ptv3", cfg) == (rid, 7)
    assert found("ds_kpconv", cfg) is None
    assert found("ds_ptv3") == ("20260102_000000_ds_ptv3", 7)

print("ok")
