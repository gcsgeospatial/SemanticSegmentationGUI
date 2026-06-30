"""Self-check for partition_scenes (scene-level train/val/test split).
Run: python test_scene_split.py  — plain asserts, no framework.
Imports the helper from local_train_kpconvx_cold.py (all 6 scripts hold an
identical copy), so this validates the logic every script shares."""
import importlib.util
import os


def _load_helper():
    p = os.path.join(os.path.dirname(__file__), "local_train_kpconvx_cold.py")
    spec = importlib.util.spec_from_file_location("_kpx_for_test", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)          # safe: run() is guarded by __main__
    return m.partition_scenes


def main():
    part = _load_helper()
    pool = [f"s{i}" for i in range(20)]

    # 1. deterministic
    assert part(pool, [], 0.15, 0.15, 42) == part(pool, [], 0.15, 0.15, 42)

    # 2. disjoint + covers the whole pool
    tr, va, te = part(pool, [], 0.15, 0.15, 42)
    assert set(tr) | set(va) | set(te) == set(pool)
    assert set(tr).isdisjoint(va) and set(tr).isdisjoint(te) and set(va).isdisjoint(te)

    # 3. fractions honored (~15% of 20 = 3 each)
    assert (len(tr), len(va), len(te)) == (14, 3, 3), (len(tr), len(va), len(te))

    # 4. a dedicated test set WINS — test_frac ignored, no train scene leaks into test
    tr2, va2, te2 = part(pool, ["X", "Y"], 0.15, 0.9, 7)
    assert te2 == ["X", "Y"]
    assert set(te2).isdisjoint(pool)
    assert (len(tr2), len(va2)) == (17, 3)

    # 5. always >=1 train scene, graceful at tiny N
    for n in range(1, 6):
        small = [f"a{i}" for i in range(n)]
        t, v, x = part(small, [], 0.3, 0.3, 1)
        assert len(t) >= 1, (n, t)
        assert set(t) | set(v) | set(x) == set(small)
        assert set(t).isdisjoint(v) and set(t).isdisjoint(x) and set(v).isdisjoint(x)

    print("ok")


if __name__ == "__main__":
    main()
