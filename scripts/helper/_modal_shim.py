"""Offline stand-in for the parts of `modal` the modal_train_*.py scripts use.

Lets each training script import and run with NO Modal account / network:

  * `Volume.from_name(...)` -> object whose methods (`.commit()`, ...) are no-ops.
  * `@app.function(...)` / `@app.local_entrypoint()` -> return plain callables;
    `train_X.remote(**kw)` (and `__call__`) run the wrapped body **in-process**.
  * `Image.debian_slim().apt_install().pip_install()...` -> a *recorder*: every
    builder call is captured (kind + payload) so tools/check_env_sync.py can diff a
    Dockerfile straight from the recipe the script already wrote.

Activate before importing a train script:

    import _modal_shim; _modal_shim.install()
    import modal_train_ptv3            # its `import modal` now resolves here
    _modal_shim.App.last              # the script's app (entrypoint + image)

The modal_train_*.py scripts only DEFINE things at module level (image, volumes, decorated
functions) — they never branch on Modal values — so importing under the shim is
side-effect-free and needs neither torch nor CUDA (both live inside the body).
"""

from __future__ import annotations

import sys
import types


# --------------------------------------------------------------------------- #
# Image builder — records each step, every method returns self (chainable).
# --------------------------------------------------------------------------- #
class Recipe:
    def __init__(self):
        self.python_version: str | None = None
        self.base: tuple = ("debian_slim", None)
        self.steps: list[tuple] = []   # (kind, payload) in call order

    # entry point (modal.Image.debian_slim(...))
    @classmethod
    def debian_slim(cls, python_version=None, **_kw):
        img = cls()
        img.base = ("debian_slim", python_version)
        img.python_version = python_version
        return img

    # builder methods (record + chain)
    def apt_install(self, *pkgs, **_kw):
        self.steps.append(("apt", [str(p) for p in pkgs]))
        return self

    def pip_install(self, *pkgs, index_url=None, extra_index_url=None,
                    find_links=None, pre=False, **_kw):
        self.steps.append(("pip", {"pkgs": [str(p) for p in pkgs],
                                   "index_url": index_url,
                                   "extra_index_url": extra_index_url,
                                   "find_links": find_links, "pre": bool(pre)}))
        return self

    def pip_install_from_requirements(self, path, **_kw):
        self.steps.append(("pip_req", str(path)))
        return self

    def add_local_dir(self, src, remote_path, copy=False, **_kw):
        self.steps.append(("copy_dir", {"src": str(src), "dst": str(remote_path)}))
        return self

    def add_local_file(self, src, remote_path, copy=False, **_kw):
        self.steps.append(("copy_file", {"src": str(src), "dst": str(remote_path)}))
        return self

    def run_commands(self, *cmds, **_kw):
        self.steps.append(("run", [str(c) for c in cmds]))
        return self

    def env(self, mapping, **_kw):
        self.steps.append(("env", dict(mapping)))
        return self

    def workdir(self, path, **_kw):
        self.steps.append(("workdir", str(path)))
        return self

    # any builder method we didn't enumerate: chain, record nothing.
    def __getattr__(self, _name):
        return lambda *a, **k: self


class _ImageNS:
    """Namespace object that `modal.Image` resolves to."""
    debian_slim = Recipe.debian_slim


# --------------------------------------------------------------------------- #
# Volume — every method is a no-op (no cloud).
# --------------------------------------------------------------------------- #
class Volume:
    @staticmethod
    def from_name(name, create_if_missing=False, **_kw):
        return Volume()

    def commit(self, *a, **k):
        return None

    def __getattr__(self, _name):           # reload, batch_upload, listdir, ...
        return lambda *a, **k: None


# --------------------------------------------------------------------------- #
# Function wrapper — .remote / __call__ run the body here.
# --------------------------------------------------------------------------- #
class Function:
    def __init__(self, fn):
        self._fn = fn

    def remote(self, *a, **k):
        return self._fn(*a, **k)

    def __call__(self, *a, **k):
        return self._fn(*a, **k)


def _is_bare(dargs, dkw):
    return len(dargs) == 1 and callable(dargs[0]) and not dkw


class App:
    last: "App | None" = None   # most-recently constructed app (launcher/gen find it here)

    def __init__(self, name=None, **_kw):
        self.name = name
        self.entrypoint = None      # the @app.local_entrypoint function
        self.image = None           # image passed to the (first) @app.function
        App.last = self

    def function(self, *dargs, **dkw):
        if dkw.get("image") is not None and self.image is None:
            self.image = dkw["image"]

        def deco(fn):
            return Function(fn)
        return deco(dargs[0]) if _is_bare(dargs, dkw) else deco

    def local_entrypoint(self, *dargs, **dkw):
        def deco(fn):
            self.entrypoint = fn
            return fn
        return deco(dargs[0]) if _is_bare(dargs, dkw) else deco


# small odds & ends the scripts reference at decoration time
def _noop_factory(*_a, **_k):
    return None


def _build_module() -> types.ModuleType:
    m = types.ModuleType("modal")
    m.App = App
    m.Image = _ImageNS
    m.Volume = Volume
    m.Retries = _noop_factory

    # Anything else (`modal.<x>`) -> a permissive no-op, so an unforeseen symbol
    # in one of the variant scripts can't break module import.
    def __getattr__(name):            # PEP 562 module-level getattr
        # Dunders MUST raise, not return a no-op. inspect.getmodule() scans every
        # sys.modules entry and reads `.__file__` (torch does this at import time in
        # register_debug_prims); a function there makes inspect call
        # `<func>.endswith(...)` -> AttributeError. The real symbols the scripts use
        # are all set explicitly above, so only junk lookups reach here.
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        return _noop_factory
    m.__getattr__ = __getattr__
    return m


def install() -> types.ModuleType:
    """Register the shim as `modal` in sys.modules and return it."""
    mod = _build_module()
    sys.modules["modal"] = mod
    return mod
