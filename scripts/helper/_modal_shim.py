"""Offline stand-in for `modal`: install() registers it in sys.modules so
modal_train_* scripts import with no account/network. Volumes no-op, functions
run in-process, Image builders record steps for tools/check_env_sync.py."""

from __future__ import annotations

import sys
import types


class Recipe:
    def __init__(self):
        self.python_version: str | None = None
        self.base: tuple = ("debian_slim", None)
        self.steps: list[tuple] = []   # (kind, payload) in call order

    @classmethod
    def debian_slim(cls, python_version=None, **_kw):
        img = cls()
        img.base = ("debian_slim", python_version)
        img.python_version = python_version
        return img

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

    # unenumerated builder methods: chain, record nothing
    def __getattr__(self, _name):
        return lambda *a, **k: self


class _ImageNS:
    """Namespace object that `modal.Image` resolves to."""
    debian_slim = Recipe.debian_slim


class Volume:
    @staticmethod
    def from_name(name, create_if_missing=False, **_kw):
        return Volume()

    def commit(self, *a, **k):
        return None

    def __getattr__(self, _name):           # reload, batch_upload, listdir, ...
        return lambda *a, **k: None


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


def _noop_factory(*_a, **_k):
    return None


def _build_module() -> types.ModuleType:
    m = types.ModuleType("modal")
    m.App = App
    m.Image = _ImageNS
    m.Volume = Volume
    m.Retries = _noop_factory

    def __getattr__(name):            # PEP 562: any other modal.<x> -> no-op
        # dunders MUST raise: inspect.getmodule (via torch import) reads
        # __file__ on every sys.modules entry and chokes on a function
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
