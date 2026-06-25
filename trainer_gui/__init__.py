"""Unified training terminal — desktop GUI orchestrating the Modal training scripts."""

__version__ = "0.1.0"


def _preload_modern_libstdcxx() -> None:
    """Linux: make the conda env's libstdc++ win over the (often older) system one.

    open3d (pip wheel) and matplotlib are compiled extensions that need a recent
    C++ ABI (e.g. CXXABI_1.3.15). The desktop app gets it for free — PySide6/Qt
    loads the env's libstdc++ at startup — but the standalone viewer subprocess
    (`python -m trainer_gui.viewer`) imports matplotlib/open3d with no Qt loaded,
    so the dynamic loader can resolve to /lib/x86_64-linux-gnu/libstdc++.so.6 and
    they fail with 'version CXXABI_… not found'. Preloading the env's libstdc++
    GLOBAL here (this runs first — both entrypoints import the package) makes its
    newer symbols satisfy every extension imported afterwards. No-op off Linux or
    if the env ships none (then the system lib is the only one anyway).
    """
    import sys
    if sys.platform != "linux":
        return
    import ctypes
    import glob
    import os
    for cand in sorted(glob.glob(os.path.join(sys.prefix, "lib", "libstdc++.so.6*")),
                       reverse=True):
        try:
            ctypes.CDLL(cand, mode=ctypes.RTLD_GLOBAL)
            return
        except OSError:
            continue


_preload_modern_libstdcxx()
