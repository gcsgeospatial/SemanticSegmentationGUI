"""Unified training terminal - desktop GUI orchestrating the Modal training scripts."""

__version__ = "0.1.0"


def _preload_modern_libstdcxx() -> None:
    """Linux: make the conda env's libstdc++ win over the (often older) system one.

    pip wheels with C++ extensions (pgeof) need a recent C++ ABI
    (CXXABI_1.3.15), which PySide6/Qt normally supplies at startup - but anything
    importing them before Qt can resolve to the system libstdc++.so.6 and fail
    with 'version CXXABI_… not found'. Preloading the env's libstdc++ GLOBAL here
    (this runs first) makes its newer symbols satisfy every extension imported
    afterwards, and it is a no-op off Linux or if the env ships none.
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
