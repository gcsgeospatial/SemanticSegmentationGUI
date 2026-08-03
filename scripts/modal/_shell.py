"""Shared plumbing for the modal_train_* shells: app + volumes + GPU function
kwargs + the launch print. Each per-backbone file keeps only its docstring,
image chain, flag list and typed entrypoint signature."""

import os

import modal


def setup(app_name, gpu_default="A100", memory=49152):
    """(app, outputs_volume, datasets_volume, function_kwargs, launch)."""
    gpu = os.environ.get("TT_GPU", gpu_default)
    hours = int(os.environ.get("TT_TIMEOUT_HOURS", "24"))
    app = modal.App(app_name)
    outputs_volume = modal.Volume.from_name(
        os.environ.get("TT_OUTPUTS_VOLUME") or f"{app_name}-outputs", create_if_missing=True)
    datasets_volume = modal.Volume.from_name(
        os.environ.get("TT_DATASET_VOLUME", "terminal-datasets"), create_if_missing=True)
    function_kwargs = dict(
        gpu=gpu,
        volumes={"/outputs": outputs_volume, "/datasets": datasets_volume},
        cpu=8,
        memory=memory,
        timeout=hours * 3600,
        # auto-restart on failure; each retry auto-resumes from the last checkpoint
        retries=modal.Retries(max_retries=10, backoff_coefficient=1.0, initial_delay=5.0),
    )

    def launch(remote_fn, mode="train", weights=None, dataset=None, **kw):
        what = {"eval": "eval-only re-score", "infer": f"infer({weights})"}.get(
            mode, f"train({dataset})")
        print(f"Launching {app_name} [{what}] on {gpu} for up to {hours}h.")
        remote_fn.remote(mode=mode, weights=weights, dataset=dataset, **kw)

    return app, outputs_volume, datasets_volume, function_kwargs, launch
