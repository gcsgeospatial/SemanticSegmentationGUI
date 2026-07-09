# Modal shells (active again as of 2026-07-09)

These are **thin shells**: each bakes its `scripts/local/local_train_*.py` twin +
`scripts/helper/train_common.py` into a `modal.Image` and subprocesses the local
script in the cloud, so local and Modal run byte-identical trainer code.

**All training/inference logic lives in `scripts/local/` — edit the local twin,
never these shells.** The shells only own: image/deps, GPU/timeout/retries,
volume mounts, flag forwarding, and the `--env-json` passthrough that delivers
the GUI's `LOSS_*` / `RARE_*` / `DG_*` / `EVAL_VOTES` knob overrides to the
trainer subprocess.

Model architecture sources are **pinned upstream git clones** (PTv3 =
Pointcept/PointTransformerV3, RandLA-Net = tsunghan-wu/RandLA-Net-pytorch,
KPConvX = apple/ml-kpconvx `Standalone/KPConvX`), each at a fixed commit SHA in
the image recipe — so images build identically on any machine, with no local
model checkouts. Bump a SHA deliberately; it is the architecture version. After
editing any image recipe, re-run `python tools/gen_dockerfiles.py` so the local
Dockerfiles follow.

Contracts (mirroring the local Docker path):

- datasets: the single `terminal-datasets` volume (override: `TT_DATASET_VOLUME`)
  mounted at `/datasets`, one dataset per `/<name>` — the Datasets page uploads
  there; inference scenes go to `/_infer/<job_id>`
- outputs: per-backbone `<app>-outputs` volume mounted at `/outputs`; runs land
  at `runs/<id>` and the Inference page reads weights from there
- GPU type / timeout: `TT_GPU` / `TT_TIMEOUT_HOURS` env at `modal run` time
  (the GUI sets `TT_GPU` from the Train page's GPU picker)

Caveat: this box (no CUDA, no Modal account) can only compile-check the shells;
first real `modal run` validates end to end.
