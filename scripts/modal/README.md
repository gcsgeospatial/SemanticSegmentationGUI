# Modal shells

These are **thin shells**: each bakes its `scripts/local/local_train_*.py` twin +
`scripts/helper/train_common.py` into a `modal.Image` and subprocesses the local
script in the cloud, so local and Modal run byte-identical trainer code.

**All training/inference logic lives in `scripts/local/` - edit the local twin,
never these shells.** The shells only own: image/deps, GPU/timeout/retries,
volume mounts, flag forwarding, and the `--env-json` passthrough that delivers
the GUI's `LOSS_*` / `RARE_*` / `DG_*` / `EVAL_VOTES` knob overrides to the
trainer subprocess.

Model architecture sources are **pinned upstream git clones** (PTv3 =
Pointcept/PointTransformerV3, RandLA-Net = tsunghan-wu/RandLA-Net-pytorch,
KPConvX = orion-hoch/ml-kpconvx-windows-acessible `Standalone/KPConvX`,
KPConv = HuguesTHOMAS/KPConv-PyTorch, Sonata = facebookresearch/sonata,
Concerto = Pointcept/Concerto, Utonia = Pointcept/Utonia), each at a fixed
commit SHA in the image recipe - so images build identically on any machine,
with no local model checkouts. Bump a SHA deliberately; it is the architecture
version. After editing any image recipe, mirror the change in `envs/pixi.toml`
and the matching `conda-recipes/trainer-src-*` recipe -
`python tools/check_env_sync.py` fails until they agree.

Contracts (mirroring the local pixi path):

- datasets: one shared volume named on the Datasets page (default
  `terminal-datasets`; env override `TT_DATASET_VOLUME`) mounted at `/datasets`,
  one dataset per `/<name>` - the Datasets page uploads there; inference scenes
  go to `/_infer/<job_id>`
- outputs: per-backbone `<app>-outputs` volume mounted at `/outputs` (or one
  shared volume named on the Train page; env override `TT_OUTPUTS_VOLUME`);
  runs land at `runs/<id>` and the Inference page reads weights from there
- GPU type / timeout: `TT_GPU` / `TT_TIMEOUT_HOURS` env at `modal run` time
  (the Train page sets both)
- graceful stop: an uploaded `/STOP` on the outputs volume (the Train page's
  "Stop at nearest checkpoint" does this) reaches the container within ~2
  minutes; the trainer stops after that epoch and still runs its full final
  evaluation + finalize
  (the GUI sets `TT_GPU` from the Train page's GPU picker)
