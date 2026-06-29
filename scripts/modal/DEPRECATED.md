# ⚠️ Modal scripts — paused / deprecated (as of 2026-06-27)

Work on the **Modal** version of the trainer is **paused**. We are focusing on the
**local (Docker)** path only for now. Treat everything in this folder as frozen.

**Do not** add features, fix bugs, or refactor:
- `scripts/modal/modal_train_*.py` (these scripts)
- `trainer_gui/modal_cli.py`
- the Modal branches of `trainer_gui/pages/infer_page.py` and `train_page.py`

…until this note is removed.

## Why

The local pipeline is being finished first, and Modal can't be exercised on the
dev box anyway (Intel Arc — no CUDA — and no Modal account here), so changes there
can't be validated.

## Where to work instead

All real training/inference logic lives in **`scripts/local/`**. The modal scripts
are just thin shells: each bakes its `scripts/local/local_train_*.py` twin +
`scripts/helper/train_common.py` into a `modal.Image` and subprocesses the local
script in the cloud. Edit the local twin.

Local inference is driven by the self-contained **`run.json`** a local run writes
(picked directly in the GUI); the Modal run-id flow is not maintained for now.

## Status

Nothing is deleted — the scripts still import and the GUI's Modal mode still runs.
They are simply **out of scope**. To resume: delete this file and re-validate the
Modal paths.
