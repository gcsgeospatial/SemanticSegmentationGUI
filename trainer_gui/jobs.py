"""QProcess job runner with live line streaming + training-log metric parsing."""

from __future__ import annotations

import re
import threading
import traceback

from PySide6.QtCore import QObject, QProcess, QProcessEnvironment, Signal

# Every training script prints per-epoch summaries in this exact shape:
#   ep  12: loss=0.4321 acc=0.9123 miou=0.7012 s/iter=0.123 s/ep=61.4
EPOCH_RE = re.compile(
    r"ep\s+(\d+):\s+loss=([\d.]+)\s+acc=([\d.]+)\s+miou=([\d.]+)"
    r"(?:\s+lr=[\d.eE+-]+)?"   # PTv3 cold recipe prints lr= here; skip it
    r"(?:\s+s/iter=([\d.]+))?(?:\s+s/ep=([\d.]+))?")
RUN_DIR_RE = re.compile(r"/outputs/runs/(\S+)")


class LogParser(QObject):
    """Feeds on raw log text; emits structured epoch metrics and the run id."""

    epoch = Signal(dict)     # {epoch, loss, acc, miou, sec_per_iter, sec_per_epoch}
    run_id = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._buf = ""
        self._run_id_seen = False

    def feed(self, text: str):
        self._buf += text
        *lines, self._buf = self._buf.split("\n")
        for line in lines:
            m = EPOCH_RE.search(line)
            if m:
                self.epoch.emit({
                    "epoch": int(m.group(1)),
                    "loss": float(m.group(2)),
                    "acc": float(m.group(3)),
                    "miou": float(m.group(4)),
                    "sec_per_iter": float(m.group(5)) if m.group(5) else None,
                    "sec_per_epoch": float(m.group(6)) if m.group(6) else None,
                })
            if not self._run_id_seen:
                r = RUN_DIR_RE.search(line)
                if r:
                    self._run_id_seen = True
                    self.run_id.emit(r.group(1).split("/")[0])


class JobRunner(QObject):
    """One external process: merged stdout/stderr streamed line-ish, UTF-8 safe."""

    output = Signal(str)
    finished = Signal(int)   # exit code
    failed = Signal(str)     # QProcess error description

    def __init__(self, parent=None):
        super().__init__(parent)
        self.proc: QProcess | None = None

    @property
    def running(self) -> bool:
        return self.proc is not None and self.proc.state() != QProcess.NotRunning

    def start(self, program: str, args: list[str], cwd: str = "",
              extra_env: dict | None = None, pre: tuple | None = None):
        """Run `program args`. If `pre=(program, args)` is given it runs first and
        its exit code is IGNORED (used for idempotent `modal volume create`, which
        errors when the volume already exists), then the main command runs and its
        code is the one `finished` reports."""
        if self.running:
            raise RuntimeError("JobRunner already has a live process")
        self._cwd = cwd
        self._extra_env = extra_env
        self._main = (program, list(args))
        if pre is not None:
            self._stage = "pre"
            self._launch(pre[0], list(pre[1]))
        else:
            self._stage = "main"
            self._launch(program, list(args))

    def _launch(self, program: str, args: list[str]):
        env = QProcessEnvironment.systemEnvironment()
        # Modal prints ✓ and box-drawing chars; on Windows the child's stdout
        # defaults to cp1252 and crashes encoding them (silently aborting e.g. a
        # `volume put`). Force UTF-8 two ways — PYTHONUTF8 enables UTF-8 mode,
        # PYTHONIOENCODING pins the stream encoding even for libs (rich/click)
        # that read it directly.
        env.insert("PYTHONUTF8", "1")
        env.insert("PYTHONIOENCODING", "utf-8")
        env.insert("PYTHONUNBUFFERED", "1")  # line-by-line streaming
        for k, v in (self._extra_env or {}).items():
            env.insert(k, str(v))
        self.proc = QProcess(self)
        self.proc.setProcessEnvironment(env)
        self.proc.setProcessChannelMode(QProcess.MergedChannels)
        if self._cwd:
            self.proc.setWorkingDirectory(self._cwd)
        self.proc.readyReadStandardOutput.connect(self._on_output)
        self.proc.finished.connect(self._on_finished)
        self.proc.errorOccurred.connect(
            lambda e: self.failed.emit(str(e)))
        self.proc.start(program, args)

    def terminate(self):
        if self.running:
            self.proc.kill()

    def _on_output(self):
        data = bytes(self.proc.readAllStandardOutput()).decode("utf-8", "replace")
        self.output.emit(data)

    def _on_finished(self, code, _status):
        if self._stage == "pre":      # ignore create's exit; run the real command
            self._stage = "main"
            self._launch(*self._main)
            return
        self.finished.emit(int(code))
        self.proc = None


class Stopped(Exception):
    """Raised inside a job's progress() callback when the user hits Stop."""


class FuncWorker(QObject):
    """Run a Python callable on a background thread; signals are queued to the
    GUI thread. The callable receives a `progress(str)` callback.

    Cancellation is cooperative: cancel() sets a flag and the next progress()
    call raises Stopped, so a job only stops at its own checkpoints (scenes call
    progress() one-per-scene, so a build stops between scenes)."""

    output = Signal(str)
    done = Signal(object)    # return value
    error = Signal(str)
    stopped = Signal()       # user cancelled; job unwound cleanly

    def __init__(self, parent=None):
        super().__init__(parent)
        self._thread: threading.Thread | None = None
        self._cancel = threading.Event()

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self, fn, *args, **kwargs):
        if self.running:
            raise RuntimeError("FuncWorker already running")
        self._cancel.clear()

        def progress(s):
            if self._cancel.is_set():
                raise Stopped()
            self.output.emit(s)

        def _run():
            try:
                result = fn(*args, progress=progress, **kwargs)
            except Stopped:
                self.stopped.emit()
            except Exception:
                self.error.emit(traceback.format_exc())
            else:
                self.done.emit(result)

        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()

    def cancel(self):
        """Cooperative stop: the running job bails at its next progress() call."""
        self._cancel.set()
