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
              extra_env: dict | None = None):
        if self.running:
            raise RuntimeError("JobRunner already has a live process")
        env = QProcessEnvironment.systemEnvironment()
        env.insert("PYTHONUTF8", "1")        # Modal prints ✓ — avoid cp1252 crashes
        env.insert("PYTHONUNBUFFERED", "1")  # line-by-line streaming
        for k, v in (extra_env or {}).items():
            env.insert(k, str(v))
        self.proc = QProcess(self)
        self.proc.setProcessEnvironment(env)
        self.proc.setProcessChannelMode(QProcess.MergedChannels)
        if cwd:
            self.proc.setWorkingDirectory(cwd)
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
        self.finished.emit(int(code))
        self.proc = None


class FuncWorker(QObject):
    """Run a Python callable on a background thread; signals are queued to the
    GUI thread. The callable receives a `progress(str)` callback."""

    output = Signal(str)
    done = Signal(object)    # return value
    error = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._thread: threading.Thread | None = None

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self, fn, *args, **kwargs):
        if self.running:
            raise RuntimeError("FuncWorker already running")

        def _run():
            try:
                result = fn(*args, progress=self.output.emit, **kwargs)
            except Exception:
                self.error.emit(traceback.format_exc())
            else:
                self.done.emit(result)

        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()
