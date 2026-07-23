"""QProcess job runner with live line streaming + training-log metric parsing."""

from __future__ import annotations

import codecs
import re
import threading
import traceback

from PySide6.QtCore import QObject, QProcess, QProcessEnvironment, Signal

# trainer log shape: ep  12: loss=0.4321 acc=0.9123 miou=0.7012 s/iter=0.123 s/ep=61.4
EPOCH_RE = re.compile(
    r"ep\s+(\d+):\s+loss=([\d.]+)\s+acc=([\d.]+)\s+miou=([\d.]+)"
    r"(?:\s+lr=[\d.eE+-]+)?"   # PTv3 prints lr= here
    r"(?:\s+s/iter=([\d.]+))?(?:\s+s/ep=([\d.]+))?")
# [val@ep9] acc=... mIoU(5-way)=... mIoU(present 4)=...; test@epN deliberately doesn't match
VAL_RE = re.compile(
    r"\[(?:val|eval)@ep(\d+)\]\s+acc=([\d.]+)\s+"
    r"mIoU\(\d+-way\)=([\d.]+)\s+mIoU\(present \d+\)=([\d.]+)")
RUN_DIR_RE = re.compile(r"/outputs/runs/(\S+)")


class LogParser(QObject):
    """Feeds on raw log text; emits structured epoch metrics and the run id."""

    epoch = Signal(dict)        # {epoch, loss, acc, miou, sec_per_iter, sec_per_epoch}
    val_metrics = Signal(dict)  # {epoch, acc, miou (present-classes), miou_all (N-way)}
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
            v = VAL_RE.search(line)
            if v:
                self.val_metrics.emit({
                    "epoch": int(v.group(1)),
                    "acc": float(v.group(2)),
                    "miou_all": float(v.group(3)),
                    "miou": float(v.group(4)),   # headline: present-classes mIoU
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
        """Run `program args`. An optional pre=(program, args) runs first with its
        exit code ignored (idempotent `modal volume create`)."""
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
        # force UTF-8 both ways: Windows cp1252 stdout crashes on Modal's ✓/box chars
        env.insert("PYTHONUTF8", "1")
        env.insert("PYTHONIOENCODING", "utf-8")
        env.insert("PYTHONUNBUFFERED", "1")
        for k, v in (self._extra_env or {}).items():
            env.insert(k, str(v))
        # incremental decoder: multi-byte chars split across chunks must not become U+FFFD
        self._dec = codecs.getincrementaldecoder("utf-8")("replace")
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
        # read from the emitting process, not self.proc — the next stage may already own it
        proc = self.sender()
        if proc is not None:
            text = self._dec.decode(bytes(proc.readAllStandardOutput()))
            if text:
                self.output.emit(text)

    def _on_finished(self, code, _status):
        if self._stage == "pre":
            self._stage = "main"
            self._launch(*self._main)
            return
        # null BEFORE emitting: the finished slot may start() the next stage
        self.proc = None
        self.finished.emit(int(code))


class Stopped(Exception):
    """Raised inside a job's progress() callback when the user hits Stop."""


class FuncWorker(QObject):
    """Run a callable on a background thread with a progress(str) callback.
    Cancellation is cooperative: the next progress() call raises Stopped."""

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
