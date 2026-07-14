"""Engine <-> TUI bridge.

The scientific engine is a set of synchronous handlers that call
``questionary.<widget>(...).ask()`` in the middle of their work and ``print``
progress to stdout.  To drive them from a full-screen prompt_toolkit app without
ever dropping to an inline prompt, we:

  * run the handler on a WORKER thread,
  * replace the engine modules' ``questionary`` reference with a compatible shim
    whose ``.ask()`` marshals the prompt to the UI thread and BLOCKS the worker
    on a Future until the user answers,
  * redirect stdout/stderr into a bounded console buffer (ANSI stripped) and treat
    carriage-return updates (tqdm / spinner) as a live progress line.

No engine logic is changed — only the I/O seams are rerouted.
"""

import concurrent.futures
import re
import sys
import threading

_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def strip_ansi(s):
    return _ANSI.sub("", s)


# =========================================================================
# Bridge — shared between the worker thread and the UI thread
# =========================================================================
class Bridge:
    MAX_CONSOLE = 400

    def __init__(self):
        self.app = None                 # prompt_toolkit Application (set by TUI)
        self.tui = None                 # BadBloodTUI (set by TUI)
        self._future = None             # pending answer Future
        self.prompt = None              # current prompt spec dict, or None
        self.prompt_seq = 0             # monotonic id of the current prompt
        self.console = []               # captured output lines (ANSI-stripped)
        self.progress = None            # live \r line (tqdm / spinner), or None
        self.preview = None             # spectral preview data {'x','y','markers'}
        self.running = False
        self._lock = threading.Lock()

    # -- worker-thread side ------------------------------------------------
    def ask(self, spec):
        """Called FROM the worker thread. Blocks until the UI responds."""
        fut = concurrent.futures.Future()
        with self._lock:
            self._future = fut
            self.prompt = spec
            self.prompt_seq += 1
        self._invalidate()
        result = fut.result()           # blocks the worker thread
        with self._lock:
            self.prompt = None
        self._invalidate()
        return result

    # -- UI-thread side ----------------------------------------------------
    def respond(self, value):
        """Called FROM the UI thread with the user's answer."""
        with self._lock:
            fut, self._future = self._future, None
            self.prompt = None
        if fut and not fut.done():
            fut.set_result(value)
        self._invalidate()

    def has_prompt(self):
        return self.prompt is not None

    # -- console capture ---------------------------------------------------
    def emit(self, text, carriage=False):
        clean = strip_ansi(text)
        with self._lock:
            if carriage:
                self.progress = clean.strip() or None
            else:
                for line in clean.split("\n"):
                    self.console.append(line)
                if len(self.console) > self.MAX_CONSOLE:
                    del self.console[:-self.MAX_CONSOLE]
                self.progress = None
        self._invalidate()

    def reset_console(self):
        with self._lock:
            self.console = []
            self.progress = None
            self.preview = None

    # -- spectral preview (braille plot) -----------------------------------
    def set_preview(self, data):
        with self._lock:
            self.preview = data
        self._invalidate()

    def clear_preview(self):
        with self._lock:
            self.preview = None
        self._invalidate()

    def _invalidate(self):
        if self.app is not None:
            try:
                self.app.invalidate()
            except Exception:
                pass


# Global singleton (the engine shim imports this).
BRIDGE = Bridge()


# =========================================================================
# Captured output stream (stdout / stderr / tqdm target)
# =========================================================================
class ConsoleStream:
    """File-like object routing engine output into the bridge console.

    Splits on \\r (progress, in-place update) and \\n (committed line).
    """

    def __init__(self, bridge=BRIDGE):
        self.bridge = bridge
        self._buf = ""

    def write(self, s):
        if not s:
            return
        self._buf += s
        # Process carriage-return progress updates and newline commits.
        while True:
            r = self._buf.find("\r")
            n = self._buf.find("\n")
            if r == -1 and n == -1:
                break
            if n != -1 and (r == -1 or n < r):
                line, self._buf = self._buf[:n], self._buf[n + 1:]
                self.bridge.emit(line, carriage=False)
            else:
                line, self._buf = self._buf[:r], self._buf[r + 1:]
                if line:
                    self.bridge.emit(line, carriage=True)

    def flush(self):
        pass

    def isatty(self):
        return False


# =========================================================================
# questionary-compatible shim
# =========================================================================
class _Question:
    def __init__(self, spec):
        self.spec = spec

    def ask(self, *args, **kwargs):
        return BRIDGE.ask(self.spec)

    # questionary also exposes ask_async / unsafe_ask; keep parity.
    def unsafe_ask(self, *args, **kwargs):
        return BRIDGE.ask(self.spec)


def _choice_text(c):
    # NB: str has a `.title` METHOD, so getattr(str, "title") would wrongly return
    # the method — handle plain strings explicitly.
    if isinstance(c, str):
        return c
    return getattr(c, "title", str(c))


def confirm(message, default=True, **kw):
    return _Question({"type": "confirm", "message": str(message), "default": bool(default)})


def text(message, default="", validate=None, qmark=None, **kw):
    return _Question({"type": "text", "message": str(message),
                      "default": default or "", "validate": validate})


def path(message, default="", validate=None, qmark=None, **kw):
    return _Question({"type": "path", "message": str(message),
                      "default": default or "", "validate": validate})


def select(message, choices, default=None, qmark=None, **kw):
    return _Question({"type": "select", "message": str(message),
                      "choices": [_choice_text(c) for c in choices],
                      "default": default})


def checkbox(message, choices, qmark=None, **kw):
    return _Question({"type": "checkbox", "message": str(message),
                      "choices": [_choice_text(c) for c in choices]})


# Objects passed through by engine code (rare); provide a no-op Choice.
class Choice:
    def __init__(self, title, value=None, **kw):
        self.title = title
        self.value = value if value is not None else title
