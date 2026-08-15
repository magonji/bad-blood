"""Full-screen retro TUI for Bad Blood.

State machine:  welcome -> hub -> {running (drives an engine handler), help} -> hub

The engine is untouched; handlers run on a worker thread and their questionary
prompts are marshalled to native full-screen widgets via bridge.BRIDGE.
"""

import threading

from prompt_toolkit.application import Application
from prompt_toolkit.filters import Condition
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.keys import Keys
from prompt_toolkit.layout import Layout, Window, HSplit, WindowAlign, DynamicContainer
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.styles import Style

from bad_blood_pkg import __version__, __author__
from bad_blood_pkg.tui import theme, runner
from bad_blood_pkg.tui.bridge import BRIDGE, ConsoleStream
from bad_blood_pkg.tui.plot import braille_fragments

# The guided wizard (check -> load -> prepare -> export) runs first, straight
# from the welcome screen.  Afterwards you land in this menu to iterate on the
# data already held in memory without reloading the .mzz files.
MENU = [
    ("prepare", "Re-prepare (QC)", "Re-run quality-control filters on the loaded spectra"),
    ("export",  "Export",          "Select wavenumbers, filter metadata, write out"),
]
ACTIONS = [
    ("newrun",  "New analysis",    "Start over with another folder"),
    ("help",    "Help",            "Workflow reference"),
    ("quit",    "Quit",            "Leave Bad Blood"),
]
ITEMS = MENU + ACTIONS
STEP_LABEL = {k: lbl for k, lbl, _ in ITEMS}
STEP_LABEL["pipeline"] = "Analysis"


class BadBloodTUI:
    def __init__(self, version=None, author=None):
        self.version = version or __version__
        self.author = author or __author__
        # Short display version: "3.0.0" -> "3.0", but keep a non-zero patch.
        _parts = self.version.split(".")
        self.version_display = (
            ".".join(_parts[:2])
            if len(_parts) >= 3 and _parts[2] in ("0", "")
            else self.version
        )
        self.state = "welcome"          # welcome | askdir | hub | running | help
        self._prev_state = "welcome"    # where to return from help
        self.index = 0
        self.status_msg = ""
        self.pipeline_dir = None         # folder entered on the welcome screen
        self.scroll_offset = 0           # console lines scrolled up from the bottom

        # engine-facing state
        self.directory = None
        self.df = None
        self.prepared_df = None
        self.n_loaded = None
        self.n_prepared = None
        self.active_step = None
        # None while a step runs; (ok, summary) once it ends. The running screen
        # stays up until ⏎ so the log can be read instead of vanishing.
        self.finished = None

        # prompt interaction state
        self.choice_index = 0
        self.checkbox_selected = set()
        self.input_buffer = ""
        self.input_error = None

        self.bridge = BRIDGE
        self.bridge.tui = self

        self.kb = self._build_keys()
        self.app = Application(
            layout=Layout(DynamicContainer(self._root)),
            key_bindings=self.kb,
            style=self._style(),
            full_screen=True,
            mouse_support=False,
            refresh_interval=0.15,
        )
        self.bridge.app = self.app

    # =====================================================================
    # style
    # =====================================================================
    def _style(self):
        return Style.from_dict({
            "subtitle": f"fg:{theme.GOLD}",
            "author": f"fg:{theme.DIM}",
            "press": f"fg:{theme.FG}",
            "press.key": f"fg:{theme.GOLD} bold",
            "header": f"fg:{theme.ACCENT} bold",
            "header.sep": f"fg:{theme.DIM}",
            "header.ver": f"fg:{theme.DIM}",
            "rule": f"fg:{theme.DIM}",
            "item": f"fg:{theme.FG}",
            "item.sel": f"fg:{theme.ACCENT_TOP} bold",
            "item.hint": f"fg:{theme.DIM}",
            "badge.done": f"fg:{theme.OK} bold",
            "badge.todo": f"fg:{theme.DIM}",
            "status": f"fg:{theme.DIM}",
            "status.key": f"fg:{theme.GOLD} bold",
            "status.msg": f"fg:{theme.GOLD}",
            "console": f"fg:{theme.DIM}",
            "console.section": f"fg:#ffffff bg:{theme.ACCENT} bold",
            "progress": f"fg:{theme.GOLD}",
            "preview.curve": f"fg:{theme.ACCENT_TOP}",
            "preview.mark": f"fg:{theme.GOLD} bold",
            "preview.hint": f"fg:{theme.DIM}",
            "q.msg": f"fg:{theme.FG} bold",
            "q.mark": f"fg:{theme.ACCENT_TOP} bold",
            "q.sel": f"fg:{theme.ACCENT_TOP} bold",
            "q.opt": f"fg:{theme.FG}",
            "q.check": f"fg:{theme.OK} bold",
            "q.input": f"fg:{theme.GOLD}",
            "q.err": f"fg:{theme.ACCENT_TOP}",
            "help.h": f"fg:{theme.ACCENT} bold",
            "help.k": f"fg:{theme.GOLD} bold",
            "help.t": f"fg:{theme.FG}",
        })

    def _rows(self):
        try:
            return self.app.output.get_size().rows
        except Exception:
            return 40

    def _cols(self):
        try:
            return self.app.output.get_size().columns
        except Exception:
            return 96

    PREVIEW_H = 7   # braille rows

    def _preview_frags(self):
        W = max(20, min(self._cols() - 4, 110))
        return braille_fragments(self.bridge.preview, W, self.PREVIEW_H, {
            "curve": "class:preview.curve",
            "mark": "class:preview.mark",
            "hint": "class:preview.hint",
        })

    # =====================================================================
    # welcome
    # =====================================================================
    def _title_screen(self, bottom_control, bottom_h):
        """Centred logo + subtitle with a swappable bottom block.

        Shared by the welcome prompt and the on-screen directory field so that
        pressing ⏎ keeps the user on the same titled screen.
        """
        content_h = len(theme.LOGO_LINES) + 1 + 3 + 1 + bottom_h

        def top_pad():
            return max(0, (self._rows() - content_h) // 2)

        return HSplit([
            Window(height=top_pad),
            Window(FormattedTextControl(theme.logo_fragments),
                   height=len(theme.LOGO_LINES), align=WindowAlign.CENTER),
            Window(height=1),
            Window(FormattedTextControl([
                ("class:subtitle", "FTIR spectral analysis tool"), ("", "\n"),
                ("class:author", f"{self.author}   ·   v{self.version_display}"), ("", "\n"),
                ("class:author", "University of Glasgow"),
            ]), height=3, align=WindowAlign.CENTER),
            Window(height=1),
            Window(FormattedTextControl(bottom_control), height=bottom_h,
                   align=WindowAlign.CENTER),
            Window(),
        ])

    def _welcome(self):
        return self._title_screen([
            ("class:press", "Press "), ("class:press.key", "⏎"),
            ("class:press", " to begin"), ("", "\n"),
            ("class:press.key", "h"), ("class:author", " help     "),
            ("class:press.key", "q"), ("class:author", " quit"),
        ], 2)

    def _askdir_frags(self):
        frags = [
            ("class:q.mark", "? "),
            ("class:q.msg", "Path to the folder with the spectra:"), ("", "\n\n"),
            ("class:q.input", self.input_buffer), ("class:q.sel", "▏"), ("", "\n"),
        ]
        if self.input_error:
            frags += [("", "\n"), ("class:q.err", self.input_error)]
        else:
            frags += [("", "\n"), ("class:press", "⏎"), ("class:author", " start     "),
                      ("class:press", "esc"), ("class:author", " back")]
        return frags

    def _askdir(self):
        return self._title_screen(self._askdir_frags, 5)

    # =====================================================================
    # hub
    # =====================================================================
    def _header_frags(self, subtitle):
        return [
            ("class:header", "  BAD BLOOD "),
            ("class:header.sep", "▏ "),
            ("class:header.ver", subtitle),
        ]

    def _hub_body(self):
        done = {"prepare": self.n_prepared is not None}
        summary = {
            "prepare": f"  ✓ {self.n_prepared} spectra" if self.n_prepared is not None else "",
        }
        loaded = f"  {self.n_loaded} spectra loaded" if self.n_loaded is not None else "  no spectra loaded"
        frags = [("", "\n"), ("class:item.hint", loaded), ("", "\n\n")]
        for i, (key, label, hint) in enumerate(ITEMS):
            sel = (i == self.index)
            cls = "class:item.sel" if sel else "class:item"
            if key in done:
                badge = ("class:badge.done", "[✓] ") if done[key] else ("class:badge.todo", "[ ] ")
            else:
                badge = ("class:badge.todo", "    ")
            frags += [(cls, "  ▸ " if sel else "    "), badge, (cls, f"{label:<17}")]
            if sel:
                frags.append(("class:item.hint", f"  {hint}"))
            elif summary.get(key):
                frags.append(("class:badge.done", summary[key]))
            frags.append(("", "\n"))
            if key == "export":
                frags.append(("", "\n"))
        return frags

    def _status_hub(self):
        if self.status_msg:
            return [("class:status.msg", "  " + self.status_msg)]
        out = [("class:status", "  ")]
        for k, v in [("↑↓/jk", "move"), ("⏎", "select"), ("q", "quit")]:
            out += [("class:status.key", k), ("class:status", f" {v}    ")]
        return out

    def _hub(self):
        return HSplit([
            Window(FormattedTextControl(lambda: self._header_frags("FTIR spectral pipeline")),
                   height=1),
            Window(height=1, char="─", style="class:rule"),
            Window(FormattedTextControl(self._hub_body), height=len(ITEMS) + 5),
            Window(),
            Window(FormattedTextControl(self._status_hub), height=1),
        ])

    # =====================================================================
    # running (drives an engine handler + renders its prompts)
    # =====================================================================
    def _console_view(self):
        """Return (start, end, total, avail) for the scrolled console window.

        `avail` is the console window's real height: total rows minus every fixed
        region in _running (banner, step label, rules, optional progress/preview,
        the active prompt and the status bar). Matching it exactly means the
        newest lines are never clipped behind the prompt and the scroll bounds
        are correct.

        scroll_offset counts lines scrolled up from the bottom (0 = follow the
        latest output). It is clamped here so it can never run past the top.
        """
        overhead = len(theme.BANNER_LINES) + 1 + 1 + 1   # banner + label + rule + status
        if self.bridge.progress:
            overhead += 1
        if self.bridge.preview is not None:
            has_marks = bool(self.bridge.preview.get("markers"))
            overhead += 1 + (1 + self.PREVIEW_H + 1 + (1 if has_marks else 0))
        if self.bridge.has_prompt():
            overhead += 1 + self._prompt_height()
        avail = max(1, self._rows() - overhead)
        total = len(self.bridge.console)
        self.scroll_offset = max(0, min(self.scroll_offset, max(0, total - avail)))
        end = total - self.scroll_offset
        start = max(0, end - avail)
        return start, end, total, avail

    def _console_frags(self):
        start, end, total, avail = self._console_view()
        frags = []
        for ln in self.bridge.console[start:end]:
            # Section rules (── TITLE ───) get the accent-red header colour.
            cls = "class:console.section" if ln.lstrip().startswith("──") else "class:console"
            frags.append((cls, "  " + ln + "\n"))
        return frags

    def _prompt_frags(self):
        p = self.bridge.prompt
        frags = [("class:q.mark", "  ? "), ("class:q.msg", p["message"]), ("", "\n\n")]
        t = p["type"]
        if t in ("select", "confirm", "checkbox"):
            choices = self._current_choices()
            for i, c in enumerate(choices):
                sel = (i == self.choice_index)
                pointer = "  ❯ " if sel else "    "
                cls = "class:q.sel" if sel else "class:q.opt"
                mark = ""
                if t == "checkbox":
                    mark = "[✓] " if i in self.checkbox_selected else "[ ] "
                frags += [(cls, pointer), ("class:q.check" if (t == "checkbox" and i in self.checkbox_selected) else cls, mark), (cls, str(c) + "\n")]
        else:  # text / path
            shown = self.input_buffer
            frags += [("class:q.opt", "    "), ("class:q.input", shown), ("class:q.sel", "▏"), ("", "\n")]
            if self.input_error:
                frags += [("", "\n"), ("class:q.err", "    " + self.input_error + "\n")]
        return frags

    def _current_choices(self):
        p = self.bridge.prompt
        if p["type"] == "confirm":
            return ["Yes", "No"]
        return p.get("choices", [])

    def _status_running(self):
        p = self.bridge.prompt
        start, end, total, avail = self._console_view()
        scrollable = total > avail
        if self.finished is not None:
            ok, msg = self.finished
            out = [("class:status.msg" if ok else "class:q.err", "  " + msg),
                   ("class:status", "    "),
                   ("class:status.key", "⏎"), ("class:status", " menu")]
            if scrollable:
                out += [("class:status", "    "), ("class:status.key", "^U/^D"),
                        ("class:status", " scroll")]
            return out
        if p is None:
            out = [("class:progress", "  working…")]
            if scrollable:
                out += [("class:status", "     "), ("class:status.key", "^U/^D"),
                        ("class:status", " scroll")]
            return out
        out = [("class:status", "  ")]
        t = p["type"]
        if t in ("select", "confirm", "checkbox"):
            pairs = [("↑↓", "move")]
            if t == "checkbox":
                pairs.append(("space", "toggle"))
            pairs += [("⏎", "confirm"), ("esc", "cancel")]
        else:
            pairs = [("type", "edit"), ("⏎", "submit"), ("esc", "cancel")]
        if scrollable:
            pairs.append(("^U/^D", "scroll"))
        for k, v in pairs:
            out += [("class:status.key", k), ("class:status", f" {v}    ")]
        if start > 0:
            out += [("class:preview.hint", f"▲ {start} more above")]
        return out

    def _running(self):
        self._begin_prompt_if_new()          # sync widget state to new prompts
        children = [
            Window(FormattedTextControl(theme.banner_fragments),
                   height=len(theme.BANNER_LINES), align=WindowAlign.CENTER),
            Window(FormattedTextControl(
                lambda: [("class:header.ver", STEP_LABEL.get(self.active_step, ""))]),
                height=1, align=WindowAlign.CENTER),
            Window(height=1, char="─", style="class:rule"),
            Window(FormattedTextControl(self._console_frags)),   # greedy console
        ]
        if self.bridge.progress:
            children.append(Window(FormattedTextControl(
                lambda: [("class:progress", "  " + (self.bridge.progress or ""))]), height=1))
        if self.bridge.preview is not None:
            has_marks = bool(self.bridge.preview.get("markers"))
            ph = 1 + self.PREVIEW_H + 1 + (1 if has_marks else 0)
            children.append(Window(height=1, char="─", style="class:rule"))
            children.append(Window(FormattedTextControl(self._preview_frags), height=ph))
        if self.bridge.has_prompt():
            children.append(Window(height=1, char="─", style="class:rule"))
            children.append(Window(FormattedTextControl(self._prompt_frags),
                                   height=self._prompt_height()))
        children.append(Window(FormattedTextControl(self._status_running), height=1))
        return HSplit(children)

    def _prompt_height(self):
        p = self.bridge.prompt
        if p is None:
            return 1
        if p["type"] in ("select", "confirm", "checkbox"):
            return len(self._current_choices()) + 3
        return 5 if self.input_error else 3

    # =====================================================================
    # help
    # =====================================================================
    def _help(self):
        rows = [
            ("class:help.h", "  Workflow"), ("", "\n\n"),
            ("class:help.k", "  check   "), ("class:help.t", "Audit file names & date formats (optional, first)."), ("", "\n"),
            ("class:help.k", "  load    "), ("class:help.t", "Scan .mzz files → unified matrix with metadata."), ("", "\n"),
            ("class:help.k", "  prepare "), ("class:help.t", "Quality filters: duplicates, background, atmosphere, low signal."), ("", "\n"),
            ("class:help.k", "  export  "), ("class:help.t", "Pick wavenumbers, filter metadata, write parquet."), ("", "\n\n"),
            ("class:help.t", "  Press ⏎ on the welcome screen to run the whole flow:"), ("", "\n"),
            ("class:help.t", "  check → load → prepare → export, asking the folder once."), ("", "\n\n"),
            ("class:help.t", "  Afterwards a menu lets you re-prepare or re-export the"), ("", "\n"),
            ("class:help.t", "  spectra already in memory without reloading."), ("", "\n"),
        ]
        return HSplit([
            Window(FormattedTextControl(lambda: self._header_frags("Help")), height=1),
            Window(height=1, char="─", style="class:rule"),
            Window(FormattedTextControl(lambda: [("", "\n")] + rows)),
            Window(),
            Window(FormattedTextControl(
                [("class:status.key", "  ←/esc"), ("class:status", " back    "),
                 ("class:status.key", "q"), ("class:status", " quit")]), height=1),
        ])

    def _root(self):
        return {
            "welcome": self._welcome,
            "askdir": self._askdir,
            "hub": self._hub,
            "running": self._running,
            "help": self._help,
        }[self.state]()

    # =====================================================================
    # step execution
    # =====================================================================
    def start_step(self, key):
        if key == "help":
            self._prev_state = self.state
            self.state = "help"
            return
        if key == "quit":
            self.app.exit()
            return
        if key == "newrun":
            key = "pipeline"
        if key == "prepare" and self.df is None:
            self.status_msg = "No spectra in memory — run a new analysis."
            return
        if key == "export" and self.prepared_df is None:
            self.status_msg = "Re-prepare the spectra first."
            return
        self.active_step = key
        self.status_msg = ""
        self.scroll_offset = 0
        self.finished = None
        self.bridge.reset_console()
        self.state = "running"
        t = threading.Thread(target=self._run_worker, args=(key,), daemon=True)
        t.start()

    def _run_worker(self, key):
        import sys
        old_out, old_err = sys.stdout, sys.stderr
        stream = ConsoleStream(self.bridge)
        sys.stdout = stream
        sys.stderr = stream
        self.bridge.running = True
        ok = True
        try:
            summary = runner.STEP_RUNNERS[key](self)
        except Exception as e:  # noqa: BLE001 — surface engine errors, don't crash UI
            import traceback
            traceback.print_exc()        # still captured: lands in the console panel
            summary = f"Error in {key}: {e}"
            ok = False
        finally:
            sys.stdout = old_out
            sys.stderr = old_err
            self.bridge.running = False
        self.status_msg = summary
        # Hold the running screen (log included) until the user presses ⏎, so a
        # fast step — or a crash — cannot wipe itself off the screen unread.
        self.finished = (ok, summary)
        try:
            self.app.invalidate()
        except Exception:
            pass

    # =====================================================================
    # prompt answering (UI thread)
    # =====================================================================
    def _begin_prompt_if_new(self):
        """Sync local widget state to a freshly-arrived prompt."""
        p = self.bridge.prompt
        if p is None:
            return
        seq = self.bridge.prompt_seq
        if getattr(self, "_seen_prompt", None) == seq:
            return
        self._seen_prompt = seq
        self.scroll_offset = 0          # jump back to the bottom for a fresh prompt
        self.choice_index = 0
        self.checkbox_selected = set()
        self.input_error = None
        self.input_buffer = p.get("default", "") or ""
        if p["type"] == "confirm":
            self.choice_index = 0 if p.get("default", True) else 1

    def _answer(self):
        p = self.bridge.prompt
        if p is None:
            return
        t = p["type"]
        if t == "confirm":
            self.bridge.respond(self.choice_index == 0)
        elif t == "select":
            self.bridge.respond(self._current_choices()[self.choice_index])
        elif t == "checkbox":
            chosen = [c for i, c in enumerate(p["choices"]) if i in self.checkbox_selected]
            self.bridge.respond(chosen)
        else:  # text / path
            val = self.input_buffer
            validate = p.get("validate")
            if validate:
                res = validate(val)
                if res is not True:
                    self.input_error = res if isinstance(res, str) else "Invalid input."
                    return
            self.bridge.respond(val)

    # =====================================================================
    # key bindings
    # =====================================================================
    def _build_keys(self):
        kb = KeyBindings()

        welcome = Condition(lambda: self.state == "welcome")
        askdir = Condition(lambda: self.state == "askdir")
        hub = Condition(lambda: self.state == "hub")
        helpmode = Condition(lambda: self.state == "help")
        choosing = Condition(lambda: self.state == "running" and self.bridge.prompt
                             and self.bridge.prompt["type"] in ("confirm", "select", "checkbox"))
        typing = Condition(lambda: self.state == "running" and self.bridge.prompt
                           and self.bridge.prompt["type"] in ("text", "path"))
        prompting = Condition(lambda: self.state == "running" and self.bridge.has_prompt())
        running = Condition(lambda: self.state == "running")
        finished = Condition(lambda: self.state == "running" and self.finished is not None)

        @kb.add("c-c")
        def _(event):
            event.app.exit()

        # -- console scrollback (works alongside any active prompt) --
        # Ctrl+U / Ctrl+D are used (not PageUp) because many terminals capture
        # PageUp for their own scrollback and never deliver it to the app.
        def _page():
            return max(1, (self._rows() - 12) // 2)

        @kb.add("c-u", filter=running)
        @kb.add("pageup", filter=running)
        def _(event):
            self.scroll_offset += _page()

        @kb.add("c-d", filter=running)
        @kb.add("pagedown", filter=running)
        def _(event):
            self.scroll_offset = max(0, self.scroll_offset - _page())

        @kb.add("q", filter=welcome | hub | helpmode)
        def _(event):
            event.app.exit()

        # -- finished step: the log stays on screen until dismissed --
        @kb.add("enter", filter=finished)
        @kb.add("escape", filter=finished)
        @kb.add("space", filter=finished)
        def _(event):
            self.finished = None
            self.scroll_offset = 0
            self.state = "hub"

        @kb.add("enter", filter=welcome)
        def _(event):
            self.status_msg = ""
            self.input_buffer = ""
            self.input_error = None
            self.state = "askdir"          # stay on the titled screen, ask the folder

        @kb.add("h", filter=welcome)
        def _(event):
            self._prev_state = "welcome"
            self.state = "help"

        # -- directory field on the welcome screen --
        @kb.add("enter", filter=askdir)
        def _(event):
            res = runner.valid_dir(self.input_buffer)
            if res is not True:
                self.input_error = res
                return
            self.pipeline_dir = runner.clean_dir(self.input_buffer)
            self.start_step("pipeline")

        @kb.add("backspace", filter=askdir)
        def _(event):
            self.input_error = None
            self.input_buffer = self.input_buffer[:-1]

        @kb.add("escape", filter=askdir)
        def _(event):
            self.state = "welcome"

        @kb.add(Keys.Any, filter=askdir)
        def _(event):
            data = event.data
            if data and data.isprintable():
                self.input_error = None
                self.input_buffer += data

        @kb.add("enter", filter=hub)
        def _(event):
            self.start_step(ITEMS[self.index][0])

        @kb.add("up", filter=hub)
        @kb.add("k", filter=hub)
        def _(event):
            self.status_msg = ""
            self.index = (self.index - 1) % len(ITEMS)

        @kb.add("down", filter=hub)
        @kb.add("j", filter=hub)
        def _(event):
            self.status_msg = ""
            self.index = (self.index + 1) % len(ITEMS)

        @kb.add("left", filter=helpmode)
        @kb.add("escape", filter=helpmode)
        def _(event):
            self.state = self._prev_state

        # -- choosing prompts (confirm/select/checkbox) --
        @kb.add("up", filter=choosing)
        @kb.add("k", filter=choosing)
        def _(event):
            self._begin_prompt_if_new()
            n = len(self._current_choices())
            self.choice_index = (self.choice_index - 1) % n

        @kb.add("down", filter=choosing)
        @kb.add("j", filter=choosing)
        def _(event):
            self._begin_prompt_if_new()
            n = len(self._current_choices())
            self.choice_index = (self.choice_index + 1) % n

        @kb.add("space", filter=choosing)
        def _(event):
            if self.bridge.prompt["type"] == "checkbox":
                if self.choice_index in self.checkbox_selected:
                    self.checkbox_selected.discard(self.choice_index)
                else:
                    self.checkbox_selected.add(self.choice_index)

        @kb.add("enter", filter=choosing)
        def _(event):
            self._answer()

        # -- text / path prompts --
        @kb.add("enter", filter=typing)
        def _(event):
            self._answer()

        @kb.add("backspace", filter=typing)
        def _(event):
            self.input_error = None
            self.input_buffer = self.input_buffer[:-1]

        @kb.add(Keys.Any, filter=typing)
        def _(event):
            data = event.data
            if data and data.isprintable():
                self.input_error = None
                self.input_buffer += data

        @kb.add("escape", filter=prompting)
        def _(event):
            self.bridge.respond(None)

        return kb

    def run(self):
        self.app.run()


def run():
    runner.inject_shim()
    BadBloodTUI().run()


if __name__ == "__main__":
    run()
