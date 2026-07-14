"""Verify the TUI's real vertical layout with a genuine pty + pyte.

An in-process Vt100_Output over StringIO lies about vertical distribution; a real
pty driven by pyte does not.  Usage:

    python -m bad_blood_pkg.tui._ptycap [keys]

`keys` is a string of tokens separated by commas, e.g. "enter,down,down".
It boots the app in a child pty, feeds the keys, and prints the rendered screen.
"""
import os
import pty
import select
import sys
import time

import pyte

COLS, ROWS = 96, 40

KEYMAP = {
    "enter": "\r",
    "up": "\x1b[A",
    "down": "\x1b[B",
    "left": "\x1b[D",
    "right": "\x1b[C",
    "pageup": "\x1b[5~",
    "pagedown": "\x1b[6~",
    "c-u": "\x15",
    "c-d": "\x04",
    "backspace": "\x7f",
    "space": " ",
    "comma": ",",
    "esc": "\x1b",
    "q": "q",
    "j": "j",
    "k": "k",
}


def _pump(fd, screen, stream, seconds):
    end = time.time() + seconds
    while time.time() < end:
        r, _, _ = select.select([fd], [], [], 0.05)
        if r:
            try:
                data = os.read(fd, 65536)
            except OSError:
                break
            if not data:
                break
            stream.feed(data.decode("utf-8", "replace"))


def main():
    keys = sys.argv[1].split(",") if len(sys.argv) > 1 and sys.argv[1] else []

    screen = pyte.Screen(COLS, ROWS)
    stream = pyte.ByteStream(screen) if hasattr(pyte, "ByteStream") else pyte.Stream(screen)
    stream = pyte.Stream(screen)

    pid, fd = pty.fork()
    if pid == 0:  # child
        os.environ["TERM"] = "xterm-256color"
        os.environ["COLUMNS"] = str(COLS)
        os.environ["LINES"] = str(ROWS)
        cmd = os.environ.get("PTY_CMD",
                             "from bad_blood_pkg.tui.app import run; run()")
        os.execvp(sys.executable, [sys.executable, "-c", cmd])
    else:  # parent
        import fcntl
        import struct
        import termios
        fcntl.ioctl(fd, termios.TIOCSWINSZ, struct.pack("HHHH", ROWS, COLS, 0, 0))
        _pump(fd, screen, stream, 1.2)
        delay = float(os.environ.get("KEY_DELAY", "0.5"))
        for k in keys:
            try:
                os.write(fd, KEYMAP.get(k, k).encode())
            except OSError:
                break                       # child died; stop feeding keys
            _pump(fd, screen, stream, delay)
        _pump(fd, screen, stream, float(os.environ.get("PTY_WAIT", "0.8")))
        # Capture the LIVE frame: hard-kill the child so its full-screen exit
        # repaint cannot overwrite what we are about to read.
        import signal
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass
        try:
            os.close(fd)
        except OSError:
            pass

    lines = screen.display
    print("┌" + "─" * COLS + "┐")
    for i, line in enumerate(lines):
        print(f"{i:2d}│{line}│")
    print("└" + "─" * COLS + "┘")


if __name__ == "__main__":
    main()
