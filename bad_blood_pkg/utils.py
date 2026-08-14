import re
import sys
import time
from typing import Iterable, Optional


# =========================================================================
# Age tags in file names
# =========================================================================
# An age tag is a number followed by a unit letter: D (days), W (weeks),
# M (months) or Y (years) — e.g. 03D, 6W, 12M, 2Y.
AGE_UNIT_NAMES = {'D': 'days', 'W': 'weeks', 'M': 'months', 'Y': 'years'}
AGE_UNIT_LETTERS = ''.join(AGE_UNIT_NAMES)

# Searched inside a whole file name.  The tag must not be glued to another
# letter or digit, so units like 50MG or 10MM are not mistaken for ages.
AGE_TAG_RE = re.compile(rf'\d{{1,3}}[{AGE_UNIT_LETTERS}](?![A-Za-z0-9])')
# Matched against a single name part (one metadata value).
AGE_VALUE_RE = re.compile(rf'^(\d{{1,3}})([{AGE_UNIT_LETTERS}])$')

# Example tags, used in the on-screen explanations.
AGE_TAG_EXAMPLES = '03D, 6W, 12M, 2Y'


def age_unit(values: Iterable[str]) -> Optional[str]:
    """Return the unit letter shared by every age tag in *values*.

    Returns None when the values mix units (e.g. 03D and 02W), so callers can
    fall back to a neutral wording/name.
    """
    units = set()
    for value in values:
        match = AGE_VALUE_RE.match(str(value))
        if match:
            units.add(match.group(2))
    return units.pop() if len(units) == 1 else None


def spinner(stop_event):
    """Function to display an animated spinner."""
    symbols = ['-', '\\', '|', '/']
    while not stop_event.is_set():
        for symbol in symbols:
            sys.stdout.write(f"\rProcessing... {symbol}")
            sys.stdout.flush()
            time.sleep(0.1)
    sys.stdout.write("\rProcessing... Done!  \n\n")
    sys.stdout.flush()