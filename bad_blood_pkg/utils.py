import sys
import time

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