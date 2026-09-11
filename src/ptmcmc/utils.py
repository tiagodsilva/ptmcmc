import base64
import io
import os
import sys

import matplotlib.pyplot as plt


def is_kitty():
    term = os.getenv("TERM", "")
    supports_graphics = term.startswith("xterm-kitty") or term.startswith(
        "xterm-ghostty"
    )
    return supports_graphics


def show_kitty():
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    data = base64.b64encode(buf.getvalue())
    sys.stdout.buffer.write(b"\x1b_Gf=100,a=T,m=0;" + data + b"\x1b\\")
    sys.stdout.flush()
