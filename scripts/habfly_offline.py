"""Run HabFly CLI with Python network connects and Sheets initialization denied.

Usage: .venv/bin/python scripts/habfly_offline.py knowledge validate
This is a reproducibility tripwire, not an OS-level sandbox for untrusted code.
"""

import runpy
import socket
import sys


def deny(*args, **kwargs):
    raise RuntimeError("Offline run attempted network or Google Sheets access")


socket.socket.connect = deny
socket.socket.connect_ex = deny
socket.create_connection = deny

from habfly import spreadsheet

spreadsheet.SpreadsheetAdapter.__init__ = deny
sys.argv = ["habfly", *sys.argv[1:]]
runpy.run_module("habfly", run_name="__main__")
