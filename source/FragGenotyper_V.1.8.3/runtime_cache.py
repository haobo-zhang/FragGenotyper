"""Remember the caller's font-cache setting before PyInstaller changes it.

Custom runtime hooks run before PyInstaller's matplotlib runtime hook. Startup
can restore this value after that hook creates its per-process temporary cache.
Keep this hook independent of Qt and all scientific/plotting dependencies.
"""
import os
import sys


sys._fraggenotyper_original_mplconfigdir = os.environ.get("MPLCONFIGDIR")
