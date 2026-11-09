"""Make `import kelvins` work without relying on an editable install.

On macOS, Python >= 3.12 ignores .pth files flagged as hidden, and iCloud-synced folders
can set that flag on the editable-install .pth file.
"""
import sys
from pathlib import Path

SRC = str(Path(__file__).resolve().parents[1] / "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)
