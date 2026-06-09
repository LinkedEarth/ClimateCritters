"""Pre-render script: copy notebooks from repo root into docs/notebooks/.

Quarto's project root is docs/, so it cannot resolve paths that escape upward
via ../. This script mirrors the top-level notebooks/ tree into docs/notebooks/
before each render so _quarto.yml can reference them with plain relative paths.

Only notebooks listed in the INCLUDE sets below are copied; add new ones here
as they are created.
"""

import shutil
from pathlib import Path

# Paths relative to this script's location (docs/scripts/)
DOCS_DIR = Path(__file__).parent.parent          # docs/
REPO_ROOT = DOCS_DIR.parent                      # PaleoBeasts/
SRC = REPO_ROOT / "notebooks"
DST = DOCS_DIR / "notebooks"

# Subdirectories to mirror
SUBDIRS = [
    "model_demos",
    "functionality_demos",
    "base_classes"
]

def sync():
    # SUBDIRS = [x.name for x in SRC.iterdir() if x.is_dir()]

    for subdir in SUBDIRS:
        src_dir = SRC / subdir
        dst_dir = DST / subdir
        if not src_dir.exists():
            continue
        dst_dir.mkdir(parents=True, exist_ok=True)
        for nb in sorted(src_dir.glob("*.ipynb")) + sorted(src_dir.glob("*.qmd")):
            dst = dst_dir / nb.name
            if not dst.exists() or nb.stat().st_mtime > dst.stat().st_mtime:
                shutil.copy2(nb, dst)
                print(f"  synced {nb.relative_to(REPO_ROOT)} → docs/notebooks/{subdir}/{nb.name}")

if __name__ == "__main__":
    print("sync_notebooks: copying notebooks into docs/notebooks/")
    sync()
    print("sync_notebooks: done")
