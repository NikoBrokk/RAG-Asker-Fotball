"""
Script for å bygge vektorindeksen for Asker Fotball.
Kjør denne før du starter appen hvis du har endret innholdet i `kb/`.
"""

import sys
from pathlib import Path

# Sørg for at prosjektets rot ligger på sys.path slik at `src` kan importeres
PROJ_ROOT = Path(__file__).resolve().parents[1]
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))

from src.ingest import build_index


def main() -> None:
    kb_dir = sys.argv[1] if len(sys.argv) > 1 else "kb"
    build_index(kb_dir)


if __name__ == "__main__":
    main()