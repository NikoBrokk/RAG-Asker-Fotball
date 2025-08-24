"""
Utility‑funksjoner for RAG Asker Fotball.

Inneholder funksjoner for å lese markdown‑filer og dele tekst i enkle biter.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, TYPE_CHECKING
from datetime import datetime

if TYPE_CHECKING:  # pragma: no cover - kun for typehinting
    from huggingface_hub import HfApi


def _read_text_file(p: Path) -> str:
    """Try å lese en fil med UTF‑8, fall tilbake til latin‑1 om nødvendig."""
    try:
        return p.read_text(encoding="utf-8")
    except Exception:
        return p.read_text(encoding="latin-1", errors="ignore")


def read_markdown_files(kb_dir: str) -> List[Dict]:
    """
    Gå gjennom alle markdown- og tekstfiler i `kb_dir` og returner
    en liste med dicts som inneholder tittel, kilde, tekst og versjonsdato.
    """
    root = Path(kb_dir)
    if not root.exists():
        return []
    out: List[Dict] = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in {".md", ".txt"}:
            continue
        text = _read_text_file(p)
        title = p.stem.replace("_", " ").strip() or "Uten tittel"
        version_date = datetime.fromtimestamp(p.stat().st_mtime).date().isoformat()
        out.append({"title": title, "source": str(p), "text": text, "version_date": version_date})
    return out


def simple_chunks(text: str, size: int = 800, overlap: int = 120) -> List[str]:
    """Del en streng i biter på ca. `size` ord, med overlapping."""
    if size <= 0:
        return [text]
    toks = text.split()
    if not toks:
        return []
    out: List[str] = []
    i = 0
    while i < len(toks):
        chunk = " ".join(toks[i : i + size]).strip()
        if chunk:
            out.append(chunk)
        if i + size >= len(toks):
            break
        i += max(1, size - overlap)
    return out


def env_flag(name: str, default: bool = False) -> bool:
    """Les boolsk miljøvariabel case-insensitive.

    "1", "true", "yes" eller "on" blir ``True``. Funksjonen sjekker både
    originalnavn, ``upper`` og ``lower`` varianter for å være robust mot
    små avvik i navngivning av miljøvariabler (f.eks. ``useopenai`` vs
    ``USE_OPENAI``).
    """
    for key in {name, name.upper(), name.lower()}:
        v = os.getenv(key)
        if v is not None:
            return v.strip().lower() in {"1", "true", "yes", "on"}
    return default

def get_hf_api() -> "HfApi":
    """Autentiser og returner en ``HfApi``-klient."""
    token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if not token:
        raise RuntimeError("HUGGINGFACEHUB_API_TOKEN mangler")
    from huggingface_hub import HfApi
    return HfApi(token=token)
