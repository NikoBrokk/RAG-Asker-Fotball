"""
Søkemotor for Asker Fotball.

Denne modulen gir funksjonalitet for å søke etter relevante biter i
kunnskapsbasen, enten via TF‑IDF eller OpenAI‑embeddings. Dokumentene
leses fra `kb/` og `data/processed/`, deles i biter og indeksene
lastes på første forespørsel.

`search()` returnerer en liste med dicts med felter som `text`,
`source`, `title`, `score`, `doc_type`, `version_date`, `page`,
`chunk_idx` og `id`. Doc_type beregnes heuristisk basert på filnavn
og innhold for å gi et hint om dokumentets type (billett, terminliste,
kontakt, samfunn, historie, stadion, lag, marked, aktivitet eller annet).

Basert på RAG‑Asker‑Tennis, men tilpasset Asker Fotball.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Iterable, Optional

import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from src.utils import env_flag

# --- Konfig ---
# Søk i både kb/ og eventuelt forhåndsprosesserte data under data/processed
KB_DIRS: List[Path] = [Path("kb"), Path("data/processed")]
CHUNK_SIZE = 700
CHUNK_OVERLAP = 120

DATA_DIR: Path = Path(os.getenv("DATA_DIR", "data"))
USE_OPENAI: bool = env_flag("USE_OPENAI", False)
EMBED_MODEL: str = os.getenv("EMBED_MODEL", "text-embedding-3-small")

# OpenAI-klient (brukes kun hvis USE_OPENAI)
_openai = None
if USE_OPENAI:
    try:
        from openai import OpenAI  # type: ignore
        api_key = os.getenv("OPENAI_API_KEY")
        project = os.getenv("OPENAI_PROJECT")
        if api_key:
            _openai = OpenAI(api_key=api_key, project=project or None)
        else:
            _openai = OpenAI()  # fallback til standard config
    except Exception:
        _openai = None

# TF‑IDF state
_VEC: Optional[TfidfVectorizer] = None
_MTX = None  # scipy sparse matrix
_META: List[Dict] = []  # én entry per rad i _MTX

# OpenAI state
_EMB: Optional[np.ndarray] = None  # shape (n_chunks, dim)
_META_OAI: List[Dict] = []


# ---------- Utils ----------


def _read_text_file(p: Path) -> str:
    try:
        return p.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""


def _strip_markdown_noise(txt: str) -> str:
    # Fjern codefences og komprimer whitespace
    txt = re.sub(r"```.*?```", " ", txt, flags=re.S)
    txt = re.sub(r"\s+", " ", txt)
    return txt.strip()


def _title_from_markdown(txt: str, fallback: str) -> str:
    """Hent første overskrift (h1) som tittel, eller bruk fallback."""
    m = re.search(r"^\s*#\s+(.+)$", txt, flags=re.M)
    if m:
        return m.group(1).strip()
    for line in txt.splitlines():
        s = line.strip()
        if s:
            return s[:120]
    return fallback


def _infer_doc_type(name: str, text: str) -> str:
    """
    Grov inndeling av dokumenttyper for rangering. Basert på nøkkelord i filnavn og innhold.

    Avkast "billett", "terminliste", "kontakt", "samfunn", "historie",
    "stadion", "lag", "marked", "aktivitet" eller "annet".
    """
    low = (name + " " + text[:400]).lower()
    # Billett / sesongkort / pris
    if any(w in low for w in ["billett", "billetter", "sesongkort", "foyka+", "foyka plus", "pris", "kostnad", "inngang", "adgang"]):
        return "billett"
    # Terminliste, kamper, resultater, tabell
    if any(w in low for w in ["terminliste", "kamp", "kamper", "resultat", "resultater", "tabell", "serie", "postnord"]):
        return "terminliste"
    # Kontaktinformasjon
    if any(w in low for w in ["kontakt", "telefon", "tlf", "mail", "e-post", "epost", "adresse", "kirkeveien", "postadresse"]):
        return "kontakt"
    # Samfunn / gatelag / united
    if any(w in low for w in ["samfunn", "gatelag", "asker united", "hæppe", "brobygger", "samfunnslag", "aktivt lokalsamfunn", "sammen for fotball"]):
        return "samfunn"
    # Historie og fakta
    if any(w in low for w in ["historie", "historisk", "stiftet", "grunnlagt", "rekord", "adelskalender", "fakta", "spillere", "topp", "legender"]):
        return "historie"
    # Stadion, arena, fasiliteter, parkering
    if any(w in low for w in ["stadion", "føyka", "foyka", "fotballhuset", "tribune", "kapasitet", "parkering", "vip", "medie"]):
        return "stadion"
    # Lag, spillere, trener
    if any(w in low for w in ["a-lag", "spillere", "keeper", "forsvar", "midtbane", "angrep", "trener", "spillertropp", "lag"]):
        return "lag"
    # Marked / sponsor
    if any(w in low for w in ["marked", "partner", "sponsor", "synlighet", "nettverk", "sponsoravtale"]):
        return "marked"
    # Aktivitet / akademi / camp
    if any(w in low for w in ["akademi", "camp", "obos", "trening", "aktivitet", "kurs", "leir"]):
        return "aktivitet"
    return "annet"


def _chunk(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Del tekst i biter med overlappende vinduer for TF‑IDF."""
    text = text.strip()
    if not text:
        return []
    chunks: List[str] = []
    i = 0
    n = len(text)
    while i < n:
        j = min(i + size, n)
        chunk = text[i:j]
        chunks.append(chunk)
        if j == n:
            break
        i = max(j - overlap, 0)
    return chunks


def _iter_kb_files() -> Iterable[Path]:
    """Iterer over alle markdown- og jsonl-filer i kunnskapskatalogene."""
    seen: set[Path] = set()
    for d in KB_DIRS:
        if not d.exists():
            continue
        for p in d.rglob("*.md"):
            if p.is_file():
                seen.add(p.resolve())
        for p in d.rglob("*.jsonl"):
            if p.is_file():
                seen.add(p.resolve())
    for p in sorted(seen):
        yield Path(p)


def _load_corpus() -> List[Dict]:
    """Last hele korpuset og del i biter med metadata."""
    docs: List[Dict] = []
    for p in _iter_kb_files():
        source_path = str(p).replace("\\", "/")
        if p.suffix.lower() == ".md":
            raw = _read_text_file(p)
            clean = _strip_markdown_noise(raw)
            title = _title_from_markdown(raw, p.stem.replace("-", " "))
            doc_type = _infer_doc_type(p.name, clean)
            chunks = _chunk(clean)
            for ci, ch in enumerate(chunks):
                docs.append({
                    "text": ch,
                    "source": source_path,
                    "title": title,
                    "doc_type": doc_type,
                    "version_date": None,
                    "page": None,
                    "chunk_idx": ci,
                    "id": f"{source_path}#{ci}",
                })
        elif p.suffix.lower() == ".jsonl":
            ci = 0
            for line in _read_text_file(p).splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                txt = obj.get("text", "")
                meta = obj.get("metadata", {})
                if not txt.strip():
                    continue
                txt_clean = _strip_markdown_noise(txt)
                title = meta.get("title") or _title_from_markdown(txt, Path(meta.get("source", p.stem)).stem)
                doc_type = meta.get("doc_type") or _infer_doc_type(title, txt)
                src = (meta.get("source") or source_path).replace("\\", "/")
                page = meta.get("page")
                docs.append({
                    "text": txt_clean,
                    "source": src,
                    "title": title,
                    "doc_type": doc_type,
                    "version_date": meta.get("version_date"),
                    "page": page,
                    "chunk_idx": ci,
                    "id": f"{Path(src).as_posix()}#{ci}",
                })
                ci += 1
    return docs


# ---------- Indeksering ----------


def _ensure_index_tfidf() -> None:
    """Lazy bygging av TF‑IDF indeks ved første kall."""
    global _VEC, _MTX, _META
    if _VEC is not None and _MTX is not None and _META:
        return
    corpus = _load_corpus()
    _META = corpus
    texts = [d["text"] for d in corpus]
    if not texts:
        # ingen dokumenter; opprett tom vectorizer for å unngå crash
        _VEC = TfidfVectorizer(ngram_range=(1, 2), max_features=1000)
        _MTX = _VEC.fit_transform([""])
        return
    _VEC = TfidfVectorizer(
        ngram_range=(1, 2),
        max_df=0.95,
        min_df=1,
        strip_accents="unicode",
        lowercase=True,
        norm="l2",
        sublinear_tf=True,
        max_features=60000,
    )
    _MTX = _VEC.fit_transform(texts)


def _ensure_index_openai() -> None:
    """Lazy last OpenAI-indeks fra disk og håndter pickled arrays."""
    global _EMB, _META_OAI
    if _EMB is not None and _META_OAI:
        return
    vec_path = DATA_DIR / "vectors.npy"
    meta_path = DATA_DIR / "meta.jsonl"
    if not vec_path.exists() or not meta_path.exists():
        raise FileNotFoundError(
            "OpenAI-indeks mangler. Kjør ingestion med USE_OPENAI=1 for å generere embeddings."
        )
    # Last vektorfil med allow_pickle og håndter objekt-array
    try:
        arr = np.load(vec_path, allow_pickle=True)
    except Exception as e:
        raise RuntimeError(f"Kunne ikke laste vektorfil '{vec_path}': {e}")
    if arr.dtype == object:
        try:
            arr = np.vstack([np.asarray(v, dtype="float32") for v in arr])
        except Exception as e:
            raise RuntimeError(
                f"Vektorfil '{vec_path}' inneholder uventet format: {e}. "
                "Bygg indeksen på nytt med ingest.py."
            )
    else:
        arr = arr.astype("float32")
    _EMB = arr
    _META_OAI = []
    with meta_path.open("r", encoding="utf-8") as f:
        for line in f:
            _META_OAI.append(json.loads(line))


# ---------- Public API ----------


def search(query: str, k: int = 6) -> List[Dict]:
    """
    Søk etter de k mest relevante dokumentbitene.

    Returnerer en liste med dicts med felter: text, source, title, score,
    doc_type, version_date, page, chunk_idx og id.
    """
    if USE_OPENAI and _openai is not None:
        _ensure_index_openai()
        # Embedd spørringen med OpenAI embeddings
        try:
            r = _openai.embeddings.create(model=EMBED_MODEL, input=query)
        except Exception:
            # Faller tilbake til TF‑IDF hvis embed feiler
            return search_tfidf(query, k)
        qvec = np.array(r.data[0].embedding, dtype="float32")
        qvec = qvec / (np.linalg.norm(qvec) + 1e-12)
        sims = _EMB @ qvec  # type: ignore
        order = np.argsort(-sims)[:k]
        out: List[Dict] = []
        for idx in order:
            m = dict(_META_OAI[idx])
            m["score"] = float(sims[idx])
            out.append(m)
        return out
    # TF‑IDF
    return search_tfidf(query, k)


def search_tfidf(query: str, k: int = 6) -> List[Dict]:
    """Indre funksjon for TF‑IDF-søk, tilgjengelig for fallback."""
    _ensure_index_tfidf()
    if _VEC is None or _MTX is None:
        return []
    qvec = _VEC.transform([query])  # type: ignore
    sims = linear_kernel(qvec, _MTX).ravel()  # type: ignore
    order = np.argsort(-sims)[:k]
    out: List[Dict] = []
    for idx in order:
        m = dict(_META[idx])
        m["score"] = float(sims[idx])
        out.append(m)
    return out
