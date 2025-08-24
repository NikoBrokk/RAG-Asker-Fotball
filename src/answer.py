"""
answer.py
----------
Henter svar fra en TF-IDF-indeks bygd av ingest.py.

Returnerer:
- text: kort, ekstraktivt svar basert på de beste treffene
- hits: liste av {id, source, score} for transparens i UI
"""

from __future__ import annotations

import os
import json
import re
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
import joblib
from sklearn.metrics.pairwise import cosine_similarity


# --- Stier (samme som i ingest.py)
DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
VECTORS_PATH = DATA_DIR / "vectors.npy"
META_PATH = DATA_DIR / "meta.jsonl"
VECTORIZER_PATH = DATA_DIR / "vectorizer.pkl"


# --- Enkle globale caches for ytelse
_V: Optional[np.ndarray] = None
_META: Optional[List[Dict]] = None
_VECTORIZER = None


def _load_index() -> Tuple[np.ndarray, List[Dict], object]:
    """Laster (og cacher) TF-IDF-matrisen, metadata og vectorizer."""
    global _V, _META, _VECTORIZER

    if _V is None or _META is None or _VECTORIZER is None:
        if not VECTORS_PATH.exists() or not META_PATH.exists() or not VECTORIZER_PATH.exists():
            raise FileNotFoundError(
                "Indeks mangler. Kjør først bygging (build_index) i ingest.py."
            )

        _V = np.load(VECTORS_PATH)
        _META = [json.loads(line) for line in META_PATH.read_text(encoding="utf-8").splitlines()]
        _VECTORIZER = joblib.load(VECTORIZER_PATH)

        # Robusthet: sørg for samsvar
        if _V.shape[0] != len(_META):
            raise RuntimeError(
                f"Ulikt antall rader i vectors ({_V.shape[0]}) og meta ({len(_META)}). "
                "Bygg indeksen på nytt."
            )

    return _V, _META, _VECTORIZER


def _sentences(text: str) -> List[str]:
    """Splitt tekst i (enkle) setninger."""
    text = re.sub(r"\s+", " ", text.strip())
    if not text:
        return []
    # Del på punktum, utrop, spørsmål – behold skilletegn
    parts = re.split(r"(?<=[\.\!\?])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def _summarize_snippets(text: str, query: str, max_chars: int = 420) -> str:
    """
    Velg 1–3 setninger fra dokumentet som matcher spørringen.
    Hvis ingen match: ta de første setningene.
    """
    sents = _sentences(text)
    if not sents:
        return ""

    q_terms = [t for t in re.findall(r"\w+", query.lower()) if len(t) > 2]
    picks: List[str] = []

    # Score setninger grovt på overlap
    def score_sent(s: str) -> int:
        s_low = s.lower()
        return sum(1 for t in q_terms if t in s_low)

    ranked = sorted(sents, key=score_sent, reverse=True)
    for s in ranked[:4]:
        if score_sent(s) > 0:
            picks.append(s)
        if len(" ".join(picks)) > max_chars:
            break

    if not picks:
        picks = sents[:2]

    out = " ".join(picks)
    if len(out) > max_chars:
        out = out[:max_chars].rsplit(" ", 1)[0] + "…"
    return out


def _expand_query(q: str) -> str:
    """
    Litt enkel synonym-utvidelse for domenet (kan bygges ut senere).
    Returnerer en utvidet streng som kan gi bedre recall i TF-IDF.
    """
    SYNS = {
        "billetter": ["billett", "billetter", "sesongkort", "pris", "priser", "ticket", "inngang"],
        "kamp": ["kamp", "kamper", "terminliste", "kampstart", "avspark", "match"],
        "parkering": ["parkering", "parkere", "p-plass"],
        "stadion": ["stadion", "arena", "føyka", "føykA", "anlegg"],
        "medlemskap": ["medlemskap", "medlem", "kontingent"],
        "kontakt": ["kontakt", "telefon", "epost", "e-post", "mail"],
        "åpningstider": ["åpningstider", "åpent", "åpner"],
        "sponsor": ["sponsor", "partnere", "bedriftsnettverk", "marked"],
    }
    q_low = q.lower()
    extra: List[str] = []
    for base, alts in SYNS.items():
        if base in q_low:
            extra.extend(alts)
    if extra:
        return q + " " + " ".join(sorted(set(extra)))
    return q


def _retrieve(query: str, k: int = 5) -> Tuple[List[int], np.ndarray]:
    """Returnerer indeksene til topp-k dokumenter og tilsvarende score."""
    V, _, vectorizer = _load_index()
    q_expanded = _expand_query(query)
    qv = vectorizer.transform([q_expanded]).toarray()  # (1, d)
    sims = cosine_similarity(qv, V)[0]                 # (n_docs,)
    order = np.argsort(-sims)[:k]
    return order.tolist(), sims


def answer(query: str, k: int = 5) -> Tuple[str, List[Dict]]:
    """
    Hovedfunksjon brukt av app.py
    - query: brukerens spørsmål
    - k: hvor mange dokumenter som returneres i hits
    """
    try:
        order, sims = _retrieve(query, k=max(k, 5))
        _, meta, _ = _load_index()
    except FileNotFoundError as e:
        return (
            "Indeksen er ikke bygget ennå. Bygg først (scripts/build_index.py) eller "
            "la appen bygge automatisk ved oppstart.",
            [],
        )
    except Exception as e:
        return (f"Noe gikk galt ved søk: {e}", [])

    # Lag kort svar ved å kombinere 2–3 beste snippets
    snippets: List[str] = []
    for i in order[:3]:
        txt = meta[i].get("text", "")
        snip = _summarize_snippets(txt, query, max_chars=280)
        if snip:
            snippets.append(snip)

    text = " ".join(snippets).strip()
    if not text:
        text = "Fant ingen gode treff i kunnskapsbasen."

    # Kilder til UI
    hits: List[Dict] = []
    for i in order[:k]:
        hits.append(
            {
                "id": meta[i].get("id", f"doc-{i}"),
                "source": meta[i].get("source", "?"),
                "score": float(sims[i]),
            }
        )

    return text, hits


# Eksporter _expand_query slik app.py kan importere den
__all__ = ["answer", "_expand_query"]
