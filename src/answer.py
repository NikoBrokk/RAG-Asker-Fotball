"""
Henter svar fra en TF-IDF- eller OpenAI-indeks for Asker Fotball.

Returnerer:
- text: kort, ekstraktivt eller generativt svar basert på de beste treffene
- hits: liste av {id, source, score, ...} for transparens i UI

Denne implementasjonen bygger på RAG‑Asker‑Tennis og benytter synonymer,
dokumenttyper og en enkel reranking for å gi bedre svar. Hvis USE_OPENAI er
aktivert og en gyldig OpenAI API-nøkkel er tilgjengelig, genereres svaret med
ChatGPT; ellers brukes en ekstraktiv tilnærming basert på første setning av
beste treff.
"""

from __future__ import annotations

import os
import re
from typing import Dict, List, Tuple, Set

from src.utils import env_flag
from src.retrieve import search

USE_OPENAI: bool = env_flag("USE_OPENAI", False)
CHAT_MODEL: str = os.getenv("CHAT_MODEL", "gpt-4o-mini")

# Initialiser OpenAI-klient dersom aktivert
_openai = None
if USE_OPENAI:
    try:
        from openai import OpenAI  # type: ignore
        api_key = os.getenv("OPENAI_API_KEY")
        project = os.getenv("OPENAI_PROJECT")
        if api_key:
            _openai = OpenAI(api_key=api_key, project=project or None)
        else:
            _openai = OpenAI()  # stol på globale config
    except Exception:
        _openai = None

# Systemprompt brukt for generativ modus
SYSTEM_PROMPT = (
    "Du er en vennlig og hjelpsom assistent for Asker Fotball.\n"
    "Svar kort (1–3 setninger) på norsk bokmål, med egne ord. "
    "Hvis kildene ikke dekker spørsmålet, si 'Jeg vet ikke'."
)

# Synonymer for utvidet søk i fotballdomenet
SYN: Dict[str, List[str]] = {
    "billett": [
        "billett", "billetter", "sesongkort", "sesong-kort", "sesongabonnement",
        "foyka+", "foyka plus", "pris", "priser", "kostnad", "inngang", "adgang"
    ],
    "kamp": [
        "kamp", "kamper", "terminliste", "kampdag", "kampdager", "avspark",
        "match", "program", "kampstart"
    ],
    "parkering": [
        "parkering", "parkere", "p-plass", "p-plasser", "parkeringsplass",
        "easypark", "bil"
    ],
    "stadion": [
        "stadion", "arena", "føyka", "foyka", "anlegg", "tribune", "stadio",
        "fotballhuset"
    ],
    "medlemskap": [
        "medlemskap", "medlem", "kontingent", "medlemskontingent",
        "innmelding", "bli medlem"
    ],
    "kontakt": [
        "kontakt", "telefon", "tlf", "mail", "e-post", "email", "adresse", "epost"
    ],
    "åpningstider": [
        "åpningstider", "åpner", "åpent", "stengt", "åpningstid"
    ],
    "sponsor": [
        "sponsor", "sponsorer", "partner", "partnere", "marked", "bedriftsnettverk"
    ],
    "samfunn": [
        "samfunn", "gatelag", "asker united", "community", "sammen for fotball",
        "aktiviteter"
    ],
    "historie": [
        "historie", "historisk", "grunnlagt", "stiftet", "rekord", "legender",
        "fakta"
    ],
    "lag": [
        "lag", "spillere", "spillertropp", "trener", "keeper", "forsvar",
        "midtbane", "angrep", "a-lag"
    ],
    "marked": [
        "marked", "partner", "sponsor", "sponsorer", "nettverk", "synlighet"
    ],
    "aktivitet": [
        "aktivitet", "akademi", "camp", "kurs", "leir", "trening", "lek"
    ],
}

# Kart over dokumenttyper til triggere for foretrukne kategorier
DOC_HINTS: Dict[str, List[str]] = {
    "billett": SYN["billett"],
    "terminliste": SYN["kamp"],
    "kontakt": SYN["kontakt"],
    "samfunn": SYN["samfunn"],
    "historie": SYN["historie"],
    "stadion": SYN["stadion"],
    "lag": SYN["lag"],
    "marked": SYN["marked"],
    "aktivitet": SYN["aktivitet"],
}


def _expand_query(q: str) -> Tuple[str, Set[str], List[str]]:
    """
    Fjern støy (klubbnavn), identifiser foretrukne dokumenttyper basert på triggere,
    og bygg en utvidet spørring med synonymer.

    Returnerer (expanded_query, preferred_doc_types, extra_terms).
    """
    ql = q.lower()
    # Fjern klubbnavn for å unngå bias
    ql = re.sub(r"\basker fotball\b|\basker fk\b|\bføyka\b", " ", ql)
    ql = ql.strip()

    extra: List[str] = []
    preferred: Set[str] = set()

    # Legg til doc hints basert på triggere
    for dt, triggers in DOC_HINTS.items():
        if any(t in ql for t in triggers):
            preferred.add(dt)
    # Legg til utvidede søkeord basert på synonymlistene
    for key, words in SYN.items():
        if any(t in ql for t in words):
            extra += words
    expanded = q if not extra else q + " " + " ".join(sorted(set(extra)))
    return expanded, preferred, sorted(set(extra))


def _first_sentence(txt: str) -> str:
    """Returner første komplette setning (slutter med punktum, utrop eller spørsmålstegn)"""
    txt = re.sub(r"\s+", " ", (txt or "").strip())
    m = re.search(r"(.+?[.!?])\s", txt + " ")
    return (m.group(1) if m else txt)[:280]


def _extractive(hits: List[Dict]) -> str:
    """
    En enkel ekstraktiv strategi: bruk første setning fra første treff.
    """
    if not hits:
        return "Jeg vet ikke"
    return _first_sentence(hits[0].get("text", "")) or "Jeg vet ikke"


def _score(h: Dict, keys: List[str], preferred: Set[str]) -> float:
    """
    Beregn en heuristisk score for et dokument basert på:
    - Basisscore fra søket (cosinus-similaritet)
    - Bonus hvis dokumenttypen er i preferred
    - Bonus hvis søkeordene forekommer i teksten
    """
    base = float(h.get("score", 0.0))
    bonus = 0.15 if h.get("doc_type") in preferred else 0.0
    txt = (h.get("text") or "").lower()
    bonus += min(0.10, 0.02 * sum(1 for t in keys if t in txt))
    return base + bonus


def _rerank(hits: List[Dict], preferred: Set[str], keys: List[str], k: int, min_score: float = 0.15) -> List[Dict]:
    """
    Rerank treff basert på `_score` og filtrer bort lave scores.
    """
    scored = [(h, _score(h, keys, preferred)) for h in hits]
    scored.sort(key=lambda x: x[1], reverse=True)
    good = [h for h, s in scored if s >= min_score]
    return good[:k] if good else []


def _llm(q: str, hits: List[Dict]) -> str:
    """
    Generer et svar med OpenAI Chat API dersom `_openai` er tilgjengelig.
    Bruker de fem beste dokumentbitene som kontekst sammen med tidligere meldinger.
    """
    if _openai is None:
        return _extractive(hits)
    # Bygg kontekst av topp 5 utdrag
    ctx = "\n\n".join(f"Utdrag {i+1}:\n{h.get('text','')}" for i, h in enumerate(hits[:5]))
    # Hent historikk fra Streamlit session_state hvis tilgjengelig
    history_msgs = []
    try:
        import streamlit as st  # type: ignore
        for prev in st.session_state.get("chat_items", [])[-3:]:
            uq = prev.get("user", "")
            ua = prev.get("answer", "")
            if uq:
                history_msgs.append({"role": "user", "content": uq})
            if ua:
                history_msgs.append({"role": "assistant", "content": ua})
    except Exception:
        history_msgs = []
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages += history_msgs
    messages.append({
        "role": "user",
        "content": f"Spørsmål: {q}\n\nKontekst:\n{ctx}\n\nInstruks: Svar med egne ord i 1–3 setninger."
    })
    try:
        r = _openai.chat.completions.create(
            model=CHAT_MODEL,
            messages=messages,
            temperature=0.2,
            max_tokens=150
        )
        return (r.choices[0].message.content or "").strip()
    except Exception:
        return _extractive(hits)


def answer(q: str, k: int = 6) -> Tuple[str, List[Dict]]:
    """
    Hovedfunksjon brukt av Streamlit-appen.
    Returnerer et kort svar og en liste med treff (kilder).
    """
    qx, preferred, keys = _expand_query(q)
    raw_hits = search(qx, max(k * 2, 6))
    hits = _rerank(raw_hits, preferred, keys, k)
    if not hits:
        # Ingen gode treff – returner rå treff for transparens
        return "Jeg vet ikke", raw_hits[:k]
    out = _llm(q, hits) if USE_OPENAI and _openai is not None else _extractive(hits)
    if not out or len(out.split()) < 2:
        out = "Jeg vet ikke"
    return out, hits
