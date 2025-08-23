"""
Generator av svar for Asker Fotball.

Denne modulen tar et bruker‑spørsmål, utvider det med synonymer for bedre
søk, henter relevante biter via `src.retrieve.search`, og genererer et kort
svar. Dersom `USE_OPENAI=1` og en gyldig OpenAI‑nøkkel er tilgjengelig,
brukes OpenAI Chat API til å formulere svaret. Ellers returneres den
første setningen av det best rangerte utdraget.

Basert på RAG‑Asker‑Tennis, men tilpasset Asker Fotball.
"""

from __future__ import annotations

import os
import re
from typing import Dict, List, Tuple, Set

from src.utils import env_flag
from src.retrieve import search

USE_OPENAI: bool = env_flag("USE_OPENAI", False)
CHAT_MODEL: str = os.getenv("CHAT_MODEL", "gpt-4o-mini")

_openai = None
if USE_OPENAI:
    try:
        from openai import OpenAI  # type: ignore
        api_key = os.getenv("OPENAI_API_KEY")
        project = os.getenv("OPENAI_PROJECT")
        if api_key:
            _openai = OpenAI(api_key=api_key, project=project or None)
    except Exception:
        _openai = None

# Systemprompt til LLM: svar i 1–3 setninger på norsk bokmål.
SYSTEM_PROMPT: str = (
    "Du er en vennlig og hjelpsom assistent for Asker Fotball. "
    "Svar kort (1–3 setninger) på norsk bokmål, med egne ord. "
    "Hvis kildene ikke dekker spørsmålet, si 'Jeg vet ikke'."
)

# Synonymer for å utvide søket. Hver nøkkel representerer en dokumenttype
# eller tema, med triggere (verdier) som kan forekomme i brukerens spørsmål.
SYN: Dict[str, List[str]] = {
    # Billetter, sesongkort, priser
    "billett": [
        "billett", "billetter", "sesongkort", "foyka+", "foyka plus",
        "pris", "priser", "kostnad", "inngang", "adgang", "kampbillett"
    ],
    # Kampoversikt, terminliste og resultater
    "terminliste": [
        "terminliste", "kamp", "kamper", "kampdag", "resultat", "resultater",
        "tabell", "serie", "postnord", "på stillingen", "ligatabell"
    ],
    # Kontaktinformasjon og adresser
    "kontakt": [
        "kontakt", "telefon", "tlf", "mail", "e-post", "epost", "adresse",
        "postadresse", "kirkeveien", "postboks"
    ],
    # Samfunn og sosiale prosjekter
    "samfunn": [
        "samfunn", "gatelag", "asker united", "hæppe", "brobygger", "brobyggercup",
        "samfunnslag", "aktivt lokalsamfunn", "sammen for fotball", "sosialt"
    ],
    # Historie og fakta
    "historie": [
        "historie", "historisk", "fakta", "stiftet", "grunnlagt", "rekord",
        "adelskalender", "spillere", "topp", "legender"
    ],
    # Stadion og fasiliteter
    "stadion": [
        "stadion", "føyka", "foyka", "fotballhuset", "tribune", "kapasitet",
        "parkering", "vip", "medie", "anlegg", "fasiliteter", "sitteplasser"
    ],
    # Lag og spillere
    "lag": [
        "a-lag", "spillere", "spiller", "keeper", "forsvar", "midtbane",
        "angrep", "trener", "treners", "trenere", "spillertropp", "lag",
        "spillere", "personell", "støtteapparat"
    ],
    # Partner- og sponsorinformasjon
    "marked": [
        "marked", "partner", "partnere", "sponsor", "sponsorer", "synlighet",
        "nettverk", "bedrift", "sponsoravtale", "bedriftsavtale"
    ],
    # Aktiviteter som akademi, camps og treningstilbud
    "aktivitet": [
        "akademi", "obos", "camp", "camps", "trening", "aktivitet", "kurs",
        "tilbud", "leir", "campen", "oboscamp"
    ],
    # Generell info / annet
    "info": [
        "klubb", "foreningen", "asker", "fotballklubb", "asker fotball", "postnord-ligaen",
        "kontor", "administrasjon"
    ],
}

# Dokument‑hint mapping: når noen av trigger‑ordene er i spørsmålet,
# foretrekk dokumenter av denne typen. Nøkkelen samsvarer med doc_type i
# retrieve.
DOC_HINTS: Dict[str, List[str]] = {
    "billett": SYN["billett"],
    "terminliste": SYN["terminliste"],
    "kontakt": SYN["kontakt"],
    "samfunn": SYN["samfunn"],
    "historie": SYN["historie"],
    "stadion": SYN["stadion"],
    "lag": SYN["lag"],
    "marked": SYN["marked"],
    "aktivitet": SYN["aktivitet"],
    "info": SYN["info"],
}


def _expand_query(q: str) -> Tuple[str, Set[str], List[str]]:
    """
    Utvid brukerens spørsmål med synonymer for å bedre treff i indeksen.

    Returnerer tuple (expanded, preferred_doc_types, extra_terms).
    """
    ql = q.lower()
    # Fjern klubbnavnet fra spørringen slik at det ikke påvirker søket
    ql = re.sub(r"\basker fotballklubb\b|\basker fotball\b|\basker fk\b", " ", ql)
    ql = ql.strip()
    extra: List[str] = []
    preferred: Set[str] = set()
    # Legg til doc hints basert på triggere
    for dt, triggers in DOC_HINTS.items():
        if any(t in ql for t in triggers):
            preferred.add(dt)
    # Legg til utvidede søkeord (synonymer) basert på synonymlistene
    for key, words in SYN.items():
        if any(t in ql for t in words):
            extra += words
    # Bygg utvidet spørring
    expanded = q if not extra else q + " " + " ".join(sorted(set(extra)))
    return expanded, preferred, sorted(set(extra))


def _first_sentence(txt: str) -> str:
    """Hent første setning fra et tekstutdrag, maksimalt 280 tegn."""
    txt = re.sub(r"\s+", " ", (txt or "").strip())
    m = re.search(r"(.+?[.!?])\s", txt + " ")
    return (m.group(1) if m else txt)[:280]


def _extractive(hits: List[Dict]) -> str:
    """Hvis ingen LLM, returner første setning av beste treff."""
    if not hits:
        return "Jeg vet ikke"
    return _first_sentence(hits[0].get("text", "")) or "Jeg vet ikke"


def _score(h: Dict, keys: List[str], preferred: Set[str]) -> float:
    """
    Rangér treff. Basispoeng fra TF‑IDF/OpenAI supplementeres med bonus for
    foretrukket doc_type og for treff av nøkkelord i teksten.
    """
    base = float(h.get("score", 0.0))
    bonus = 0.15 if h.get("doc_type") in preferred else 0.0
    txt = (h.get("text") or "").lower()
    # Bonus for hvert nøkkelord som faktisk forekommer i dokumentteksten
    bonus += min(0.10, 0.02 * sum(1 for t in keys if t in txt))
    return base + bonus


def _rerank(hits: List[Dict], preferred: Set[str], keys: List[str], k: int, min_score: float = 0.15) -> List[Dict]:
    """Re-ranger treff basert på _score og filtrer ut de under terskel."""
    scored = [(h, _score(h, keys, preferred)) for h in hits]
    scored.sort(key=lambda x: x[1], reverse=True)
    good = [h for h, s in scored if s >= min_score]
    return good[:k] if good else []


def _llm(q: str, hits: List[Dict]) -> str:
    """Bruk LLM til å formulere et svar gitt kontekst fra treff og samtale."""
    if _openai is None:
        return _extractive(hits)
    # Bygg kontekst av topp 5 utdrag
    ctx = "\n\n".join(
        f"Utdrag {i + 1}:\n{h.get('text', '')}"
        for i, h in enumerate(hits[:5])
    )
    # Inkluder tidligere meldinger for kontekst (lagret i Streamlit session_state)
    history_msgs: List[Dict[str, str]] = []
    try:
        import streamlit as st  # type: ignore
        for prev_q, prev_a in st.session_state.get("history", [])[-3:]:  # siste 3 meldinger
            # Fjern prefix "Spørsmål:" fra lagret historikkinnlegg
            uq = prev_q.replace("**Spørsmål:**", "Spørsmål:").strip()
            history_msgs.append({"role": "user", "content": uq})
            history_msgs.append({"role": "assistant", "content": prev_a})
    except Exception:
        history_msgs = []
    # Bygg meldingsliste til OpenAI chat-komplettering
    messages: List[Dict[str, str]] = []
    messages.append({"role": "system", "content": SYSTEM_PROMPT})
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
            max_tokens=120
        )
        return (r.choices[0].message.content or "").strip()
    except Exception:
        # Fallback til ekstraktivt svar hvis LLM feiler
        return _extractive(hits)


def answer(q: str, k: int = 6) -> Tuple[str, List[Dict]]:
    """
    Hovedfunksjon for å besvare brukerens spørsmål.

    Utvider spørringen med synonymer, henter raw treff, re-rangerer dem
    og genererer et svar. Returnerer en tuple (svartekst, listen over
    rangerte treff).
    """
    qx, preferred, keys = _expand_query(q)
    raw_hits = search(qx, max(k * 2, 6))
    hits = _rerank(raw_hits, preferred, keys, k)
    if not hits:
        return "Jeg vet ikke", raw_hits[:k]
    out = _llm(q, hits) if USE_OPENAI and _openai is not None else _extractive(hits)
    if not out or len(out.split()) < 2:
        out = "Jeg vet ikke"
    return out, hits