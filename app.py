"""
Streamlit-app for Asker Fotball sin RAG-demo.

- Bygger en TF-IDF- eller OpenAI-indeks fra .md-filer i `kb/` første gang (eller ved behov)
- Lar brukeren stille spørsmål og får et kort, kildebasert svar + topp-k kilder

Denne versjonen håndterer trygt pickled numpy-arrays (allow_pickle=True) og
støtter generering av svar via OpenAI dersom ``USE_OPENAI`` og en gyldig API-nøkkel
er satt. Hvis ikke benyttes en ekstraktiv strategi.
"""

from __future__ import annotations

import os
from pathlib import Path
import numpy as np
import streamlit as st

# Lokale moduler
from src.answer import answer
from src.ingest import build_index, OPENAI_API_KEY
from src.utils import get_hf_api


# ---------- Konfig & utils ----------

def _env_flag(name: str, default: bool = False) -> bool:
    """Hent bool fra env eller secrets case-insensitive og uten understrek."""
    base = name.replace("_", "")
    variants = {
        name,
        name.upper(),
        name.lower(),
        base,
        base.upper(),
        base.lower(),
    }
    for key in variants:
        v = os.getenv(key)
        if v is None:
            try:
                v = st.secrets.get(key)
            except Exception:
                v = None
        if v is not None:
            return str(v).strip().lower() in {"1", "true", "yes", "on"}
    return default


def _secret(name: str, default=None):
    """Trygg henter for str-secrets."""
    try:
        return st.secrets.get(name, default)
    except Exception:
        return default


USE_OPENAI = _env_flag("USE_OPENAI", bool(OPENAI_API_KEY))
CHAT_MODEL = os.getenv("CHAT_MODEL", _secret("CHAT_MODEL", "tf-idf"))
DATA_DIR = Path(os.getenv("DATA_DIR", _secret("DATA_DIR", "data")))
KB_DIR = os.getenv("KB_DIR", _secret("KB_DIR", "kb"))
DEBUG_UI = _env_flag("DEBUG_UI", False)
HF_SPACE = os.getenv("HF_SPACE", _secret("HF_SPACE"))


hf_space_info = None
if HF_SPACE:
    try:
        hf_space_info = get_hf_api().space_info(HF_SPACE)
    except Exception as e:  # pragma: no cover - kun best effort
        hf_space_info = {"error": str(e)}


def ensure_index() -> None:
    """
    Bygg indeksen første gang eller når filer mangler eller er korrupt.
    Denne funksjonen tillater pickled numpy-arrays og forsøker å konvertere dem
    til en 2D float-array for enkel validering.
    """
    vec = DATA_DIR / "vectors.npy"
    meta = DATA_DIR / "meta.jsonl"
    need_rebuild = False

    # Hvis noen vil skru på OpenAI senere, gi tidlig beskjed om nøkkel mangler
    if USE_OPENAI and not OPENAI_API_KEY:
        st.error(
            "OPENAI_API_KEY mangler. Legg den inn som secret eller miljøvariabel, "
            "eller sett USE_OPENAI=0 for å bruke TF-IDF."
        )
        st.stop()

    if not vec.exists() or not meta.exists():
        need_rebuild = True
    else:
        try:
            X = np.load(vec, allow_pickle=True)
            # Hvis objekt-array: forsøk å stable til en matrise av floats
            if X.dtype == object:
                try:
                    X_tmp = np.vstack([np.asarray(v, dtype="float32") for v in X])
                except Exception:
                    X_tmp = None
                    need_rebuild = True
                X = X_tmp if X_tmp is not None else X
            # Sjekk at matrisen har forventet form
            if X is None or getattr(X, "ndim", 0) != 2 or X.shape[0] == 0:
                need_rebuild = True
        except Exception:
            need_rebuild = True

    if need_rebuild:
        st.info("Indeks ikke funnet eller korrupt – bygger nå …")
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        try:
            build_index(KB_DIR)
        except FileNotFoundError:
            st.error(
                f"Ingen .md-filer funnet i '{KB_DIR}'. "
                "Legg inn kunnskapsfiler i mappen `kb/` (f.eks. `kb/billetter.md`)."
            )
            st.stop()


# ---------- UI ----------

st.set_page_config(page_title="Chatbot Asker Fotball", page_icon="⚽", layout="wide")
st.title("RAG Asker Fotball")
st.caption("Still et spørsmål – få svar med kilder fra klubbens informasjon.")

ensure_index()

mode_label = "**OpenAI**" if USE_OPENAI else "**TF-IDF**"
st.markdown(
    f"Status: indeks `ok` • Modus: {mode_label} • Modell: `{CHAT_MODEL}`  "
)

# Inndata
col_q1, col_q2 = st.columns([4, 1])
with col_q1:
    q = st.text_input(
        "Skriv spørsmålet ditt:",
        placeholder="F.eks. Hva koster sesongkort? Hvor parkerer jeg på kampdag?",
        key="q_input",
    )
with col_q2:
    submit = st.button("Svar", use_container_width=True)

# Reset
if st.button("Start ny samtale"):
    for k in list(st.session_state.keys()):
        if k.startswith("chat_") or k in {"q_input"}:
            del st.session_state[k]
    st.rerun()

# Historikk
if "chat_items" not in st.session_state:
    st.session_state.chat_items = []

# Svarlogikk
if submit:
    q = (q or "").strip()
    if not q:
        st.warning("Skriv et spørsmål først.")
    else:
        with st.spinner("Henter svar …"):
            text, hits = answer(q, k=6)
        st.session_state.chat_items.append(
            {"user": q, "answer": text, "hits": hits}
        )

# Vis samtale
for i, turn in enumerate(st.session_state.chat_items, start=1):
    st.markdown(f"### Spørsmål {i}")
    st.write(turn["user"])
    st.markdown("**Svar**")
    st.write(turn["answer"])
    # Kilder
    st.markdown("**Kilder**")
    hits = turn.get("hits", []) or []
    if not hits:
        st.write("Ingen kilder funnet.")
    else:
        for h in hits:
            src = h.get("source", "?")
            hid = h.get("id", "?")
            sc = h.get("score", 0.0)
            try:
                sc_str = f"{float(sc):.3f}"
            except Exception:
                sc_str = str(sc)
            if isinstance(src, str) and src.startswith("http"):
                st.markdown(f"- `{src}` — `{hid}` (score {sc_str})")
            else:
                st.markdown(f"- **{src}** — `{hid}` (score {sc_str})")
    st.divider()

# Hjelp/eksempler
with st.expander("Hva kan jeg spørre om?"):
    st.markdown(
        "- Billetter og sesongkort\n"
        "- Kampdager, avspark, terminliste\n"
        "- Parkering og stadioninformasjon (Føyka)\n"
        "- Medlemskap, akademi, samfunn\n"
    )
