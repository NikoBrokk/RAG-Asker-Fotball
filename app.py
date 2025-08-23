"""
Streamlit‑app for Asker Fotballs RAG‑demo.

Starter en liten chatbot som henter svar fra en TF‑IDF eller OpenAI‑basert
indeks bygget fra filene i `kb/`.
"""

import os
from pathlib import Path
import streamlit as st

# --- Lokale moduler
from src.answer import answer, _expand_query  # noqa: F401, used via answer
from src.ingest import build_index, OPENAI_API_KEY


def _env_flag(name: str, default: bool = False) -> bool:
    """Hent en boolsk miljøvariabel eller Streamlit‑secret."""
    v = os.getenv(name)
    if v is None and name in st.secrets:
        v = str(st.secrets[name])
    if v is None:
        return default
    return str(v).strip().lower() in {"1", "true", "yes", "on"}


# --- Konfig
USE_OPENAI = _env_flag("USE_OPENAI", False)
CHAT_MODEL = os.getenv(
    "CHAT_MODEL",
    st.secrets.get("CHAT_MODEL", "gpt-4o-mini" if USE_OPENAI else "tf-idf"),
)
DATA_DIR = Path(os.getenv("DATA_DIR", st.secrets.get("DATA_DIR", "data")))
KB_DIR = os.getenv("KB_DIR", st.secrets.get("KB_DIR", "kb"))
DEBUG_UI = _env_flag("DEBUG_UI", False)


def ensure_index() -> None:
    """Bygg indeksen første gang eller når den mangler på disk."""
    vec = DATA_DIR / "vectors.npy"
    meta = DATA_DIR / "meta.jsonl"

    if USE_OPENAI and not OPENAI_API_KEY:
        st.error(
            "Kan ikke bygge indeks – OPENAI_API_KEY mangler. "
            "Legg den i .env eller i Streamlit Secrets."
        )
        st.stop()

    if not vec.exists() or not meta.exists():
        st.info("Indeks mangler – bygger nå …")
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        build_index(KB_DIR)


# --- UI
st.set_page_config(page_title="Chatbot Asker Fotball", page_icon="", layout="centered")
ensure_index()

if "history" not in st.session_state:
    st.session_state.history = []
if "q" not in st.session_state:
    st.session_state.q = ""

st.title("Chatbot Asker Fotball")
st.caption("Få svar på ofte stilte spørsmål om klubben, billetter, kamper m.m.")

mode_label = "**OpenAI**" if USE_OPENAI else "**TF‑IDF**"
st.caption(f"Status: indeks `ok` • Modus: {mode_label} (modell: {CHAT_MODEL})")

# Input + knapper
st.session_state.q = st.text_input(
    "Skriv spørsmålet ditt:",
    value=st.session_state.q,
    placeholder="F.eks. Hva koster et sesongkort?",
    key="q_input",
)
col1, col2 = st.columns([1, 1])
svar_btn = col1.button("Svar")
reset_btn = col2.button("Start ny samtale")

if reset_btn:
    # Tøm historikk og input, deretter restart
    st.session_state.history = []
    st.session_state.q = ""
    st.rerun()

if svar_btn:
    q = (st.session_state.q or "").strip()
    if not q:
        st.warning("Skriv et spørsmål først.")
    else:
        with st.spinner("Henter svar …"):
            text, hits = answer(q, k=6)
            st.session_state.history.append(("**Spørsmål:** " + q, text))

        st.markdown("### Svar")
        # vis hele samtalen
        for i, (u, a) in enumerate(st.session_state.history, start=1):
            st.write(u)
            st.write(a)
            if i < len(st.session_state.history):
                st.markdown("---")

        # vis kilder (ikke-klikkbare)
        st.markdown("### Kilder")
        if not hits:
            st.write("Ingen kilder.")
        else:
            for h in hits:
                src = h.get("source", "?")
                hid = h.get("id", "?")
                sc = h.get("score", "")
                try:
                    sc = f"{float(sc):.3f}"
                except Exception:
                    sc = str(sc)
                if isinstance(src, str) and src.startswith("http"):
                    st.markdown(f"- `{src}` — `{hid}` (score {sc})")
                else:
                    st.markdown(f"- **{src}** — `{hid}` (score {sc})")

# Hjelpetekst nederst
with st.expander("Hva kan jeg spørre om?"):
    st.markdown(
        "- Billetter og sesongkort\n"
        "- Terminliste og kamper\n"
        "- Medlemskap og aktiviteter\n"
        "- Kontaktinfo, stadion og parkering\n"
        "- Klubbinformasjon (historie, lag, samfunn)"
    )