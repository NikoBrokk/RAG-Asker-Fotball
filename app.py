"""
Streamlit-app for Asker Fotball sin RAG-demo.

- Bygger en TF-IDF-indeks fra .md-filer i `kb/` første gang (eller ved behov)
- Lar brukeren stille spørsmål og får et kort, kildebasert svar + topp-k kilder
"""

from __future__ import annotations

import os
from pathlib import Path
import streamlit as st

# Lokale moduler
from src.answer import answer, _expand_query
from src.ingest import build_index, OPENAI_API_KEY

# ---------- Konfig & utils ----------

def _env_flag(name: str, default: bool = False) -> bool:
    """Hent bool fra env eller st.secrets."""
    v = os.getenv(name)
    if v is None and name in st.secrets:
        v = str(st.secrets[name])
    if v is None:
        return default
    return str(v).strip().lower() in {"1", "true", "yes", "on"}

USE_OPENAI = _env_flag("USE_OPENAI", False)  # vi bruker TF-IDF nå, men flagget kan stå
CHAT_MODEL = os.getenv("CHAT_MODEL", st.secrets.get("CHAT_MODEL", "tf-idf"))
DATA_DIR = Path(os.getenv("DATA_DIR", st.secrets.get("DATA_DIR", "data")))
KB_DIR = os.getenv("KB_DIR", st.secrets.get("KB_DIR", "kb"))
DEBUG_UI = _env_flag("DEBUG_UI", False)


def ensure_index() -> None:
    """Bygg indeksen første gang eller når filer mangler."""
    vec = DATA_DIR / "vectors.npy"
    meta = DATA_DIR / "meta.jsonl"

    # Hvis noen vil skru på OpenAI senere, gi tidlig beskjed om nøkkel mangler
    if USE_OPENAI and not OPENAI_API_KEY:
        st.error(
            "OPENAI_API_KEY mangler. Legg den inn som secret eller miljøvariabel, "
            "eller sett USE_OPENAI=0 for å bruke TF-IDF."
        )
        st.stop()

    if not vec.exists() or not meta.exists():
        st.info("Indeks ikke funnet – bygger nå …")
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        try:
            build_index(KB_DIR)
        except FileNotFoundError as e:
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

# Statuslinje
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
        "- Kontakt og praktisk info"
    )

# Debug-panel (valgfritt)
if DEBUG_UI:
    st.sidebar.header("Debug")
    st.sidebar.write("KB_DIR:", KB_DIR)
    st.sidebar.write("DATA_DIR:", str(DATA_DIR))
    if q:
        st.sidebar.write("Expanded query:", _expand_query(q))
