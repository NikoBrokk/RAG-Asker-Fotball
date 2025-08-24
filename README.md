---
sdk: streamlit
app_file: app.py
---

# RAG Asker Fotball

Dette prosjektet implementerer en enkel Retrieval‑Augmented Generation (RAG) for
Asker Fotball. Applikasjonen består av tre hoveddeler:

* **Ingest** – leser alle markdown‑filer i `kb/` (og eventuelle
  JSONL‑filer i `data/processed/`) og bygger en TF‑IDF eller OpenAI‑basert
  vektorindeks.
* **Retrieve** – utfører søk i indeksen og returnerer de mest relevante
  dokumentbitene.
* **Answer** – bygger et kort svar basert på de beste treffene. Dersom
  `USE_OPENAI=1` i miljøvariablene brukes OpenAI chat‑modell; ellers brukes
  et ekstraktivt svar basert på TF‑IDF.

## Kom i gang

1. Opprett et virtuelt miljø og installer avhengigheter:

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. Kopier eller legg inn kunnskapsbasen din i `kb/`. I denne mappen skal
   hvert dokument være en egen `.md`-fil. Du kan også lagre prosesserte
   JSONL‑filer i `data/processed/` hvis du har forhåndsprosesserte kilder.

3. Bygg indeksen før første kjøring (eller når du endrer `kb/`):

   ```bash
   python scripts/build_index.py
   ```

4. Start Streamlit‑appen:

   ```bash
   streamlit run app.py
   ```

Som standard benytter appen TF‑IDF som søkemotor. Sett miljøvariabelen
`USE_OPENAI=1` og `OPENAI_API_KEY=<nøkkel>` for å benytte OpenAI‑baserte
embeddings og chat‑generering.
