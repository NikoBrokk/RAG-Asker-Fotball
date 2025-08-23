# Arkitektur for RAG Asker Fotball

Dette dokumentet beskriver de viktigste komponentene i prosjektet og
hvordan de samspiller for å levere et Retrieval‑Augmented Generation
(RAG) system for Asker Fotball.

## Oversikt

Prosjektet består av tre hovedfaser:

1. **Ingest** – leser alle dokumenter i `kb/` (markdown) samt eventuelt
   forhåndsprosesserte JSONL‑filer i `data/processed/`. Teksten deles i
   overlappende biter og lagres i en metadatafil (`meta.jsonl`). Det
   bygges i tillegg en vektormatrise (`vectors.npy`) enten via
   TF‑IDF (lokal modell) eller OpenAI‑embeddings, avhengig av
   miljøvariabelen `USE_OPENAI`. Hvis `faiss` er tilgjengelig, lagres
   også en flat indekseringsfil (`index.faiss`).

2. **Retrieve** – last inn vektormatrise og metadata på første kall.
   Ved søk genereres en spørringsvektor og det beregnes en
   likhetsscore mot alle dokumentbiter. De mest relevante bitene
   returneres med tilhørende metadata. Dokumenttypene klassifiseres
   heuristisk (billett, terminliste, kontakt, samfunn osv.) for å gi
   hint i re‑rangeringen.

3. **Answer** – mottar et bruker‑spørsmål, utvider det med synonymer
   for å fange opp flere dokumenter, henter de relevante bitene via
   retrieve, og formulerer et kort svar. Hvis `USE_OPENAI=1` og
   API‑nøkkel er satt, brukes OpenAI Chat API til å generere
   svaret; ellers returneres et ekstraktivt svar basert på det beste
   utdraget.

## Filstruktur

* `app.py` – Streamlit‑grensesnitt som håndterer brukerinput, bygger
  indeksen ved behov og viser svar og kilder.
* `src/ingest.py` – bygger vektorindeks og metadata fra kildefilene.
* `src/retrieve.py` – tilbyr søk i indeksen med TF‑IDF eller OpenAI.
* `src/answer.py` – utvider spørsmålet, re‑rangerer treff og
  genererer svar.
* `src/utils.py` – felles hjelpere for fillesing, tekstdeling og
  miljøvariabelhåndtering.
* `kb/` – katalogen der markdown‑filene som utgjør kunnskapsbasen
  plasseres. Hver fil beskriver et av klubbens tilbud eller
  informasjonsområder.

## Hvordan legge til nye kilder

* Legg en ny `.md`‑fil i `kb/` med tittel på første linje (`# …`).
* Kjør `python scripts/build_index.py` for å bygge indeksen på nytt.
* Start (eller restart) appen med `streamlit run app.py`. Den vil
  automatisk laste den nye indeksen.

## Miljøvariabler

Flere aspekter kan konfigureres via miljøvariabler eller
`streamlit.secrets`. De viktigste er:

* `USE_OPENAI` – sett til `1` for å bruke OpenAI‑embeddings og Chat,
  ellers brukes TF‑IDF.
* `OPENAI_API_KEY` – API‑nøkkel for OpenAI. Påkrevd hvis `USE_OPENAI=1`.
* `DATA_DIR` – katalog der indeksen lagres (standard `data`).
* `KB_DIR` – katalog der markdown‑kildene ligger (standard `kb`).
* `CHAT_MODEL` – navnet på chatmodellen som brukes med OpenAI (f.eks.
  `gpt-4o-mini`).

## Testing

En enkel test (`tests/test_retrieve.py`) sikrer at indeksen er
bygget og at vektorfilene har korrekt format. Kjør `pytest` for å
utføre testene.