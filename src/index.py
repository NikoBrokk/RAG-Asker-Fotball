"""
FAISS‑indeksbygger for Asker Fotball.

Denne modulen sørger for å laste vektorer og metadata fra disk og å bygge
en FAISS‑indeks dersom den mangler. Basert på RAG‑Asker‑Tennis.
"""

from __future__ import annotations

import json
from pathlib import Path

import faiss  # type: ignore
import numpy as np

from src.ingest import build_index

# Artefaktstier
INDEX_PATH = Path("data/index.faiss")
VEC_PATH = Path("data/vectors.npy")
META_PATH = Path("data/meta.jsonl")


def _ensure_artifacts() -> None:
    """
    Sørger for at vectors/meta/index finnes. Hvis ikke, bygges de fra kb/.
    """
    missing = [p for p in [VEC_PATH, META_PATH, INDEX_PATH] if not p.exists()]
    if missing:
        print(
            f"[index] Mangler artefakter: {', '.join(str(p) for p in missing)} – bygger…"
        )
        INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
        build_index()


def _load_vectors_raw() -> np.ndarray:
    """
    Last inn vektorfilen fra disk med allow_pickle=True.

    returnerer en numpy array (eventuelt dtype=object) som ikke er normalisert.
    """
    try:
        X = np.load(VEC_PATH, allow_pickle=True)
    except Exception as e:
        raise RuntimeError(f"Kunne ikke laste vektorfil '{VEC_PATH}': {e}")
    return X


def load_vectors() -> np.ndarray:
    """
    Last inn vektorer, sørg for riktig datatype og L2-normaliser dem.
    Tillater pickled objekt arrays og konverterer dem til float32 hvis mulig.
    """
    _ensure_artifacts()
    X = _load_vectors_raw()
    # Hvis objekt array (kan komme fra eldre numpy versjoner eller varierende dimensjoner)
    if X.dtype == object:
        try:
            # Vektorer er antatt å være 1D-arrays eller lister av float
            X = np.vstack([np.asarray(v, dtype="float32") for v in X])
        except Exception as e:
            raise RuntimeError(
                f"Vektorfil '{VEC_PATH}' inneholder uventet format: {e}. "
                "Kjør ingest.py på nytt for å generere en korrekt vektorfil."
            )
    else:
        # Sørg for riktig dtype
        try:
            X = X.astype("float32")
        except Exception:
            pass
    # Normaliser i stedet for å anta at det allerede er normalisert
    if X.size > 0:
        faiss.normalize_L2(X)
    return X


def build_faiss_index() -> faiss.Index:
    """
    Bygg en FAISS-indeks fra lagrede vektorer. Skriver filen til disk.
    """
    X = load_vectors()
    index = faiss.IndexFlatIP(X.shape[1])
    if X.size > 0:
        index.add(X)
    faiss.write_index(index, str(INDEX_PATH))
    return index


def load_meta():
    """
    Last inn metadatafilen fra disk og returner en liste av dicts.
    """
    _ensure_artifacts()
    with META_PATH.open(encoding="utf-8") as f:
        return [json.loads(l) for l in f]


if __name__ == "__main__":
    build_faiss_index()
    print("Indeks skrevet til", INDEX_PATH)
