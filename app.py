import os, json
from pathlib import Path
from typing import List, Dict
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", None)

def _load_docs(kb_dir: str) -> List[Dict]:
    root = Path(kb_dir)
    paths = sorted(root.rglob("*.md"))
    if not paths:
        raise FileNotFoundError(f"Ingen .md-filer i {root.resolve()}")
    docs = []
    for i, p in enumerate(paths):
        text = p.read_text(encoding="utf-8", errors="ignore")
        docs.append({"id": f"doc-{i}", "source": str(p), "text": text})
    return docs

def build_index(kb_dir: str = "kb") -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    docs = _load_docs(kb_dir)
    corpus = [d["text"] for d in docs]

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=30000,
        lowercase=True,
        token_pattern=r"(?u)\b\w\w+\b",
    )
    X = vectorizer.fit_transform(corpus).astype(np.float32)

    np.save(DATA_DIR / "vectors.npy", X.toarray())
    with (DATA_DIR / "meta.jsonl").open("w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    joblib.dump(vectorizer, DATA_DIR / "vectorizer.pkl")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--kb", default=os.getenv("KB_DIR", "kb"))
    args = ap.parse_args()
    build_index(args.kb)
