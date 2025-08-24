import os
import numpy as np


def test_artifacts_exist():
    """Sjekk at indeksartefakter finnes etter bygging og har forventet format."""
    assert os.path.exists("data/meta.jsonl"), "meta.jsonl mangler – kjør build_index"
    assert os.path.exists("data/vectors.npy"), "vectors.npy mangler – kjør build_index"
    assert os.path.exists("data/index.faiss"), "index.faiss mangler – kjør build_index"
    X = np.load("data/vectors.npy")
    assert X.ndim == 2 and X.shape[0] > 0, "Vektorfilen må ha minst én rad"