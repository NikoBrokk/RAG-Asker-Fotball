import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.utils import env_flag


def test_env_flag_variant_names(monkeypatch):
    monkeypatch.delenv("USE_OPENAI", raising=False)
    monkeypatch.delenv("use_openai", raising=False)
    monkeypatch.setenv("useopenai", "1")
    assert env_flag("USE_OPENAI") is True
    monkeypatch.setenv("useopenai", "0")
    assert env_flag("USE_OPENAI", True) is False
