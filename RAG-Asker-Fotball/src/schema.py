"""
Pydantic-skjemaer for Asker Fotball RAG.

Disse modellene kan brukes til å validere strukturen på objekter som
returneres fra API-et eller til testing. Skjemaene er ikke i aktiv
bruk i applikasjonen, men kan være nyttige for utvidelser.
"""

from __future__ import annotations
from typing import List

from pydantic import BaseModel, Field


class Source(BaseModel):
    """Metainformasjon om et kildeutdrag fra kunnskapsbasen."""
    text: str = Field(..., description="Utdrag av tekst som ble brukt som kilde")
    source: str = Field(..., description="Filnavn eller URL til kilden")
    id: str = Field(..., description="Unik ID for tekstbiten (fil#chunk)")
    score: float = Field(..., description="Relevansscore for utdraget")
    doc_type: str = Field(..., description="Dokumenttype slik den er klassifisert av retrieval")


class Answer(BaseModel):
    """Skjema for et svar fra chatbottjenesten."""
    answer: str = Field(..., description="Selve svaret i naturlig språk")
    sources: List[Source] = Field([], description="Liste over kilder brukt til å generere svaret")