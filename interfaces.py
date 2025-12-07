"""Shared Pydantic models for SimpleTranslate API."""

from pydantic import BaseModel


class TranslateRequest(BaseModel):
    text_source: str
    temperature: float | None = None
    beams: int | None = None


class TranslateResponse(BaseModel):
    translation: str
