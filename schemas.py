"""Shared Pydantic models for SimpleTranslate API."""

from typing import Literal

from pydantic import BaseModel

TranslationDirection = Literal["en2fr", "fr2en"]


class TranslateRequest(BaseModel):
    text: str
    direction: TranslationDirection
    temperature: float | None = None


class TranslateResponse(BaseModel):
    translation: str
