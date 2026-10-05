"""Clean, validated shapes for ingested data (Phase 4.2–4.3).

Raw rows are converted into these records before anything touches the database. Pydantic does the
checking: a row that fails validation (empty text, rating out of range, malformed ID) is skipped
and counted instead of being stored.
"""

import re
from datetime import date
from typing import Literal

from pydantic import BaseModel, Field, field_validator


def clean_text(value: str | None) -> str | None:
    """Collapse runs of whitespace (including newlines) to one space and trim."""
    if value is None:
        return None
    cleaned = re.sub(r"\s+", " ", str(value)).strip()
    return cleaned or None


class MovieRecord(BaseModel):
    id: str = Field(pattern=r"^tt\d+$")  # IMDb ID
    title: str = Field(min_length=1)
    year: int | None = Field(default=None, ge=1880, le=2100)
    director: str | None = None
    rating: float | None = Field(default=None, ge=0, le=10)  # IMDb average
    genres: list[str] = []
    description: str | None = None
    # Extras that go into the JSONB metadata column
    cast: list[str] = []
    writers: list[str] = []
    keywords: list[str] = []
    runtime_minutes: int | None = Field(default=None, ge=1)
    imdb_votes: int | None = Field(default=None, ge=0)
    original_language: str | None = None
    rt_slug: str
    rt_match: str

    @field_validator("title", "director", "description", mode="before")
    @classmethod
    def _clean(cls, value):
        return clean_text(value)

    def extras(self) -> dict:
        return {
            "cast": self.cast,
            "writers": self.writers,
            "keywords": self.keywords,
            "runtime_minutes": self.runtime_minutes,
            "imdb_votes": self.imdb_votes,
            "original_language": self.original_language,
            "rt_slug": self.rt_slug,
            "rt_match": self.rt_match,
        }


class ReviewRecord(BaseModel):
    id: int = Field(gt=0)  # Rotten Tomatoes review ID
    movie_id: str = Field(pattern=r"^tt\d+$")
    text: str = Field(min_length=1)
    rating: float | None = Field(default=None, ge=0, le=10)  # normalised to 0–10
    source: Literal["rotten_tomatoes"] = "rotten_tomatoes"
    # Extras for the JSONB metadata column
    critic: str | None = None
    publication: str | None = None
    original_score: str | None = None
    sentiment: Literal["fresh", "rotten"]
    top_critic: bool = False
    published: date | None = None
    url: str | None = None

    @field_validator("text", "critic", "publication", mode="before")
    @classmethod
    def _clean(cls, value):
        return clean_text(value)

    def extras(self) -> dict:
        return {
            "critic": self.critic,
            "publication": self.publication,
            "original_score": self.original_score,
            "sentiment": self.sentiment,
            "top_critic": self.top_critic,
            "published": self.published.isoformat() if self.published else None,
            "url": self.url,
        }
