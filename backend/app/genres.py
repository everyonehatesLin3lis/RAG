"""The genres present in our movie data (TMDB genre names), shared by query translation and the tools."""

KNOWN_GENRES = [
    "Action", "Adventure", "Animation", "Comedy", "Crime", "Drama", "Family", "Fantasy", "History",
    "Horror", "Music", "Mystery", "Romance", "Science Fiction", "Thriller", "War", "Western",
]

_LOOKUP = {g.lower(): g for g in KNOWN_GENRES} | {"sci-fi": "Science Fiction", "scifi": "Science Fiction"}


def normalise_genre(value: str) -> str | None:
    """'thriller' -> 'Thriller', 'Sci-Fi' -> 'Science Fiction', anything unknown -> None."""
    return _LOOKUP.get(value.strip().lower())
