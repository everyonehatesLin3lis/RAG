"""RAG documents and chunking (Phase 4.5–4.6).

Two kinds of document:
- review:  one per critic review. Answers "what do critics say about X?" and cites a single review.
- profile: one per movie, built from metadata (director, cast, plot, keywords). Answers questions
           reviews rarely cover, like "movies about dreams" or "what is Prisoners about?".

Every document is a short header (title, year, genres) plus a body. Only the body is split; the
header is repeated at the top of every chunk, so each chunk still says which movie it is about
when it is retrieved on its own.

Chunking: a chunk is the unit we embed and retrieve. Too large and one vector blurs several topics
and wastes prompt tokens; too small and it loses context. CHUNK_SIZE is in characters of body text.
CHUNK_OVERLAP repeats the end of one chunk at the start of the next so a sentence cut at a boundary
still appears whole in one of them. Critic reviews here are at most ~360 characters, so a review is
never split; long profiles (many keywords) are.
"""

from dataclasses import dataclass, field

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.records import MovieRecord, ReviewRecord

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

# Tries paragraph breaks first, then lines, then words, so chunks end at natural boundaries.
_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)


@dataclass
class Chunk:
    movie_id: str
    content: str
    metadata: dict = field(default_factory=dict)


def movie_header(movie: MovieRecord) -> str:
    lines = [f"Movie: {movie.title}"]
    if movie.year:
        lines.append(f"Year: {movie.year}")
    if movie.genres:
        lines.append(f"Genres: {', '.join(movie.genres)}")
    return "\n".join(lines)


def review_body(review: ReviewRecord) -> str:
    who = review.critic or "Unknown critic"
    if review.publication:
        who += f" ({review.publication})"
    verdict = ", ".join(v for v in [review.original_score, review.sentiment] if v)
    return f"Review by {who}, {verdict}:\n{review.text}"


def profile_body(movie: MovieRecord) -> str:
    facts = []
    if movie.director:
        facts.append(f"Director: {movie.director}")
    if movie.writers:
        facts.append(f"Writers: {', '.join(movie.writers)}")
    if movie.cast:
        facts.append(f"Cast: {', '.join(movie.cast)}")
    if movie.runtime_minutes:
        facts.append(f"Runtime: {movie.runtime_minutes} minutes")
    if movie.rating is not None and movie.imdb_votes:
        facts.append(f"IMDb rating: {movie.rating} ({movie.imdb_votes:,} votes)")

    paragraphs = ["\n".join(facts)] if facts else []
    if movie.description:
        paragraphs.append(f"Overview:\n{movie.description}")
    if movie.keywords:
        paragraphs.append(f"Keywords: {', '.join(movie.keywords)}")
    return "\n\n".join(paragraphs)


def chunk_document(movie: MovieRecord, body: str, key_prefix: str, metadata: dict) -> list[Chunk]:
    """Split the body, put the movie header on each piece, and attach citation metadata."""
    header = movie_header(movie)
    pieces = _splitter.split_text(body) or [body]
    return [
        Chunk(
            movie_id=movie.id,
            content=f"{header}\n\n{piece}",
            metadata={
                **metadata,
                "movie_id": movie.id,
                "movie_title": movie.title,
                "year": movie.year,
                "chunk_key": f"{key_prefix}:{i}",  # stable across re-ingestion, unlike the row ID
                "chunk_index": i,
                "chunk_count": len(pieces),
            },
        )
        for i, piece in enumerate(pieces)
    ]


def review_chunks(movie: MovieRecord, review: ReviewRecord) -> list[Chunk]:
    metadata = {
        "doc_type": "review",
        "review_id": review.id,
        "source": review.source,
        "critic": review.critic,
        "publication": review.publication,
    }
    return chunk_document(movie, review_body(review), f"review:{review.id}", metadata)


def profile_chunks(movie: MovieRecord) -> list[Chunk]:
    metadata = {"doc_type": "profile", "review_id": None, "source": "tmdb_imdb"}
    return chunk_document(movie, profile_body(movie), f"profile:{movie.id}", metadata)
