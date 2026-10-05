"""Records (cleaning/validation) and RAG document chunking. Pure Python, no database."""

import pytest
from pydantic import ValidationError

from app.documents import CHUNK_SIZE, movie_header, profile_chunks, review_chunks
from app.records import MovieRecord, ReviewRecord, clean_text


def make_movie(**overrides) -> MovieRecord:
    fields = dict(
        id="tt1392214", title="Prisoners", year=2013, director="Denis Villeneuve", rating=8.2,
        genres=["Drama", "Thriller", "Crime"], description="A father takes matters into his own hands.",
        cast=["Hugh Jackman", "Jake Gyllenhaal"], keywords=["kidnapping", "maze"], imdb_votes=912002,
        rt_slug="prisoners_2013", rt_match="slug+year",
    )
    return MovieRecord(**{**fields, **overrides})


def make_review(**overrides) -> ReviewRecord:
    fields = dict(
        id=1822, movie_id="tt1392214", text="Dark, slow-burning and extremely tense.", rating=8.75,
        critic="Some Critic", publication="Some Paper", original_score="3.5/4", sentiment="fresh",
    )
    return ReviewRecord(**{**fields, **overrides})


def test_clean_text_collapses_whitespace():
    assert clean_text("  Dark,\n\n slow   burn ") == "Dark, slow burn"
    assert clean_text("   ") is None


@pytest.mark.parametrize("bad", [{"text": "   "}, {"rating": 11}, {"movie_id": "1392214"}, {"sentiment": "meh"}])
def test_review_record_rejects_invalid_rows(bad):
    with pytest.raises(ValidationError):
        make_review(**bad)


def test_review_becomes_one_chunk_with_header_and_citation_metadata():
    movie, review = make_movie(), make_review()

    [chunk] = review_chunks(movie, review)

    assert chunk.content.startswith("Movie: Prisoners\nYear: 2013\nGenres: Drama, Thriller, Crime\n\n")
    assert "Review by Some Critic (Some Paper), 3.5/4, fresh:\nDark, slow-burning" in chunk.content
    assert chunk.movie_id == "tt1392214"
    expected = {
        "doc_type": "review",
        "movie_title": "Prisoners",
        "review_id": 1822,
        "source": "rotten_tomatoes",
        "chunk_key": "review:1822:0",
        "chunk_count": 1,
    }
    assert expected.items() <= chunk.metadata.items()


def test_profile_contains_metadata_facts():
    [chunk] = profile_chunks(make_movie())

    assert "Director: Denis Villeneuve" in chunk.content
    assert "Cast: Hugh Jackman, Jake Gyllenhaal" in chunk.content
    assert "Overview:\nA father takes matters into his own hands." in chunk.content
    assert chunk.metadata["doc_type"] == "profile"
    assert chunk.metadata["review_id"] is None


def test_long_profile_is_split_and_every_chunk_keeps_the_header():
    movie = make_movie(keywords=[f"keyword number {i}" for i in range(150)])  # ~2,700 characters of keywords

    chunks = profile_chunks(movie)

    header = movie_header(movie)
    assert len(chunks) > 1
    for i, chunk in enumerate(chunks):
        assert chunk.content.startswith(header + "\n\n")
        assert len(chunk.content) - len(header) - 2 <= CHUNK_SIZE
        assert chunk.metadata["chunk_key"] == f"profile:tt1392214:{i}"
        assert chunk.metadata["chunk_count"] == len(chunks)
