"""Phase 19: build evaluation/evaluation_dataset.jsonl from questions whose expected answers come from the database.

Expected answers are never typed from memory: directors, years, IMDb ratings, "which is higher", critic averages
and acceptable films for recommendations are read from PostgreSQL here, and the build stops if a film is missing or
a checked word does not appear in that film's reviews. So the dataset matches what the system can actually know.

The questions are new: none of the 54 questions used to tune retrieval (Phases 5, 7, 18) are reused, so the
evaluation does not reward settings that were picked on its own questions.

Each line:
  id, type, question, expected_answer, expected_movie (the plan's three fields), plus
  expected_movies   films that should be retrieved (Recall@K); [] for no-answer questions
  must_contain_any  an answer counts as correct if it contains at least one of these (case-insensitive)
  expected_tool     tools a correct answer should use (any of them), or null when none is needed
  acceptable_movies for recommendations: every film that satisfies the request
  no_answer         true when the film is not in the data and the answer must say so

Run from the repo root (backend venv active): python scripts/build_eval_dataset.py
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from sqlalchemy import any_, func, literal, select  # noqa: E402
from sqlalchemy.orm import Session  # noqa: E402

from app.db import get_engine  # noqa: E402
from app.models import Movie, Review  # noqa: E402

OUT = ROOT / "evaluation" / "evaluation_dataset.jsonl"
NO_ANSWER_TEXT = "The film is not in the database, so the answer should say there is not enough information."

FACTUAL = [  # (id, question, title, year, field)
    ("F1", "Who directed Interstellar?", "Interstellar", 2014, "director"),
    ("F2", "Who directed Get Out?", "Get Out", 2017, "director"),
    ("F3", "Who directed Mad Max: Fury Road?", "Mad Max: Fury Road", 2015, "director"),
    ("F4", "In what year was Groundhog Day released?", "Groundhog Day", 1993, "year"),
    ("F5", "Who directed Hereditary?", "Hereditary", 2018, "director"),
    ("F6", "Who directed the 2015 film Room?", "Room", 2015, "director"),
    ("F7", "Who directed The Shape of Water?", "The Shape of Water", 2017, "director"),
    ("F8", "Which war film from 2017 did Christopher Nolan direct?", "Dunkirk", 2017, "title"),
]

RATING = [  # (id, question, kind, films)
    ("R1", "What is the IMDb rating of The Dark Knight?", "imdb", [("The Dark Knight", 2008)]),
    ("R2", "Which is rated higher on IMDb, Inception or Interstellar?", "higher", [("Inception", 2010), ("Interstellar", 2014)]),
    ("R3", "Which has the higher IMDb rating, Alien or Aliens?", "higher", [("Alien", 1979), ("Aliens", 1986)]),
    ("R4", "What is the critics' average score for Whiplash in this database?", "critics", [("Whiplash", 2014)]),
    ("R5", "What is the critics' average score for Suicide Squad?", "critics", [("Suicide Squad", 2016)]),
    ("R6", "Is Heat rated higher than Joker on IMDb?", "higher", [("Heat", 1995), ("Joker", 2019)]),
    ("R7", "Which is rated higher, Logan or Deadpool?", "higher", [("Logan", 2017), ("Deadpool", 2016)]),
]

OPINION = [  # (id, question, title, year, aspect, words that must appear in its reviews; an answer needs one)
    ("O1", "What do critics praise about Mad Max: Fury Road?", "Mad Max: Fury Road", 2015, "its action and stunts",
     ["action", "stunt", "chase", "Theron"]),
    ("O2", "What did critics dislike about Suicide Squad?", "Suicide Squad", 2016, "its weaknesses",
     ["mess", "plot", "Leto", "Joker"]),
    ("O3", "What do critics say about Heath Ledger's performance in The Dark Knight?", "The Dark Knight", 2008,
     "Heath Ledger's Joker", ["Ledger"]),
    ("O4", "What do critics say about Joaquin Phoenix in Joker?", "Joker", 2019, "Joaquin Phoenix's performance",
     ["Phoenix"]),
    ("O5", "What do reviewers think of the visuals in Avatar?", "Avatar", 2009, "its visual effects",
     ["visual", "effects", "3-D", "spectacle"]),
    ("O6", "What are the main criticisms of Batman v Superman: Dawn of Justice?", "Batman v Superman: Dawn of Justice", 2016,
     "its weaknesses", ["Snyder", "dark", "mess", "long"]),
    ("O7", "Do critics think Gravity is worth seeing?", "Gravity", 2013, "whether it is worth seeing",
     ["Bullock", "Cuarón", "space", "visual"]),
    ("O8", "What do critics say about the music in La La Land?", "La La Land", 2016, "its music and songs",
     ["music", "song", "Gosling", "Stone"]),
]

PLOT = [  # (id, question, title, year)
    ("P1", "Which film follows a family that must stay silent to survive creatures that hunt by sound?", "A Quiet Place", 2018),
    ("P2", "Which movie is about a young con artist who poses as a pilot and a doctor while an FBI agent chases him?",
     "Catch Me If You Can", 2002),
    ("P3", "Which film is about a cowboy doll who feels replaced by a new space ranger toy?", "Toy Story", 1995),
    ("P4", "Which movie tells of a mute cleaning woman who falls in love with an amphibian creature held in a government lab?",
     "The Shape of Water", 2017),
    ("P5", "Which film is about a lonely writer who falls in love with his computer's operating system?", "Her", 2013),
    ("P6", "Which movie follows a programmer invited to test whether a humanoid robot is truly conscious?", "Ex Machina", 2015),
    ("P7", "Which film follows journalists uncovering child abuse covered up by the Catholic Church in Boston?", "Spotlight", 2015),
    ("P8", "Which movie is about a young woman and her son held captive in a small shed for years?", "Room", 2015),
]

RECOMMENDATION = [  # (id, question, genres (all required), year_min, rating_min)
    ("C1", "Recommend some animated family films rated 8 or higher.", ["Animation", "Family"], None, 8.0),
    ("C2", "Suggest war movies released since 2010.", ["War"], 2010, None),
    ("C3", "Any good horror films from 2015 onwards rated at least 7?", ["Horror"], 2015, 7.0),
    ("C4", "I want a music-themed movie with an IMDb rating of 8 or higher.", ["Music"], None, 8.0),
    ("C5", "Recommend crime thrillers from 2010 or later rated 8 or higher.", ["Crime", "Thriller"], 2010, 8.0),
]

COMPARISON = [  # (id, question, films) — no pair from the 8 two-film tuning questions of Phase 18
    ("M1", "Compare Arrival and Interstellar as science-fiction films.", [("Arrival", 2016), ("Interstellar", 2014)]),
    ("M2", "How do critics compare Get Out and Us, both directed by Jordan Peele?", [("Get Out", 2017), ("Us", 2019)]),
    ("M3", "Compare Whiplash and La La Land, both by Damien Chazelle.", [("Whiplash", 2014), ("La La Land", 2016)]),
    ("M4", "Is Black Swan more disturbing than Hereditary?", [("Black Swan", 2010), ("Hereditary", 2018)]),
    ("M5", "Compare Heat and Nightcrawler as crime films.", [("Heat", 1995), ("Nightcrawler", 2014)]),
    ("M6", "Which do critics like more, Up or Inside Out?", [("Up", 2009), ("Inside Out", 2015)]),
    ("M7", "Compare Gravity with The Martian as survival-in-space films.", [("Gravity", 2013), ("The Martian", 2015)]),
    ("M8", "How does Ex Machina compare to Her as films about artificial intelligence?", [("Ex Machina", 2015), ("Her", 2013)]),
    ("M9", "Compare Dunkirk and 1917 as war films.", [("Dunkirk", 2017), ("1917", 2019)]),
]

NO_ANSWER = [  # (id, question, title as asked) — none of these is in the data
    ("N1", "Who directed The Shawshank Redemption?", "The Shawshank Redemption"),
    ("N2", "What did critics think of Oppenheimer (2023)?", "Oppenheimer"),
    ("N3", "What is the IMDb rating of The Matrix?", "The Matrix"),
    ("N4", "What do critics say about Parasite (2019)?", "Parasite"),
]


def label(m: Movie) -> str:
    return f"{m.title} ({m.year})"


def movie(session: Session, title: str, year: int) -> Movie:
    found = session.scalars(select(Movie).where(Movie.title == title, Movie.year == year)).all()
    if len(found) != 1:
        sys.exit(f"Expected exactly one movie {title} ({year}) in the database, found {len(found)}")
    return found[0]


def critic_average(session: Session, m: Movie) -> float:
    return round(float(session.scalar(select(func.avg(Review.review_rating)).where(Review.movie_id == m.id))), 2)


def base(id_: str, type_: str, question: str, expected_answer: str, movies: list[Movie], must: list[str],
         tool: list[str] | None = None, **extra) -> dict:
    return {
        "id": id_, "type": type_, "question": question, "expected_answer": expected_answer,
        "expected_movie": label(movies[0]) if movies else None,
        "expected_movies": [label(m) for m in movies],
        "must_contain_any": must, "expected_tool": tool, "no_answer": False, **extra,
    }


def build(session: Session) -> list[dict]:
    rows = []
    for id_, q, title, year, field in FACTUAL:
        m = movie(session, title, year)
        value = {"director": m.director, "year": str(m.year), "title": m.title}[field]
        rows.append(base(id_, "factual", q, f"{value}.", [m], [value]))

    for id_, q, kind, films in RATING:
        ms = [movie(session, t, y) for t, y in films]
        if kind == "imdb":
            r = f"{float(ms[0].rating):.1f}"
            # No tool returns one film's IMDb rating (get_movie_metadata comes in Phase 24): it is in the profile chunk.
            rows.append(base(id_, "rating", q, f"{label(ms[0])} has an IMDb rating of {r}.", ms, [r]))
        elif kind == "critics":
            avg = critic_average(session, ms[0])
            rows.append(base(id_, "rating", q, f"The critics' average for {label(ms[0])} is {avg} out of 10.", ms,
                             [f"{avg:.2f}", f"{avg:.1f}"], ["rating_summary"]))
        else:
            a, b = ms
            ra, rb = float(a.rating), float(b.rating)
            if ra == rb:
                answer, must = f"Neither: {a.title} and {b.title} are tied at {ra:.1f} on IMDb.", [f"{ra:.1f}", "tie", "same"]
            else:
                hi, lo = (a, b) if ra > rb else (b, a)
                answer, must = (f"{hi.title} is rated higher ({float(hi.rating):.1f} vs {float(lo.rating):.1f}).",
                                [hi.title])
            rows.append(base(id_, "rating", q, answer, ms, must, ["compare_movies"]))

    for id_, q, title, year, aspect, words in OPINION:
        m = movie(session, title, year)
        texts = " ".join(session.scalars(select(Review.review_text).where(Review.movie_id == m.id)).all()).lower()
        present = [w for w in words if w.lower() in texts]
        if not present:
            sys.exit(f"{id_}: none of {words} appear in the reviews of {label(m)}")
        rows.append(base(id_, "opinion", q, f"A summary of what critics in the database say about {aspect} in {label(m)}, "
                         f"grounded in the retrieved reviews.", [m], present))

    for id_, q, title, year in PLOT:
        m = movie(session, title, year)
        rows.append(base(id_, "plot", q, f"{label(m)}.", [m], [m.title]))

    for id_, q, genres, year_min, rating_min in RECOMMENDATION:
        stmt = select(Movie).where(*[literal(g) == any_(Movie.genres) for g in genres])
        if year_min:
            stmt = stmt.where(Movie.year >= year_min)
        if rating_min is not None:
            stmt = stmt.where(Movie.rating >= rating_min)
        matches = session.scalars(stmt.order_by(Movie.rating.desc(), Movie.title)).all()
        if not matches:
            sys.exit(f"{id_}: no film satisfies the request")
        rows.append(base(id_, "recommendation", q,
                         f"Any of the {len(matches)} films that satisfy the request, e.g. {', '.join(label(m) for m in matches[:3])}.",
                         [], [m.title for m in matches], ["filter_movies"],
                         acceptable_movies=[label(m) for m in matches],
                         criteria={"genres": genres, "year_min": year_min, "rating_min": rating_min}))

    for id_, q, films in COMPARISON:
        ms = [movie(session, t, y) for t, y in films]
        rows.append(base(id_, "comparison", q, f"A comparison of {' and '.join(label(m) for m in ms)} that covers both films.",
                         ms, [m.title for m in ms]))

    for id_, q, title in NO_ANSWER:
        if session.scalar(select(func.count()).select_from(Movie).where(Movie.title.ilike(f"{title}%"))):
            sys.exit(f"{id_}: {title} IS in the database, so it cannot be a no-answer question")
        rows.append({"id": id_, "type": "no_answer", "question": q, "expected_answer": NO_ANSWER_TEXT,
                     "expected_movie": None, "expected_movies": [], "must_contain_any": [],
                     "expected_tool": None, "no_answer": True})
    return rows


def main() -> None:
    with Session(get_engine()) as session:
        rows = build(session)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    counts = {}
    for r in rows:
        counts[r["type"]] = counts.get(r["type"], 0) + 1
    print(f"{len(rows)} questions written to {OUT.relative_to(ROOT)}: {counts}")
    for r in rows:
        extra = f" ({len(r['acceptable_movies'])} acceptable films)" if r.get("acceptable_movies") else ""
        print(f"  {r['id']:3} {r['type']:14} {r['question'][:70]:70} -> {r['expected_answer'][:70]}{extra}")


if __name__ == "__main__":
    main()
