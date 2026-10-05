# Dataset (Phase 2)

## Sources

| Source | What it gives us | Size |
|---|---|---|
| [`contemmcm/rotten_tomatoes`](https://huggingface.co/datasets/contemmcm/rotten_tomatoes) (`original.csv`) | Critic reviews: text, critic, publication, original score, fresh/rotten, date, URL | 1,444,963 reviews of 69,263 movies |
| [`HenryWaltson/TMDB-IMDB-Movies-Dataset`](https://huggingface.co/datasets/HenryWaltson/TMDB-IMDB-Movies-Dataset) | Movie metadata: title, release date, directors, writers, cast, genres, overview, keywords, IMDb rating and votes, IMDb ID | 434,803 rows, 427,311 after dropping duplicate IMDb IDs |

No single Hugging Face dataset has both reviews and full metadata. Most review datasets are sentiment
sets (text + positive/negative label) with no movie title, so they cannot support citations or tools.
Neither source states a licence; they are used here for coursework only.

Reproduce: `python scripts/download_datasets.py`, `python scripts/inspect_dataset.py`, `python scripts/select_subset.py`.
Full numbers: [`dataset_inspection.json`](dataset_inspection.json).

## What the inspection found

**Reviews**
- 4.8% have no text; 12,394 rows are exact duplicates of another review ID.
- Reviews are short: median 130 characters (21 words), longest 363 characters. None exceeds 500 characters.
- 0.3% are not English (mostly Portuguese and Spanish publications); 3.6% have encoding damage
  (`%u0161`-style escapes, dropped accents).
- Scores come in many formats: `x/5` (31%), `x/4` (20%), letter grades (11%), `x/10` (6%), `x/100`, and 30% have none.
  69% can be mapped to a 0–10 scale. Every review has a fresh/rotten flag (67% fresh).
- `reviewState` is identical to `scoreSentiment`. Some dates are placeholders (`1800-01-01`).

**Movies**
- 7,492 duplicate rows of the same IMDb ID. Missing: genres 18%, cast 16%, overview 9%, keywords 60%, directors 2%.
- 54% are English-language. Only about 4,500 movies have 50k+ IMDb votes.

## Joining the two sources

There is no shared ID. Rotten Tomatoes names a movie by URL slug (`prisoners_2013`, `the-social-network`,
`1193230-state_of_play`); we turn the TMDB title into the same slug form and compare.

| Match type | Rule | Share of reviews |
|---|---|---|
| `slug+year` | slug ends in a year; same title released within one year | 19% |
| `unique` | exactly one TMDB movie has the title | 53% |
| `dominant` | several share the title, the top one has 10× the IMDb votes of the next | 12% |
| `ambiguous` | several share the title, no clear winner (not used) | 5% |
| unmatched | no TMDB title fits (not used) | 12% |

When several RT pages land on one TMDB movie (`dune_2021` and Lynch's `1006364-dune`), the strongest match is
kept. Random spot checks of `unique`, `dominant` and `slug+year` matches were all correct.

## First subset

500 English-language movies with the most IMDb votes (337k–3.1M) that matched safely and have 10+ clean reviews,
and up to 20 reviews each: **9,987 reviews**. Per movie we prefer reviews with a usable score, then top critics,
then longer text. Every selected review has a score; 89% are from top critics. Includes the plan's example
movies (Prisoners, Zodiac, Inception).

## Consequence for chunking (Phase 4)

Because critic reviews are at most 363 characters, a single review never needs splitting. Phase 4 therefore
builds one document per review (so a retrieved chunk cites exactly one review) plus one profile document per
movie from its metadata, and chunks bodies at 1,000 characters with 150 overlap. Result for the first subset:
9,987 review chunks and 553 profile chunks; 53 long profiles (many keywords) split into two.
