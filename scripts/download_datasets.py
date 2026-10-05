"""Download the two raw Hugging Face files into data/raw/.

- contemmcm/rotten_tomatoes: critic reviews (one row per review)
- HenryWaltson/TMDB-IMDB-Movies-Dataset: movie metadata (one row per movie)

Run from the repo root: python scripts/download_datasets.py
"""

from pathlib import Path

from huggingface_hub import hf_hub_download

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"

FILES = [
    ("contemmcm/rotten_tomatoes", "original.csv"),
    ("HenryWaltson/TMDB-IMDB-Movies-Dataset", "TMDB  IMDB Movies Dataset.csv"),
]


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for repo_id, filename in FILES:
        path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset", local_dir=RAW_DIR)
        print(f"{repo_id}/{filename} -> {path}")


if __name__ == "__main__":
    main()
