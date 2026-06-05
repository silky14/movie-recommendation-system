"""
setup.py
--------
One-time setup script.  Run this ONCE before launching the Streamlit app.

What it does:
    1. Checks whether the TMDB CSV files already exist in data/
    2. If not, prompts the user with download instructions
    3. Runs the full ML pipeline:
         preprocess → TF-IDF → cosine similarity → save artifacts
    4. Prints a summary and confirms the app is ready to launch

Usage:
    python setup.py

Expected files in data/ before running:
    data/tmdb_5000_movies.csv
    data/tmdb_5000_credits.csv

Dataset source:
    https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata
"""

import os
import sys
import time

# ── path setup ─────────────────────────────────────────────────────────────────
PROJECT_ROOT  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR      = os.path.join(PROJECT_ROOT, "data")
MOVIES_PATH   = os.path.join(DATA_DIR, "tmdb_5000_movies.csv")
CREDITS_PATH  = os.path.join(DATA_DIR, "tmdb_5000_credits.csv")

# ── ANSI color helpers ─────────────────────────────────────────────────────────
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def info(msg):    print(f"{CYAN}ℹ  {msg}{RESET}")
def success(msg): print(f"{GREEN}✅ {msg}{RESET}")
def warn(msg):    print(f"{YELLOW}⚠  {msg}{RESET}")
def error(msg):   print(f"{RED}✗  {msg}{RESET}")
def header(msg):  print(f"\n{BOLD}{CYAN}{'─'*60}\n   {msg}\n{'─'*60}{RESET}\n")


# ── dataset validation ─────────────────────────────────────────────────────────

def check_dataset() -> bool:
    """Return True if both CSV files exist and are non-empty."""
    movies_ok  = os.path.exists(MOVIES_PATH)  and os.path.getsize(MOVIES_PATH)  > 0
    credits_ok = os.path.exists(CREDITS_PATH) and os.path.getsize(CREDITS_PATH) > 0
    return movies_ok and credits_ok


def print_download_instructions():
    """Print clear instructions for downloading the TMDB dataset."""
    print(f"""
{YELLOW}{BOLD}Dataset not found in data/ directory.{RESET}

{BOLD}Please download the TMDB 5000 Movie Dataset:{RESET}

  Option 1 — Kaggle Web UI (recommended):
  ─────────────────────────────────────────
  1. Go to: https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata
  2. Click "Download" (you need a free Kaggle account)
  3. Unzip the archive
  4. Copy these two files into the {YELLOW}data/{RESET} folder:
       • tmdb_5000_movies.csv
       • tmdb_5000_credits.csv

  Option 2 — Kaggle CLI (if you have it installed):
  ──────────────────────────────────────────────────
  kaggle datasets download -d tmdb/tmdb-movie-metadata -p data/ --unzip

{BOLD}Expected file locations:{RESET}
  {GREEN}data/tmdb_5000_movies.csv{RESET}
  {GREEN}data/tmdb_5000_credits.csv{RESET}

After placing the files, re-run:  {BOLD}python setup.py{RESET}
""")


# ── main pipeline ──────────────────────────────────────────────────────────────

def run_pipeline():
    """Execute the full ML preprocessing + model building pipeline."""

    from src.preprocess import load_and_preprocess
    from src.vectorizer import (
        build_tfidf_matrix,
        compute_cosine_similarity,
        build_index,
        save_artifacts,
        artifacts_exist,
    )

    header("MOVIE RECOMMENDER — SETUP PIPELINE")

    # ── Step 1: Preprocess ────────────────────────────────────────────────────
    info("Starting data preprocessing...")
    t0 = time.time()
    df_clean = load_and_preprocess(MOVIES_PATH, CREDITS_PATH)
    success(f"Preprocessing done in {time.time() - t0:.1f}s")

    # ── Step 2: TF-IDF ────────────────────────────────────────────────────────
    info("Building TF-IDF matrix...")
    t0 = time.time()
    tfidf_matrix, vectorizer = build_tfidf_matrix(df_clean)
    success(f"TF-IDF done in {time.time() - t0:.1f}s")

    # ── Step 3: Cosine Similarity ─────────────────────────────────────────────
    info("Computing cosine similarity matrix...")
    t0 = time.time()
    cosine_sim = compute_cosine_similarity(tfidf_matrix)
    success(f"Cosine similarity done in {time.time() - t0:.1f}s")

    # ── Step 4: Build Index ───────────────────────────────────────────────────
    info("Building title index...")
    movie_indices = build_index(df_clean)
    success(f"Index built — {len(movie_indices):,} titles mapped")

    # ── Step 5: Save ──────────────────────────────────────────────────────────
    info("Saving artifacts to models/...")
    save_artifacts(cosine_sim, movie_indices, df_clean)

    # ── Done ─────────────────────────────────────────────────────────────────
    header("SETUP COMPLETE")
    print(f"""
{GREEN}{BOLD}Everything is ready!{RESET}

  Artifacts saved to: {YELLOW}models/{RESET}
    • cosine_sim.pkl       (similarity matrix)
    • movie_indices.pkl    (title → index mapping)
    • movies_clean.pkl     (cleaned DataFrame)

  Launch the app with:
    {BOLD}streamlit run app.py{RESET}
""")


# ── entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # Check Python version
    if sys.version_info < (3, 9):
        error("Python 3.9+ is required.")
        sys.exit(1)

    # Check dataset presence
    if not check_dataset():
        print_download_instructions()
        sys.exit(1)

    info(f"Dataset found in {DATA_DIR}/")
    info(f"  tmdb_5000_movies.csv  : {os.path.getsize(MOVIES_PATH)  / 1e6:.1f} MB")
    info(f"  tmdb_5000_credits.csv : {os.path.getsize(CREDITS_PATH) / 1e6:.1f} MB\n")

    # Run the ML pipeline
    try:
        run_pipeline()
    except ImportError as e:
        error(f"Missing dependency: {e}")
        error("Run:  pip install -r requirements.txt")
        sys.exit(1)
    except Exception as e:
        error(f"Pipeline failed: {e}")
        raise
