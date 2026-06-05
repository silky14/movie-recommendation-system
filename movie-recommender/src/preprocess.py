"""
src/preprocess.py
-----------------
Handles all data loading, cleaning, and feature engineering for the
Movie Recommendation System.

Pipeline:
    1. Load TMDB movies + credits CSVs
    2. Merge on movie title
    3. Parse JSON-encoded columns (genres, keywords, cast, crew)
    4. Extract top cast members and director
    5. Lowercase and strip spaces to avoid token duplication
    6. Build a unified 'soup' string per movie for TF-IDF input
"""

import ast
import pandas as pd
import numpy as np
import os


# ── helpers ────────────────────────────────────────────────────────────────────

def _safe_parse(obj):
    """
    Safely convert a stringified Python list/dict to a Python object.
    Returns an empty list on any parse failure.
    """
    try:
        return ast.literal_eval(obj)
    except (ValueError, SyntaxError):
        return []


def _extract_names(obj, top_n: int = None) -> list[str]:
    """
    Extract the 'name' field from a list of dicts.

    Args:
        obj   : Raw value from the DataFrame cell (string or list).
        top_n : If set, return only the first top_n names.

    Returns:
        List of lowercase, space-stripped name strings.
    """
    parsed = _safe_parse(obj) if isinstance(obj, str) else obj
    if not isinstance(parsed, list):
        return []
    names = [item["name"].lower().replace(" ", "") for item in parsed if "name" in item]
    return names[:top_n] if top_n else names


def _extract_director(obj) -> list[str]:
    """
    Extract the director name from the crew list.
    Returns a list with one element (for uniform handling), or empty list.
    """
    parsed = _safe_parse(obj) if isinstance(obj, str) else obj
    if not isinstance(parsed, list):
        return []
    for member in parsed:
        if member.get("job") == "Director":
            return [member["name"].lower().replace(" ", "")]
    return []


def _clean_text(text) -> str:
    """
    Lowercase and strip whitespace from a plain text field.
    Returns empty string for null values.
    """
    if pd.isna(text):
        return ""
    return str(text).lower().strip()


def _tokenize_overview(text) -> list[str]:
    """
    Split the overview into individual lowercase words.
    Filters out very short tokens (length < 3).
    """
    if pd.isna(text) or text == "":
        return []
    return [word.lower() for word in str(text).split() if len(word) >= 3]


# ── main preprocessing function ────────────────────────────────────────────────

def load_and_preprocess(movies_path: str, credits_path: str) -> pd.DataFrame:
    """
    Full preprocessing pipeline.

    Args:
        movies_path  : Path to tmdb_5000_movies.csv
        credits_path : Path to tmdb_5000_credits.csv

    Returns:
        Cleaned DataFrame with a 'soup' column ready for TF-IDF.
    """

    # ── 1. Load CSVs ────────────────────────────────────────────────────────
    print("[1/6] Loading datasets...")
    movies  = pd.read_csv(movies_path)
    credits = pd.read_csv(credits_path)

    print(f"      Movies  : {movies.shape[0]:,} rows, {movies.shape[1]} cols")
    print(f"      Credits : {credits.shape[0]:,} rows, {credits.shape[1]} cols")

    # ── 2. Merge ─────────────────────────────────────────────────────────────
    print("[2/6] Merging movies + credits on title...")
    # credits has columns: movie_id, title, cast, crew
    credits = credits.rename(columns={"movie_id": "id"})
    df = movies.merge(credits[["id", "cast", "crew"]], on="id", how="left")
    print(f"      Merged shape: {df.shape}")

    # ── 3. Select relevant columns ───────────────────────────────────────────
    print("[3/6] Selecting and validating columns...")
    required = ["id", "title", "genres", "keywords", "overview",
                "cast", "crew", "vote_average", "vote_count", "release_date",
                "popularity"]
    df = df[required].copy()

    # Drop rows with no title (can't recommend without it)
    df.dropna(subset=["title"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    print(f"      Rows after title-null drop: {df.shape[0]:,}")

    # ── 4. Parse JSON columns ────────────────────────────────────────────────
    print("[4/6] Parsing JSON-encoded columns...")
    df["genres_list"]   = df["genres"].apply(lambda x: _extract_names(x))
    df["keywords_list"] = df["keywords"].apply(lambda x: _extract_names(x, top_n=5))
    df["cast_list"]     = df["cast"].apply(lambda x: _extract_names(x, top_n=3))
    df["director"]      = df["crew"].apply(_extract_director)
    df["overview_tokens"] = df["overview"].apply(_tokenize_overview)

    # ── 5. Handle missing / null overview ───────────────────────────────────
    print("[5/6] Handling missing values...")
    df["overview"].fillna("", inplace=True)
    # Report missing stats
    null_overview = df["overview_tokens"].apply(len).eq(0).sum()
    print(f"      Movies with empty overview : {null_overview}")

    # ── 6. Build feature soup ────────────────────────────────────────────────
    # Each movie becomes a single string combining all semantic signals.
    # Director and cast are repeated to upweight their importance.
    print("[6/6] Building feature soup for TF-IDF...")

    def build_soup(row) -> str:
        tokens = (
            row["overview_tokens"]          # Natural language overview
            + row["genres_list"]            # Genres  (e.g. "action", "comedy")
            + row["keywords_list"]          # Plot keywords
            + row["cast_list"] * 2          # Top 3 cast — doubled for importance
            + row["director"] * 3           # Director  — tripled for importance
        )
        return " ".join(tokens)

    df["soup"] = df.apply(build_soup, axis=1)

    # ── 7. Clean up display columns ──────────────────────────────────────────
    df["genres_display"] = df["genres_list"].apply(
        lambda x: ", ".join([g.capitalize() for g in x]) if x else "N/A"
    )
    df["year"] = pd.to_datetime(df["release_date"], errors="coerce").dt.year.fillna(0).astype(int)
    df["year"] = df["year"].replace(0, "N/A")

    # Final cleaned DataFrame columns
    clean_cols = [
        "id", "title", "soup", "genres_display",
        "overview", "vote_average", "vote_count",
        "popularity", "year"
    ]
    df_clean = df[clean_cols].copy()

    print(f"\n✅ Preprocessing complete. Final shape: {df_clean.shape}")
    print(f"   Sample soup:\n   '{df_clean['soup'].iloc[0][:120]}...'\n")

    return df_clean
