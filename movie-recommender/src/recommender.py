"""
src/recommender.py
------------------
Core recommendation engine.  Loads pre-built ML artifacts and exposes
a clean API for:
    - Fuzzy title search (handles typos / partial titles)
    - Top-N content-based recommendations with similarity scores
    - Movie metadata lookup

This module is intentionally decoupled from the UI layer (app.py) so
it can be reused in scripts, notebooks, or a future API.
"""

import os
import pandas as pd
import numpy as np
from difflib import get_close_matches
from src.vectorizer import load_artifacts, artifacts_exist


class MovieRecommender:
    """
    Content-Based Movie Recommendation Engine.

    Attributes:
        cosine_sim    (np.ndarray)  : Pairwise cosine similarity matrix (N×N)
        movie_indices (pd.Series)   : Lowercase title → DataFrame index mapping
        df            (pd.DataFrame): Cleaned movie metadata DataFrame
        titles        (list[str])   : All lowercase movie titles for fuzzy search
    """

    def __init__(self):
        """Load artifacts from disk and initialise the engine."""
        if not artifacts_exist():
            raise RuntimeError(
                "ML artifacts not found. Run `python setup.py` to build the model."
            )

        print("[Recommender] Loading ML artifacts...")
        self.cosine_sim, self.movie_indices, self.df = load_artifacts()
        self.titles = self.movie_indices.index.tolist()   # lowercase title list
        print(f"[Recommender] Ready — {len(self.df):,} movies indexed.\n")

    # ── public API ─────────────────────────────────────────────────────────────

    def search_title(self, query: str, n_suggestions: int = 5) -> list[str]:
        """
        Fuzzy-search for movie titles matching the query string.

        Uses Python's difflib.get_close_matches for approximate matching,
        then falls back to substring search if no close matches are found.

        Args:
            query         : User's search string (case-insensitive).
            n_suggestions : Maximum number of suggestions to return.

        Returns:
            List of original-case movie titles sorted by relevance.
        """
        query_lower = query.lower().strip()

        # 1. Exact match check
        if query_lower in self.movie_indices:
            idx = self.movie_indices[query_lower]
            # Handle duplicate keys (returns scalar or Series)
            if isinstance(idx, pd.Series):
                idx = idx.iloc[0]
            return [self.df.iloc[idx]["title"]]

        # 2. Fuzzy match (handles typos)
        fuzzy_matches = get_close_matches(
            query_lower,
            self.titles,
            n=n_suggestions,
            cutoff=0.5
        )

        # 3. Substring fallback (handles partial titles like "dark knight")
        if not fuzzy_matches:
            fuzzy_matches = [
                t for t in self.titles if query_lower in t
            ][:n_suggestions]

        # Map lowercase matches back to original-case titles
        results = []
        for match in fuzzy_matches:
            idx = self.movie_indices[match]
            if isinstance(idx, pd.Series):
                idx = idx.iloc[0]
            results.append(self.df.iloc[idx]["title"])

        return results

    def get_recommendations(self,
                            title: str,
                            top_n: int = 10) -> pd.DataFrame:
        """
        Return top-N movies most similar to the given title.

        Algorithm:
            1. Retrieve the row index for the query movie
            2. Extract its cosine similarity scores against all other movies
            3. Sort scores descending, exclude the query movie itself
            4. Return metadata for the top_n results

        Args:
            title : Exact movie title (use search_title() to resolve first).
            top_n : Number of recommendations to return (default 10).

        Returns:
            DataFrame with columns:
                rank, title, genres_display, year,
                similarity_score, vote_average, vote_count, overview

        Raises:
            ValueError: If the title is not found in the index.
        """
        title_lower = title.lower().strip()

        if title_lower not in self.movie_indices:
            raise ValueError(
                f"Movie '{title}' not found in the dataset. "
                "Use search_title() to find the correct title."
            )

        # Get the integer row index for this movie
        idx = self.movie_indices[title_lower]
        if isinstance(idx, pd.Series):
            idx = int(idx.iloc[0])
        else:
            idx = int(idx)

        # Extract cosine similarity scores for this movie vs all others
        sim_scores = list(enumerate(self.cosine_sim[idx]))

        # Sort by similarity score, descending
        sim_scores_sorted = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        # Skip the first result (the query movie itself, score = 1.0)
        sim_scores_sorted = sim_scores_sorted[1 : top_n + 1]

        # Build results DataFrame
        movie_indices_top = [i for i, _ in sim_scores_sorted]
        scores_top        = [round(float(s), 4) for _, s in sim_scores_sorted]

        results = self.df.iloc[movie_indices_top][
            ["title", "genres_display", "year", "vote_average", "vote_count", "overview"]
        ].copy()

        results.insert(0, "rank", range(1, len(results) + 1))
        results["similarity_score"] = scores_top

        # Round vote_average to 1 decimal place for display
        results["vote_average"] = results["vote_average"].round(1)

        return results.reset_index(drop=True)

    def get_movie_info(self, title: str) -> dict:
        """
        Return metadata for a single movie by title.

        Args:
            title : Exact movie title (case-insensitive).

        Returns:
            Dict with movie metadata, or None if not found.
        """
        title_lower = title.lower().strip()
        if title_lower not in self.movie_indices:
            return None

        idx = self.movie_indices[title_lower]
        if isinstance(idx, pd.Series):
            idx = int(idx.iloc[0])
        else:
            idx = int(idx)

        row = self.df.iloc[idx]
        return {
            "title"         : row["title"],
            "genres"        : row["genres_display"],
            "year"          : row["year"],
            "rating"        : round(float(row["vote_average"]), 1),
            "vote_count"    : int(row["vote_count"]),
            "overview"      : row["overview"],
            "popularity"    : round(float(row["popularity"]), 1),
        }

    def get_all_titles(self) -> list[str]:
        """Return the full list of original-case movie titles (sorted A-Z)."""
        return sorted(self.df["title"].tolist())
