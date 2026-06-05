"""
src/vectorizer.py
-----------------
Fits a TF-IDF vectorizer on the movie 'soup' column and computes the
pairwise cosine similarity matrix.  Serializes all artifacts to disk
so the Streamlit app can load them instantly without recomputing.

ML Concepts:
    TF-IDF (Term Frequency–Inverse Document Frequency)
        TF(t,d)  = (# times term t appears in doc d) / (# terms in doc d)
        IDF(t)   = log(N / df(t))  where N = total docs, df(t) = docs containing t
        TF-IDF   = TF × IDF  →  high for terms that are distinctive to a document

    Cosine Similarity
        sim(A, B) = (A · B) / (||A|| × ||B||)
        Range [0, 1] where 1 = identical direction in vector space.
        We use it instead of Euclidean distance because vector magnitude
        (i.e. movie overview length) should not affect similarity.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ── paths ───────────────────────────────────────────────────────────────────────
MODELS_DIR          = os.path.join(os.path.dirname(__file__), "..", "models")
COSINE_SIM_PATH     = os.path.join(MODELS_DIR, "cosine_sim.pkl")
MOVIE_INDICES_PATH  = os.path.join(MODELS_DIR, "movie_indices.pkl")
MOVIES_CLEAN_PATH   = os.path.join(MODELS_DIR, "movies_clean.pkl")


def build_tfidf_matrix(df: pd.DataFrame):
    """
    Fit a TF-IDF vectorizer on the 'soup' column.

    Args:
        df : Preprocessed DataFrame with a 'soup' column.

    Returns:
        tfidf_matrix : Sparse matrix of shape (n_movies, n_features)
        vectorizer   : Fitted TfidfVectorizer instance
    """
    print("[TF-IDF] Fitting vectorizer on feature soup...")

    vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),      # Unigrams + bigrams (e.g. "science fiction")
        min_df=2,                 # Ignore terms appearing in < 2 documents
        max_df=0.95,              # Ignore terms appearing in > 95% of documents
        sublinear_tf=True,        # Apply log normalization to TF: 1 + log(TF)
        strip_accents="unicode",  # Normalize accented characters
        stop_words="english",     # Remove English stop words
    )

    tfidf_matrix = vectorizer.fit_transform(df["soup"])

    n_movies, n_features = tfidf_matrix.shape
    print(f"      Matrix shape : {n_movies:,} movies × {n_features:,} features")
    print(f"      Sparsity     : {100 * (1 - tfidf_matrix.nnz / (n_movies * n_features)):.1f}%")

    return tfidf_matrix, vectorizer


def compute_cosine_similarity(tfidf_matrix) -> np.ndarray:
    """
    Compute the pairwise cosine similarity matrix from the TF-IDF matrix.

    Uses sklearn's optimized implementation which handles sparse matrices
    efficiently.  For 4,800 movies this produces a 4800×4800 float32 matrix
    (~88 MB), which is acceptable for a local deployment.

    Args:
        tfidf_matrix : Sparse TF-IDF matrix (n_movies × n_features)

    Returns:
        cosine_sim : Dense ndarray of shape (n_movies, n_movies)
    """
    print("[Cosine Similarity] Computing pairwise similarity matrix...")
    print("      This may take 10–30 seconds for ~4,800 movies...")

    # linear_kernel is faster than cosine_similarity for TF-IDF
    # because TF-IDF rows are already L2-normalized by sklearn
    from sklearn.metrics.pairwise import linear_kernel
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

    print(f"      Similarity matrix shape : {cosine_sim.shape}")
    print(f"      Memory usage            : {cosine_sim.nbytes / 1e6:.1f} MB")

    return cosine_sim


def build_index(df: pd.DataFrame) -> pd.Series:
    """
    Build a reverse mapping from movie title → DataFrame row index.
    Lowercased for case-insensitive lookup.

    Args:
        df : Cleaned DataFrame with a 'title' column.

    Returns:
        Series mapping title (str, lowercase) → integer index
    """
    indices = pd.Series(df.index, index=df["title"].str.lower()).drop_duplicates()
    return indices


def save_artifacts(cosine_sim: np.ndarray,
                   movie_indices: pd.Series,
                   df_clean: pd.DataFrame) -> None:
    """
    Persist all ML artifacts to the models/ directory using joblib.

    Args:
        cosine_sim    : Pairwise cosine similarity matrix
        movie_indices : Title → index mapping Series
        df_clean      : Cleaned movie DataFrame
    """
    os.makedirs(MODELS_DIR, exist_ok=True)

    print("[Save] Writing artifacts to disk...")
    joblib.dump(cosine_sim,    COSINE_SIM_PATH,    compress=3)
    joblib.dump(movie_indices, MOVIE_INDICES_PATH, compress=3)
    joblib.dump(df_clean,      MOVIES_CLEAN_PATH,  compress=3)

    sizes = {
        "cosine_sim.pkl"    : os.path.getsize(COSINE_SIM_PATH)    / 1e6,
        "movie_indices.pkl" : os.path.getsize(MOVIE_INDICES_PATH) / 1e6,
        "movies_clean.pkl"  : os.path.getsize(MOVIES_CLEAN_PATH)  / 1e6,
    }
    for fname, mb in sizes.items():
        print(f"      Saved {fname:<25} ({mb:.1f} MB)")


def load_artifacts():
    """
    Load pre-computed ML artifacts from disk.

    Returns:
        Tuple of (cosine_sim, movie_indices, df_clean)

    Raises:
        FileNotFoundError: If artifacts haven't been built yet (run setup.py).
    """
    for path in [COSINE_SIM_PATH, MOVIE_INDICES_PATH, MOVIES_CLEAN_PATH]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Artifact not found: {path}\n"
                "Please run `python setup.py` first to build the ML model."
            )

    cosine_sim    = joblib.load(COSINE_SIM_PATH)
    movie_indices = joblib.load(MOVIE_INDICES_PATH)
    df_clean      = joblib.load(MOVIES_CLEAN_PATH)

    return cosine_sim, movie_indices, df_clean


def artifacts_exist() -> bool:
    """Return True if all model artifacts are present on disk."""
    return all(os.path.exists(p) for p in [
        COSINE_SIM_PATH, MOVIE_INDICES_PATH, MOVIES_CLEAN_PATH
    ])
