# 🎬 Movie Recommendation System

A content-based movie recommendation system built with Machine Learning,
using TF-IDF Vectorization and Cosine Similarity on the TMDB 5000 Movie Dataset.

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange?logo=scikit-learn)](https://scikit-learn.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35%2B-red?logo=streamlit)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 📌 Project Overview

This project implements a **Content-Based Filtering** recommendation engine that
suggests movies similar to a user-selected title.  Given any movie in the TMDB
5000 dataset, the system computes similarity scores against all other movies and
returns the top-N most similar results.

The project is designed to demonstrate practical ML concepts
(feature engineering, TF-IDF, cosine similarity) in a clean, well-structured
codebase — suitable for resume showcasing and internship applications.

---

## ✨ Features

- 🔍 **Fuzzy title search** — handles typos and partial titles using `difflib`
- 🎯 **Top-N recommendations** — configurable (5–20) via sidebar slider
- 📊 **Similarity score display** — percentage match with visual score bar
- 🎭 **Genre tags** — color-coded genre display for each result
- ⭐ **Rating & vote count** — TMDB community rating per movie
- 📝 **Plot overview** — truncated overview for each recommendation
- 💡 **Example queries** — one-click starter suggestions on the home screen
- ⚡ **Cached model** — artifacts loaded once per session for fast queries

---

## 🧠 ML Concepts Used

### 1. Content-Based Filtering
Recommends items based on the features of the items themselves (not user
behaviour). A movie's features — genres, keywords, cast, director, and plot —
are combined into a single "feature soup" string per movie.

### 2. TF-IDF Vectorization (Term Frequency–Inverse Document Frequency)

```
TF(t, d)  = count(t in d) / total_terms(d)
IDF(t)    = log(N / df(t))          # N = corpus size, df = doc frequency
TF-IDF    = TF × IDF
```

High TF-IDF weight → term is **important to this document** and
**rare across the corpus**.  Common words like "the", "a" get suppressed.

**Configuration used:**
- `ngram_range=(1, 2)` — captures unigrams *and* bigrams (e.g. "science fiction")
- `min_df=2` — ignore terms in fewer than 2 movies
- `sublinear_tf=True` — applies log normalization to TF
- `stop_words="english"` — removes common English stop words

### 3. Cosine Similarity

```
sim(A, B) = (A · B) / (||A|| × ||B||)
```

Measures the cosine of the angle between two TF-IDF feature vectors.
- Score **1.0** → identical direction (same features)
- Score **0.0** → orthogonal (no shared features)

We use `linear_kernel` (equivalent to cosine similarity for L2-normalized
TF-IDF vectors) for speed.

### 4. Feature Engineering — The "Soup"

Each movie is represented by a concatenated string:

```python
soup = overview_tokens           # tokenized plot summary
     + genres                    # e.g. "action thriller"
     + keywords (top 5)          # e.g. "superhero gotham"
     + cast_top3 × 2             # doubled for upweighting
     + director × 3              # tripled for upweighting
```

Repeating important features (cast, director) increases their TF score,
making them stronger signals in the cosine similarity computation.

---

## 📁 Project Structure

```
movie-recommender/
│
├── data/
│   ├── .gitkeep                   # Placeholder; add CSVs here
│   ├── tmdb_5000_movies.csv       # (download separately)
│   └── tmdb_5000_credits.csv      # (download separately)
│
├── models/
│   ├── .gitkeep
│   ├── cosine_sim.pkl             # Pairwise cosine similarity matrix
│   ├── movie_indices.pkl          # Title → DataFrame index mapping
│   └── movies_clean.pkl           # Cleaned + preprocessed DataFrame
│
├── src/
│   ├── __init__.py
│   ├── preprocess.py              # Data loading, cleaning, feature engineering
│   ├── vectorizer.py              # TF-IDF + cosine similarity computation
│   └── recommender.py             # Recommendation engine (core ML logic)
│
├── app.py                         # Streamlit UI
├── setup.py                       # One-time dataset + model building script
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🗂️ Dataset

**TMDB 5000 Movie Dataset** — publicly available on Kaggle.

| File | Records | Key Columns |
|---|---|---|
| `tmdb_5000_movies.csv` | 4,803 | title, genres, keywords, overview, vote_average |
| `tmdb_5000_credits.csv` | 4,803 | cast, crew (JSON-encoded) |

**Source**: [https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)

---

## ⚙️ Installation

### Prerequisites
- Python 3.9 or higher
- pip

### Steps

**1. Clone the repository**
```bash
git clone https://github.com/YOUR_USERNAME/movie-recommender.git
cd movie-recommender
```

**2. Create a virtual environment (recommended)**
```bash
python -m venv venv

# macOS / Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Download the TMDB dataset**

Go to: https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata

Download and extract the zip, then place these two files in the `data/` folder:
```
data/tmdb_5000_movies.csv
data/tmdb_5000_credits.csv
```

Or if you have the Kaggle CLI installed:
```bash
kaggle datasets download -d tmdb/tmdb-movie-metadata -p data/ --unzip
```

**5. Build the ML model (one-time setup)**
```bash
python setup.py
```

This will:
- Preprocess and clean the dataset
- Build the TF-IDF matrix (~4,800 × 15,000 features)
- Compute the pairwise cosine similarity matrix
- Save artifacts to `models/`

Expected output:
```
[1/6] Loading datasets...
      Movies  : 4,803 rows
      Credits : 4,803 rows
[2/6] Merging...
...
✅ Setup complete. Run: streamlit run app.py
```

---

## 🚀 How to Run

```bash
streamlit run app.py
```

The app opens at **http://localhost:8501**

---

## 🖼️ Screenshots

> *Add screenshots after running the app locally*

| Home Screen | Search Results |
|---|---|
| ![Home](screenshots/home.png) | ![Results](screenshots/results.png) |

---

## 🔮 Future Improvements

| Improvement | Description |
|---|---|
| **Collaborative Filtering** | Add user-based recommendations using MovieLens ratings data |
| **Hybrid Model** | Combine content-based + collaborative filtering for better accuracy |
| **Neural Embeddings** | Replace TF-IDF with sentence-transformers for semantic similarity |
| **Poster Images** | Fetch movie posters from TMDB API |
| **Genre Filtering** | Allow users to filter recommendations by genre |
| **Popularity Weighting** | Blend cosine similarity with TMDB popularity score |
| **Evaluation Metrics** | Add precision@K, recall@K, NDCG for model evaluation |
| **Batch Recommendations** | Accept a list of liked movies and aggregate similarity |

---

## 🛠️ Technologies Used

| Technology | Version | Purpose |
|---|---|---|
| Python | 3.9+ | Core language |
| pandas | 2.0+ | Data loading and manipulation |
| NumPy | 1.24+ | Numerical operations |
| scikit-learn | 1.3+ | TF-IDF vectorization, cosine similarity |
| joblib | 1.3+ | Model artifact serialization |
| Streamlit | 1.35+ | Web application UI |

---

## 📐 ML Pipeline Diagram

```
TMDB CSVs
    │
    ▼
┌─────────────────────────────────────────────┐
│               PREPROCESSING                  │
│  • Parse JSON columns (genres, cast, crew)   │
│  • Extract director from crew                │
│  • Tokenize overview text                    │
│  • Build feature "soup" per movie            │
└─────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────┐
│            TF-IDF VECTORIZATION              │
│  • Fit TfidfVectorizer on soup column        │
│  • Output: sparse matrix (4803 × ~15000)     │
│  • bigrams, min_df=2, sublinear_tf           │
└─────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────┐
│          COSINE SIMILARITY MATRIX            │
│  • linear_kernel(tfidf, tfidf)               │
│  • Output: dense matrix (4803 × 4803)        │
│  • Each cell = similarity score [0, 1]       │
└─────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────┐
│           RECOMMENDATION ENGINE              │
│  • Lookup query movie row index              │
│  • Sort similarity scores descending         │
│  • Return top-N (excluding query movie)      │
└─────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────┐
│              STREAMLIT UI                    │
│  • Fuzzy search                              │
│  • Movie detail card                         │
│  • Recommendation cards with score bars      │
└─────────────────────────────────────────────┘
```

---

## 📄 License

This project is licensed under the MIT License.

---

## 👤 Author

Your Name — [GitHub](https://github.com/YOUR_USERNAME) · [LinkedIn](https://linkedin.com/in/YOUR_PROFILE)

---

*Built as a portfolio project for ML internship applications.*
