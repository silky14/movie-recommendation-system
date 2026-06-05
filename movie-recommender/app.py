"""
app.py
------
Streamlit front-end for the Content-Based Movie Recommendation System.

Features:
    • Fuzzy title search with auto-suggestions
    • Top-10 recommendation results with similarity scores
    • Query movie detail card (genre, year, rating, overview)
    • Recommendation cards with rank badge, score bar, and metadata
    • Similarity score explanation in sidebar

Run with:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
from src.recommender import MovieRecommender
from src.vectorizer import artifacts_exist

# ── page config (must be first Streamlit call) ──────────────────────────────────
st.set_page_config(
    page_title="Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── custom CSS ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .main { background-color: #0e1117; }

    /* Recommendation card */
    .rec-card {
        background: linear-gradient(135deg, #1a1f2e 0%, #16213e 100%);
        border: 1px solid #2d3561;
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 12px;
        transition: border-color 0.2s;
    }
    .rec-card:hover { border-color: #e50914; }

    /* Rank badge */
    .rank-badge {
        display: inline-block;
        background: #e50914;
        color: white;
        font-size: 11px;
        font-weight: 700;
        padding: 2px 8px;
        border-radius: 20px;
        margin-right: 8px;
        letter-spacing: 0.5px;
    }

    /* Score bar container */
    .score-bar-bg {
        background: #2d3561;
        border-radius: 4px;
        height: 6px;
        width: 100%;
        margin-top: 6px;
    }
    .score-bar-fill {
        background: linear-gradient(90deg, #e50914, #ff6b6b);
        border-radius: 4px;
        height: 6px;
    }

    /* Genre tag */
    .genre-tag {
        display: inline-block;
        background: #2d3561;
        color: #a8b2d8;
        font-size: 11px;
        padding: 2px 8px;
        border-radius: 10px;
        margin: 2px;
    }

    /* Query movie card */
    .query-card {
        background: linear-gradient(135deg, #1e2a4a 0%, #16213e 100%);
        border: 2px solid #e50914;
        border-radius: 14px;
        padding: 20px 24px;
        margin-bottom: 24px;
    }

    /* Star rating */
    .star-rating { color: #ffd700; font-size: 14px; }

    /* Section header */
    .section-header {
        font-size: 20px;
        font-weight: 700;
        color: #e8eaf6;
        margin: 20px 0 12px 0;
        padding-bottom: 8px;
        border-bottom: 2px solid #e50914;
    }

    /* Overview text */
    .overview-text {
        font-size: 13px;
        color: #8892b0;
        line-height: 1.6;
        margin-top: 8px;
    }

    /* Hide Streamlit footer */
    footer { visibility: hidden; }
    #MainMenu { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── helper functions ────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading ML model...")
def load_recommender() -> MovieRecommender:
    """Load and cache the recommender engine (runs once per session)."""
    return MovieRecommender()


def render_score_bar(score: float) -> str:
    """Render an HTML score bar scaled to the score value."""
    width_pct = int(score * 100)
    return f"""
    <div class="score-bar-bg">
        <div class="score-bar-fill" style="width: {width_pct}%;"></div>
    </div>
    """


def render_genre_tags(genres_str: str) -> str:
    """Convert a comma-separated genres string into HTML genre tags."""
    if genres_str == "N/A" or not genres_str:
        return '<span class="genre-tag">N/A</span>'
    tags = "".join(
        f'<span class="genre-tag">{g.strip()}</span>'
        for g in genres_str.split(",")
    )
    return tags


def stars_from_rating(rating: float) -> str:
    """Convert a 0–10 rating to a star string (out of 5)."""
    stars = round(rating / 2)  # scale 10 → 5
    return "★" * stars + "☆" * (5 - stars)


def render_query_card(info: dict) -> None:
    """Render the selected movie's detail card."""
    year_str = f" ({info['year']})" if info["year"] != "N/A" else ""
    rating   = info["rating"]
    overview = info["overview"] if info["overview"] else "No overview available."
    # Truncate long overviews
    if len(overview) > 300:
        overview = overview[:300].rsplit(" ", 1)[0] + "..."

    st.markdown(f"""
    <div class="query-card">
        <div style="font-size: 22px; font-weight: 800; color: #e8eaf6; margin-bottom: 4px;">
            🎬 {info['title']}{year_str}
        </div>
        <div style="margin: 6px 0;">
            {render_genre_tags(info['genres'])}
        </div>
        <div style="margin: 8px 0; font-size: 13px; color: #a8b2d8;">
            <span class="star-rating">{stars_from_rating(rating)}</span>
            &nbsp;{rating}/10
            &nbsp;·&nbsp;
            <span style="color: #8892b0;">{info['vote_count']:,} votes</span>
            &nbsp;·&nbsp;
            <span style="color: #8892b0;">Popularity: {info['popularity']}</span>
        </div>
        <div class="overview-text">{overview}</div>
    </div>
    """, unsafe_allow_html=True)


def render_recommendation_card(row: pd.Series, rank: int) -> None:
    """Render a single recommendation result card."""
    score    = row["similarity_score"]
    year_str = f" ({row['year']})" if row["year"] != "N/A" else ""
    overview = row["overview"] if pd.notna(row["overview"]) and row["overview"] else ""
    if len(overview) > 200:
        overview = overview[:200].rsplit(" ", 1)[0] + "..."

    st.markdown(f"""
    <div class="rec-card">
        <div style="display: flex; justify-content: space-between; align-items: flex-start;">
            <div style="flex: 1;">
                <span class="rank-badge">#{rank}</span>
                <span style="font-size: 16px; font-weight: 700; color: #e8eaf6;">
                    {row['title']}{year_str}
                </span>
            </div>
            <div style="text-align: right; min-width: 90px;">
                <div style="font-size: 18px; font-weight: 800; color: #e50914;">
                    {score:.0%}
                </div>
                <div style="font-size: 10px; color: #8892b0; letter-spacing: 0.5px;">
                    MATCH
                </div>
            </div>
        </div>

        <div style="margin: 8px 0 4px 0;">
            {render_genre_tags(row['genres_display'])}
        </div>

        {render_score_bar(score)}

        <div style="margin-top: 8px; font-size: 12px; color: #8892b0;">
            <span class="star-rating" style="font-size: 11px;">
                {stars_from_rating(row['vote_average'])}
            </span>
            {row['vote_average']}/10 · {int(row['vote_count']):,} votes
        </div>

        {'<div class="overview-text">' + overview + '</div>' if overview else ''}
    </div>
    """, unsafe_allow_html=True)


# ── sidebar ─────────────────────────────────────────────────────────────────────

def render_sidebar(recommender: MovieRecommender) -> int:
    """Render the sidebar and return the selected top_n value."""
    with st.sidebar:
        st.markdown("## 🎬 Movie Recommender")
        st.markdown("---")

        st.markdown("### ⚙️ Settings")
        top_n = st.slider(
            "Number of recommendations",
            min_value=5, max_value=20, value=10, step=1
        )

        st.markdown("---")
        st.markdown("### 📊 How It Works")
        st.markdown("""
**Content-Based Filtering**

This system recommends movies based on
similarity of their content features:

- 📝 **Plot overview** (TF-IDF weighted)
- 🎭 **Genres**
- 🔑 **Keywords**
- 👤 **Top 3 cast members**
- 🎬 **Director**

**TF-IDF Vectorization**
Converts text features into numerical
vectors. Terms rare in the corpus but
frequent in a movie get high weights.

**Cosine Similarity**
Measures the angle between two movie
vectors. Score of **1.0 = identical**,
**0.0 = no overlap**.
        """)

        st.markdown("---")
        st.markdown("### 📈 Dataset")
        st.markdown(f"""
- **Source**: TMDB 5000 Movies
- **Movies indexed**: {len(recommender.df):,}
- **Algorithm**: Content-Based
- **Vectorizer**: TF-IDF (bigrams)
        """)

        st.markdown("---")
        st.caption("Built with scikit-learn · Streamlit · Pandas")

    return top_n


# ── main app ────────────────────────────────────────────────────────────────────

def main():

    # ── Guard: check artifacts exist ──────────────────────────────────────────
    if not artifacts_exist():
        st.error("⚠️  ML model not found.")
        st.markdown("""
        **Please build the model first:**
        ```bash
        python setup.py
        ```
        Then refresh this page.
        """)
        st.stop()

    # ── Load recommender (cached) ─────────────────────────────────────────────
    recommender = load_recommender()

    # ── Sidebar ───────────────────────────────────────────────────────────────
    top_n = render_sidebar(recommender)

    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown("""
    <div style="text-align: center; padding: 10px 0 20px 0;">
        <h1 style="font-size: 38px; font-weight: 900; color: #e8eaf6; margin: 0;">
            🎬 Movie Recommender
        </h1>
        <p style="color: #8892b0; font-size: 15px; margin: 6px 0 0 0;">
            Content-Based Filtering · TF-IDF · Cosine Similarity
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Search box ────────────────────────────────────────────────────────────
    col_search, col_btn = st.columns([5, 1])
    with col_search:
        query = st.text_input(
            label="search",
            placeholder="🔍  Search a movie title (e.g. The Dark Knight, Inception, Avatar)...",
            label_visibility="collapsed",
            key="search_input"
        )
    with col_btn:
        search_clicked = st.button("Search", use_container_width=True, type="primary")

    # ── Handle input ──────────────────────────────────────────────────────────
    if not query.strip():
        # Show popular example queries when app first loads
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### 💡 Try searching for:")
        ex_cols = st.columns(5)
        examples = ["The Dark Knight", "Inception", "The Avengers", "Titanic", "Interstellar"]
        for col, example in zip(ex_cols, examples):
            with col:
                if st.button(example, use_container_width=True):
                    st.session_state["search_input"] = example
                    st.rerun()
        st.stop()

    # ── Fuzzy title search ────────────────────────────────────────────────────
    suggestions = recommender.search_title(query, n_suggestions=6)

    if not suggestions:
        st.warning(f"No movies found matching **'{query}'**. Try a different title.")
        st.stop()

    # ── Title selection (if multiple suggestions) ─────────────────────────────
    if len(suggestions) == 1:
        selected_title = suggestions[0]
    else:
        st.markdown(
            f'<div class="section-header">🔎 Search Results for "{query}"</div>',
            unsafe_allow_html=True
        )
        selected_title = st.radio(
            "Select the movie you meant:",
            options=suggestions,
            horizontal=True,
            label_visibility="collapsed"
        )

    # ── Query movie card ──────────────────────────────────────────────────────
    movie_info = recommender.get_movie_info(selected_title)
    if movie_info:
        st.markdown(
            '<div class="section-header">📽️ Selected Movie</div>',
            unsafe_allow_html=True
        )
        render_query_card(movie_info)

    # ── Get recommendations ───────────────────────────────────────────────────
    with st.spinner(f"Finding {top_n} similar movies..."):
        try:
            recommendations = recommender.get_recommendations(selected_title, top_n=top_n)
        except ValueError as e:
            st.error(str(e))
            st.stop()

    # ── Render recommendation results ─────────────────────────────────────────
    st.markdown(
        f'<div class="section-header">🎯 Top {top_n} Recommendations</div>',
        unsafe_allow_html=True
    )

    # Split into two columns for better space usage
    col_left, col_right = st.columns(2)
    mid = len(recommendations) // 2 + len(recommendations) % 2  # ceiling split

    with col_left:
        for _, row in recommendations.iloc[:mid].iterrows():
            render_recommendation_card(row, int(row["rank"]))

    with col_right:
        for _, row in recommendations.iloc[mid:].iterrows():
            render_recommendation_card(row, int(row["rank"]))

    # ── Score legend ──────────────────────────────────────────────────────────
    with st.expander("ℹ️ Understanding similarity scores"):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("🔴 80–100%", "Very Similar", "Strong content overlap")
        with col2:
            st.metric("🟡 50–79%", "Somewhat Similar", "Partial feature overlap")
        with col3:
            st.metric("⚪ 0–49%", "Loosely Related", "Few shared features")

        st.markdown("""
        Scores represent **cosine similarity** between TF-IDF feature vectors.
        Two movies with identical genres, cast, director, keywords, and plot will
        score **1.00 (100%)**. Movies with no overlapping features score **0.00 (0%)**.
        """)


if __name__ == "__main__":
    main()
