
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.sparse import load_npz
from sklearn.metrics.pairwise import cosine_similarity


# =========================================================
# PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="MovieMatch",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# =========================================================
# CINEMATIC STYLING
# =========================================================
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(
            135deg, #09090f 0%, #17121f 55%, #0b101b 100%
        );
        color: #f5f5f7;
    }

    header[data-testid="stHeader"] {
        background: transparent;
    }

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        max-width: 1400px;
    }

    h1, h2, h3 {
        color: #ffffff !important;
        font-weight: 800 !important;
    }

    p, label {
        color: #e4e0ec;
    }

    .stButton > button {
        background: linear-gradient(90deg, #e50914, #b20710);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.7rem 1.4rem;
        font-weight: 700;
        transition: all 0.2s ease;
    }

    .stButton > button:hover {
        background: #ff2631;
        color: white;
        transform: translateY(-2px);
        border: none;
    }

    div[data-baseweb="select"] > div {
        background-color: #201d2a;
        border-color: #40384d;
        border-radius: 10px;
    }

    div[data-testid="stDataFrame"] {
        border: 1px solid #393343;
        border-radius: 12px;
        overflow: hidden;
    }

    [data-testid="stCaptionContainer"] {
        color: #b6b3c4;
    }

    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: rgba(28, 24, 39, 0.72);
        border-color: rgba(255, 255, 255, 0.12);
        border-radius: 15px;
    }

    .hero {
        padding: 44px 34px;
        border-radius: 20px;
        margin: 12px 0 28px 0;
        background:
            linear-gradient(
                90deg,
                rgba(10,8,18,0.97),
                rgba(10,8,18,0.48)
            ),
            url(
                'https://images.unsplash.com/photo-1489599849927-2ee91cede3ba?auto=format&fit=crop&w=1600&q=85'
            );
        background-size: cover;
        background-position: center;
        border: 1px solid rgba(255,255,255,0.12);
    }

    .hero-label {
        color: #ff5964;
        font-size: 12px;
        letter-spacing: 4px;
        font-weight: 800;
    }

    .hero h1 {
        font-size: clamp(32px, 5vw, 54px);
        line-height: 1.12;
        max-width: 650px;
        margin: 14px 0;
    }

    .hero p {
        color: #e0dce8;
        font-size: 16px;
        line-height: 1.7;
        max-width: 550px;
    }

    .movie-carousel {
        display: flex;
        gap: 18px;
        overflow-x: auto;
        padding: 12px 2px 22px 2px;
        scroll-snap-type: x mandatory;
        scrollbar-width: thin;
        scrollbar-color: #e50914 #201d2a;
    }

    .movie-tile {
        position: relative;
        flex: 0 0 235px;
        height: 300px;
        overflow: hidden;
        border-radius: 15px;
        scroll-snap-align: start;
        border: 1px solid rgba(255,255,255,0.12);
        background: #201d2a;
        transition: transform 0.25s ease;
    }

    .movie-tile:hover {
        transform: translateY(-5px);
    }

    .movie-tile img {
        width: 100%;
        height: 100%;
        object-fit: cover;
        transition: transform 0.35s ease;
    }

    .movie-tile:hover img {
        transform: scale(1.06);
    }

    .movie-overlay {
        position: absolute;
        inset: 0;
        display: flex;
        flex-direction: column;
        justify-content: flex-end;
        padding: 18px;
        background: linear-gradient(
            180deg,
            transparent 20%,
            rgba(8,7,13,0.42) 55%,
            rgba(8,7,13,0.97) 100%
        );
    }

    .movie-overlay span {
        color: #ff6872;
        font-size: 10px;
        letter-spacing: 2px;
        font-weight: 800;
    }

    .movie-overlay h3 {
        margin: 8px 0 0 0;
        font-size: 21px;
    }

    .movie-description {
        color: #d0cadc;
        font-size: 14px;
        line-height: 1.6;
    }

    @media (max-width: 600px) {
        .hero {
            padding: 28px 20px;
        }

        .movie-tile {
            flex-basis: 190px;
            height: 260px;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# HERO SECTION
# =========================================================
st.markdown(
    """
    <div class="hero">
        <div class="hero-label">YOUR PERSONAL CINEMA</div>
        <h1>Every movie leads<br>to another.</h1>
        <p>
            Discover your next obsession with recommendations
            powered by machine learning, natural language
            processing, and semantic similarity.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# CINEMATIC INSPIRATION CAROUSEL
# =========================================================

# =========================================================
# CINEMATIC INSPIRATION CAROUSEL
# =========================================================

# =========================================================
# ARTIFACT PATHS
# =========================================================
BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"


# =========================================================
# LOAD MOVIE DATA
# =========================================================
@st.cache_data
def load_movies():
    path = ARTIFACTS_DIR / "movies.csv"

    if not path.is_file():
        raise FileNotFoundError(f"Movie dataset not found: {path}")

    df = pd.read_csv(path)

    if "original_title" not in df.columns:
        raise ValueError(
            "movies.csv must contain the 'original_title' column."
        )

    return df.reset_index(drop=True)


# =========================================================
# LOAD SAVED MODELS
# =========================================================
@st.cache_resource
def load_models():
    required_files = [
        "tfidf_matrix.npz",
        "tfidf_rich_matrix.npz",
        "embeddings.npy",
        "rich_embeddings.npy",
    ]

    missing = [
        filename
        for filename in required_files
        if not (ARTIFACTS_DIR / filename).is_file()
    ]

    if missing:
        raise FileNotFoundError(
            "Missing model artifacts: " + ", ".join(missing)
        )

    tfidf_matrix = load_npz(
        ARTIFACTS_DIR / "tfidf_matrix.npz"
    )

    tfidf_rich_matrix = load_npz(
        ARTIFACTS_DIR / "tfidf_rich_matrix.npz"
    )

    embeddings = np.load(
        ARTIFACTS_DIR / "embeddings.npy",
        mmap_mode="r",
        allow_pickle=False,
    )

    rich_embeddings = np.load(
        ARTIFACTS_DIR / "rich_embeddings.npy",
        mmap_mode="r",
        allow_pickle=False,
    )

    return (
        tfidf_matrix,
        tfidf_rich_matrix,
        embeddings,
        rich_embeddings,
    )


# =========================================================
# VALIDATE DATA AND MODELS
# =========================================================
try:
    movies = load_movies()

    (
        tfidf_matrix,
        tfidf_rich_matrix,
        embeddings,
        rich_embeddings,
    ) = load_models()

    artifact_shapes = {
        "Original TF-IDF": tfidf_matrix.shape,
        "Rich TF-IDF": tfidf_rich_matrix.shape,
        "Original embeddings": embeddings.shape,
        "Rich embeddings": rich_embeddings.shape,
    }

    if any(len(shape) != 2 for shape in artifact_shapes.values()):
        raise ValueError("Every model artifact must be a 2D matrix.")

    row_counts = {
        name: shape[0]
        for name, shape in artifact_shapes.items()
    }

    row_counts["Movie dataset"] = len(movies)

    if len(set(row_counts.values())) != 1:
        raise ValueError(
            f"Artifact row counts do not match: {row_counts}. "
            "Regenerate the artifacts using the same dataset "
            "and movie ordering."
        )

    if embeddings.shape[1] == 0 or rich_embeddings.shape[1] == 0:
        raise ValueError("The embedding matrices are empty.")

except (FileNotFoundError, ValueError, OSError) as error:
    st.error(f"MovieMatch could not load its data: {error}")
    st.info(
        "Ensure movies.csv and all four model artifacts are "
        "inside the artifacts folder."
    )
    st.stop()


# =========================================================
# APP INTRODUCTION
# =========================================================
st.divider()

st.title("🎬 MovieMatch")
st.write(
    "Discover movies similar to your favorites using "
    "machine learning and deep learning."
)

st.caption(
    "Choose a movie, experiment with four recommendation "
    "methods, and discover your next watch."
)


# =========================================================
# MOVIE SELECTION
# =========================================================
valid_indices = movies.index[
    movies["original_title"].notna()
    & movies["original_title"].astype(str).str.strip().ne("")
].tolist()

if not valid_indices:
    st.error("No valid movie titles were found in the dataset.")
    st.stop()


def get_movie_label(index):
    movie = movies.iloc[index]
    title = str(movie["original_title"]).strip()

    # Add release year when available.
    if "release_date" in movies.columns:
        release_date = movie["release_date"]

        if pd.notna(release_date):
            date_text = str(release_date).strip()

            if len(date_text) >= 4 and date_text[:4].isdigit():
                title = f"{title} ({date_text[:4]})"

    # Add ID when available to help distinguish duplicate titles.
    if "id" in movies.columns:
        movie_id = movie["id"]

        if pd.notna(movie_id):
            title = f"{title} · ID {movie_id}"

    return title


left, right = st.columns([2, 1])

with left:
    selected_idx = st.selectbox(
        "🎥 Choose a movie",
        options=valid_indices,
        format_func=get_movie_label,
        key="selected_movie",
    )

with right:
    top_n = st.slider(
        "Number of recommendations",
        min_value=5,
        max_value=20,
        value=10,
        step=5,
    )


method = st.selectbox(
    "🧠 Recommendation method",
    [
        "TF-IDF — Original Features",
        "TF-IDF — Rich Features",
        "Sentence Transformer — Original Features",
        "Sentence Transformer — Rich Features",
    ],
    help=(
        "TF-IDF compares word-based text features. "
        "Sentence Transformers compare semantic representations. "
        "Rich features use the expanded text representation "
        "prepared during model development."
    ),
)


# =========================================================
# GENERATE RECOMMENDATIONS
# =========================================================
if st.button(
    "✨ Find Similar Movies",
    type="primary",
    use_container_width=True,
):

    with st.spinner("Searching for your next favorite movie..."):

        # Select the matrix associated with the chosen method.
        if method == "TF-IDF — Original Features":
            matrix = tfidf_matrix

        elif method == "TF-IDF — Rich Features":
            matrix = tfidf_rich_matrix

        elif method == "Sentence Transformer — Original Features":
            matrix = embeddings

        else:
            matrix = rich_embeddings

        # Cosine similarity provides a consistent comparison
        # across sparse TF-IDF vectors and dense embeddings.
        try:
            scores = cosine_similarity(
                matrix[selected_idx:selected_idx + 1],
                matrix,
            ).ravel()

        except (ValueError, TypeError) as error:
            st.error(f"Could not calculate similarity: {error}")
            st.stop()

        # Identify all rows sharing the selected title.
        selected_title = (
            str(movies.iloc[selected_idx]["original_title"])
            .strip()
            .casefold()
        )

        normalized_titles = (
            movies["original_title"]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.casefold()
        )

        same_title = normalized_titles.eq(selected_title).to_numpy()

        # Exclude the selected movie and same-title entries.
        scores[same_title] = -np.inf

        # Exclude non-finite scores.
        scores[~np.isfinite(scores)] = -np.inf

        # Sort by highest similarity first.
        ranked_indices = np.argsort(-scores, kind="stable")

        top_indices = [
            int(i)
            for i in ranked_indices
            if np.isfinite(scores[i])
            and scores[i] >= 0
        ][:top_n]

        if not top_indices:
            st.warning(
                "No suitable recommendations were found. "
                "Try another movie or recommendation method."
            )
            st.stop()

        # Create results and attach the actual similarity scores.
        results = movies.iloc[top_indices].copy()
        results["Similarity Score"] = scores[top_indices]

        # Persist results across Streamlit reruns so changing
        # other controls does not immediately erase the output.
        st.session_state["moviematch_results"] = results
        st.session_state["moviematch_selected_title"] = (
            get_movie_label(selected_idx)
        )
        st.session_state["moviematch_method"] = method


# =========================================================
# DISPLAY RECOMMENDATIONS
# =========================================================
if "moviematch_results" in st.session_state:

    results = st.session_state["moviematch_results"]
    result_title = st.session_state["moviematch_selected_title"]
    result_method = st.session_state["moviematch_method"]

    st.divider()

    st.subheader(f"🍿 Movies similar to {result_title}")

    st.caption(f"Recommendation method: {result_method}")

    st.info(
        "Similarity scores indicate how closely the selected "
        "model matches each movie. They are not ratings or "
        "probabilities of enjoying a movie."
    )

    # Render attractive movie cards in rows of three.
    for start in range(0, len(results), 3):

        columns = st.columns(3)
        batch = results.iloc[start:start + 3]

        for column, (_, movie) in zip(columns, batch.iterrows()):

            with column:

                with st.container(border=True):

                    title = str(movie["original_title"])
                    st.markdown(f"### {title}")

                    # Genres
                    genres = movie.get("genres_text", None)

                    if (
                        genres is not None
                        and pd.notna(genres)
                        and str(genres).strip()
                    ):
                        st.caption(f"🎭 {genres}")
                    else:
                        st.caption("🎭 Genres unavailable")

                    # Optional release year
                    if "release_date" in movie.index:
                        release_date = movie["release_date"]

                        if pd.notna(release_date):
                            date_text = str(release_date).strip()

                            if (
                                len(date_text) >= 4
                                and date_text[:4].isdigit()
                            ):
                                st.caption(f"📅 {date_text[:4]}")

                    # Overview
                    overview = movie.get("overview", None)

                    if (
                        overview is not None
                        and pd.notna(overview)
                        and str(overview).strip()
                    ):
                        overview = str(overview).strip()

                        if len(overview) > 280:
                            overview = overview[:280].rsplit(
                                " ", 1
                            )[0] + "..."

                        st.write(overview)

                    else:
                        st.write("No overview available.")

                    # Actual model score
                    score = float(movie["Similarity Score"])

                    st.metric(
                        label="Similarity score",
                        value=f"{score:.3f}",
                    )

    # =====================================================
    # RESULTS TABLE AND CSV DOWNLOAD
    # =====================================================
    with st.expander("📊 View all recommendations"):

        display_columns = [
            column
            for column in [
                "original_title",
                "release_date",
                "genres_text",
                "Similarity Score",
            ]
            if column in results.columns
        ]

        st.dataframe(
            results[display_columns].reset_index(drop=True),
            use_container_width=True,
            hide_index=True,
        )

        csv_data = results[display_columns].to_csv(index=False)

        st.download_button(
            label="⬇️ Download recommendations as CSV",
            data=csv_data,
            file_name="moviematch_recommendations.csv",
            mime="text/csv",
            use_container_width=True,
        )

else:
    st.markdown(
        """
        <div style="
            text-align:center;
            padding:35px 15px;
            background:rgba(28,24,39,0.55);
            border:1px solid rgba(255,255,255,0.10);
            border-radius:16px;
            margin-top:20px;
        ">
            <div style="font-size:42px;">🎞️</div>
            <h3>Ready for your next movie?</h3>
            <p style="color:#b6b3c4;">
                Choose a movie and recommendation method above,
                then discover films selected just for you.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
# FOOTER
# =========================================================
st.markdown(
    """
    <div style="
        text-align:center;
        color:#8e899d;
        padding:35px 0 10px 0;
        font-size:12px;
    ">
        MOVIEMATCH · POWERED BY MACHINE LEARNING & DEEP LEARNING
    </div>
    """,
    unsafe_allow_html=True,
)
