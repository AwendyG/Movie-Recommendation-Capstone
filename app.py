import base64
import html
import inspect
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
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
# SMALL HELPERS
# =========================================================
def H(markup: str) -> str:
    """Flatten HTML to one line.

    Streamlit's markdown treats indented lines and blank lines as code
    blocks / block breaks, which silently breaks custom HTML.
    """
    return " ".join(line.strip() for line in markup.strip().splitlines() if line.strip())


def esc(value) -> str:
    """Escape text for safe, single-line HTML."""
    return html.escape(" ".join(str(value).split()), quote=True)


def _stretch_kwargs(func):
    params = inspect.signature(func).parameters
    if "width" in params:
        return {"width": "stretch"}
    if "use_container_width" in params:
        return {"use_container_width": True}
    return {}


BUTTON_STRETCH = _stretch_kwargs(st.button)
DATAFRAME_STRETCH = _stretch_kwargs(st.dataframe)
DOWNLOAD_STRETCH = _stretch_kwargs(st.download_button)

BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"


# =========================================================
# CINEMATIC STYLING
# =========================================================
st.markdown(
    H(
        """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Inter:wght@400;600;800&display=swap');

html, body, .stApp, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp {
    background:
        radial-gradient(1200px 600px at 15% -10%, rgba(229,9,20,0.18), transparent 60%),
        radial-gradient(1000px 600px at 95% 10%, rgba(88,60,200,0.20), transparent 60%),
        linear-gradient(135deg, #07070c 0%, #14101c 55%, #090d16 100%);
    color: #f5f5f7;
}
header[data-testid="stHeader"] { background: transparent; }
footer { visibility: hidden; }
.block-container { padding-top: 1.2rem; padding-bottom: 3rem; max-width: 1400px; }

h1, h2, h3 { color: #ffffff !important; font-weight: 800 !important; }
p, label { color: #e4e0ec; }

.stButton > button {
    background: linear-gradient(90deg, #e50914, #b20710);
    color: white; border: none; border-radius: 12px;
    padding: 0.75rem 1.4rem; font-weight: 800; letter-spacing: .3px;
    box-shadow: 0 8px 24px rgba(229,9,20,.35);
    transition: all .2s ease;
}
.stButton > button:hover {
    background: #ff2631; color: white; transform: translateY(-2px);
    box-shadow: 0 12px 30px rgba(255,38,49,.5); border: none;
}
div[data-baseweb="select"] > div {
    background-color: #201d2a; border-color: #40384d; border-radius: 10px;
}
div[data-testid="stDataFrame"] {
    border: 1px solid #393343; border-radius: 12px; overflow: hidden;
}
[data-testid="stCaptionContainer"] { color: #b6b3c4; }

/* Tabs */
button[data-baseweb="tab"] {
    font-weight: 800; font-size: 16px; color: #b6b3c4;
    padding: 12px 18px;
}
button[data-baseweb="tab"][aria-selected="true"] { color: #ff4d5a; }
div[data-baseweb="tab-highlight"] { background-color: #e50914 !important; height: 3px; }

/* HERO */
.hero {
    position: relative; overflow: hidden; text-align: center;
    padding: 64px 28px 54px 28px; border-radius: 26px; margin: 8px 0 22px 0;
    background:
        linear-gradient(180deg, rgba(8,6,14,.80), rgba(8,6,14,.94)),
        url('https://images.unsplash.com/photo-1489599849927-2ee91cede3ba?auto=format&fit=crop&w=1600&q=80');
    background-size: cover; background-position: center;
    border: 1px solid rgba(255,255,255,.14);
    box-shadow: 0 30px 80px rgba(0,0,0,.55), inset 0 0 120px rgba(229,9,20,.12);
}
.hero::before {
    content: ""; position: absolute; inset: -40% -10% auto -10%; height: 140%;
    background: conic-gradient(from 160deg at 50% 0%,
        transparent 0 38%, rgba(255,255,255,.13) 44%, transparent 50%,
        transparent 56%, rgba(255,209,102,.11) 62%, transparent 68% 100%);
    animation: sweep 9s ease-in-out infinite alternate; pointer-events: none;
}
.bulbs {
    position: absolute; left: 0; right: 0; height: 14px; opacity: .95;
    background-image: radial-gradient(circle, #ffd166 0 3px, rgba(255,209,102,0) 5px);
    background-size: 30px 14px; animation: twinkle 1.4s ease-in-out infinite alternate;
}
.bulbs.top { top: 8px; } .bulbs.bottom { bottom: 8px; animation-delay: .7s; }
.hero-label { position: relative; color: #ff5964; font-size: 12px; letter-spacing: 6px; font-weight: 800; }
.hero-title {
    position: relative; font-family: 'Bebas Neue', sans-serif;
    font-size: clamp(64px, 12vw, 150px); line-height: .95; letter-spacing: .06em;
    margin: 10px 0 8px 0;
    background: linear-gradient(90deg, #ffffff, #ff4d5a, #ffd166, #ffffff);
    background-size: 300% 100%; -webkit-background-clip: text; background-clip: text;
    -webkit-text-fill-color: transparent; color: transparent;
    animation: shimmer 7s linear infinite;
    filter: drop-shadow(0 6px 30px rgba(229,9,20,.45));
}
.hero-sub { position: relative; color: #e0dce8; font-size: 17px; line-height: 1.7; max-width: 640px; margin: 0 auto 22px auto; }
.chips { position: relative; display: flex; flex-wrap: wrap; gap: 10px; justify-content: center; }
.chip {
    padding: 8px 16px; border-radius: 999px; font-size: 13px; font-weight: 700;
    background: rgba(255,255,255,.08); border: 1px solid rgba(255,255,255,.18);
    backdrop-filter: blur(6px);
}
.chip b { color: #ffd166; }

/* SECTION TITLES */
.sec { font-family: 'Bebas Neue', sans-serif; font-size: 34px; letter-spacing: 3px; margin: 26px 0 6px 0; color: #fff; }
.sec small { font-family: 'Inter', sans-serif; font-size: 13px; letter-spacing: 0; color: #b6b3c4; margin-left: 10px; }
.sec::after { content: ""; display: block; height: 3px; width: 70px; margin-top: 6px; background: linear-gradient(90deg, #e50914, transparent); }

/* MARQUEE (film strip) */
.filmstrip {
    position: relative; overflow: hidden; padding: 26px 0; border-radius: 14px;
    background: #0a0910; border: 1px solid rgba(255,255,255,.08);
}
.filmstrip::before, .filmstrip::after {
    content: ""; position: absolute; left: 0; right: 0; height: 10px; z-index: 3;
    background: repeating-linear-gradient(90deg, #1d1a26 0 14px, transparent 14px 26px);
}
.filmstrip::before { top: 8px; } .filmstrip::after { bottom: 8px; }
.marquee-track { display: flex; gap: 16px; width: max-content; animation: scroll 70s linear infinite; }
.filmstrip:hover .marquee-track { animation-play-state: paused; }

.movie-tile {
    position: relative; flex: 0 0 180px; height: 270px; overflow: hidden;
    border-radius: 12px; border: 1px solid rgba(255,255,255,.14); background: #201d2a;
    transition: transform .3s ease, box-shadow .3s ease;
}
.movie-tile:hover { transform: translateY(-6px) scale(1.04); box-shadow: 0 14px 34px rgba(229,9,20,.4); z-index: 2; }
.movie-tile img { position: absolute; inset: 0; width: 100%; height: 100%; object-fit: cover; }
.movie-overlay {
    position: absolute; inset: 0; display: flex; flex-direction: column; justify-content: flex-end;
    padding: 12px; background: linear-gradient(180deg, transparent 45%, rgba(8,7,13,.97) 100%);
}
.movie-overlay span { color: #ff6872; font-size: 10px; letter-spacing: 2px; font-weight: 800; }
.movie-overlay h3 { margin: 4px 0 0 0; font-size: 15px; line-height: 1.2; }
.strip { display: flex; gap: 14px; overflow-x: auto; padding: 10px 2px 18px 2px; scrollbar-width: thin; scrollbar-color: #e50914 #201d2a; }

/* Placeholder layer (sits behind every poster image) */
.ph {
    position: absolute; inset: 0; display: flex; align-items: center; justify-content: center;
    font-size: 54px; background: linear-gradient(160deg, #2a2438, #14101c);
}

/* SPOTLIGHT */
.spot {
    position: relative; display: flex; gap: 28px; padding: 26px; border-radius: 22px; overflow: hidden;
    border: 1px solid rgba(255,255,255,.14); background: #120f1a; margin: 8px 0 12px 0;
}
.spot::before {
    content: ""; position: absolute; inset: -30px; background-image: var(--bg);
    background-size: cover; background-position: center; filter: blur(30px) brightness(.42) saturate(1.4);
}
.spot > * { position: relative; }
.spot-poster { position: relative; flex: 0 0 190px; aspect-ratio: 2/3; border-radius: 14px; overflow: hidden; box-shadow: 0 24px 60px rgba(0,0,0,.65); }
.spot-poster img { position: absolute; inset: 0; width: 100%; height: 100%; object-fit: cover; }
.spot-info { flex: 1; min-width: 0; }
.spot-info .kicker { color: #ff5964; font-size: 11px; letter-spacing: 4px; font-weight: 800; }
.spot-info h2 { font-family: 'Bebas Neue', sans-serif; font-size: clamp(34px, 5vw, 56px); letter-spacing: 2px; margin: 4px 0 8px 0; line-height: 1; }
.gchip { display: inline-block; margin: 0 8px 8px 0; padding: 4px 12px; border-radius: 999px; font-size: 12px; font-weight: 700; background: rgba(229,9,20,.22); border: 1px solid rgba(255,89,100,.5); }
.spot-info p { color: #e0dce8; font-size: 14.5px; line-height: 1.7; max-width: 760px; }

/* RESULT CARDS */
.rec-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(205px, 1fr)); gap: 22px; margin-top: 10px; }
.rcard {
    border-radius: 16px; overflow: hidden; background: rgba(28,24,39,.82);
    border: 1px solid rgba(255,255,255,.10); animation: rise .6s both; animation-delay: var(--d);
    transition: transform .25s ease, box-shadow .25s ease;
}
.rcard:hover { transform: translateY(-8px) scale(1.02); box-shadow: 0 20px 44px rgba(229,9,20,.32); }
.poster-wrap { position: relative; aspect-ratio: 2/3; overflow: hidden; background: #201d2a; }
.poster-wrap img { position: absolute; inset: 0; width: 100%; height: 100%; object-fit: cover; transition: transform .45s ease; }
.rcard:hover .poster-wrap img { transform: scale(1.09); }
.rank {
    position: absolute; top: 10px; left: 10px; z-index: 2; font-family: 'Bebas Neue', sans-serif;
    font-size: 24px; line-height: 1; padding: 4px 11px; border-radius: 8px;
    background: rgba(229,9,20,.94); box-shadow: 0 4px 14px rgba(0,0,0,.5);
}
.pop {
    position: absolute; inset: 0; z-index: 1; padding: 16px; display: flex; align-items: flex-end;
    background: linear-gradient(180deg, transparent 8%, rgba(8,7,13,.97) 68%);
    opacity: 0; transition: opacity .3s ease;
}
.rcard:hover .pop { opacity: 1; }
.pop p { margin: 0; font-size: 12.5px; line-height: 1.5; color: #ece8f5;
    display: -webkit-box; -webkit-line-clamp: 8; -webkit-box-orient: vertical; overflow: hidden; }
.rbody { padding: 14px 14px 16px 14px; }
.rtitle { font-weight: 800; font-size: 15px; line-height: 1.25; color: #fff; min-height: 38px; }
.rmeta { color: #b6b3c4; font-size: 12px; margin: 4px 0 10px 0; min-height: 16px; }
.bar { height: 6px; border-radius: 99px; background: rgba(255,255,255,.10); overflow: hidden; }
.bar i { display: block; height: 100%; border-radius: 99px; background: linear-gradient(90deg, #e50914, #ffd166); }
.rscore { display: flex; justify-content: space-between; align-items: baseline; gap: 8px; margin-top: 8px; }
.rscore b { color: #ffd166; font-size: 14px; }
.rscore span { color: #9d98ad; font-size: 11px; text-align: right; }

.empty { text-align: center; padding: 38px 15px; background: rgba(28,24,39,.55);
    border: 1px dashed rgba(255,255,255,.18); border-radius: 16px; margin-top: 18px; }
.empty .big { font-size: 46px; }
.footer { text-align: center; color: #8e899d; padding: 38px 0 10px 0; font-size: 12px; letter-spacing: 3px; }

@keyframes scroll { from { transform: translateX(0); } to { transform: translateX(-50%); } }
@keyframes shimmer { from { background-position: 0% 0; } to { background-position: 300% 0; } }
@keyframes twinkle { from { opacity: .35; } to { opacity: 1; } }
@keyframes sweep { from { transform: rotate(-6deg); } to { transform: rotate(6deg); } }
@keyframes rise { from { opacity: 0; transform: translateY(26px); } to { opacity: 1; transform: translateY(0); } }

@media (max-width: 700px) {
    .spot { flex-direction: column; align-items: center; text-align: center; }
    .movie-tile { flex-basis: 140px; height: 210px; }
}
</style>
"""
    ),
    unsafe_allow_html=True,
)


# =========================================================
# LOAD MOVIE DATA
# =========================================================
@st.cache_data
def load_movies():
    movies_path = ARTIFACTS_DIR / "movies.csv"

    if not movies_path.exists():
        raise FileNotFoundError(f"Missing file: {movies_path}")

    movies = pd.read_csv(movies_path)

    for column in ["id", "original_title"]:
        if column not in movies.columns:
            raise ValueError(f"Missing required column: {column}")

    movies["id"] = pd.to_numeric(movies["id"], errors="raise").astype(int)

    # Row order must match the model matrices, so keep it and make the
    # index positional (0..n-1).
    return movies.reset_index(drop=True)


# =========================================================
# LOAD SAVED CONTENT MODELS
# =========================================================
@st.cache_resource
def load_models():
    required_files = [
        "tfidf_matrix.npz",
        "tfidf_rich_matrix.npz",
        "embeddings.npy",
        "rich_embeddings.npy",
    ]

    missing = [f for f in required_files if not (ARTIFACTS_DIR / f).is_file()]
    if missing:
        raise FileNotFoundError("Missing model artifacts: " + ", ".join(missing))

    tfidf_matrix = load_npz(ARTIFACTS_DIR / "tfidf_matrix.npz").tocsr()
    tfidf_rich_matrix = load_npz(ARTIFACTS_DIR / "tfidf_rich_matrix.npz").tocsr()
    embeddings = np.load(ARTIFACTS_DIR / "embeddings.npy", mmap_mode="r", allow_pickle=False)
    rich_embeddings = np.load(ARTIFACTS_DIR / "rich_embeddings.npy", mmap_mode="r", allow_pickle=False)

    return tfidf_matrix, tfidf_rich_matrix, embeddings, rich_embeddings


@st.cache_data
def normalized_titles():
    movies = load_movies()
    return (
        movies["original_title"].fillna("").astype(str).str.strip().str.casefold().to_numpy()
    )


# =========================================================
# SVD ASSETS (Surprise model + training data), loaded safely
# =========================================================
def _norm_id(value) -> str:
    """Normalise ids so 12, 12.0 and '12' all match."""
    try:
        return str(int(float(value)))
    except (TypeError, ValueError):
        return str(value)


@st.cache_resource(show_spinner="Loading SVD model...")
def load_svd_assets():
    model_path = ARTIFACTS_DIR / "ndcg_tuned_svd.pkl"
    train_path = ARTIFACTS_DIR / "train.csv"

    if not model_path.exists():
        raise FileNotFoundError(f"Missing SVD model: {model_path}")
    if not train_path.exists():
        raise FileNotFoundError(f"Missing training data: {train_path}")

    algo = joblib.load(model_path)

    for attr in ("pu", "qi", "trainset"):
        if not hasattr(algo, attr):
            raise ValueError(
                "ndcg_tuned_svd.pkl is not a fitted Surprise SVD model "
                f"(missing '{attr}'). Save the fitted model, e.g. "
                "grid_search.best_estimator['rmse'] after fitting."
            )

    train = pd.read_csv(train_path)
    for column in ["userId", "tmdb_id", "rating"]:
        if column not in train.columns:
            raise ValueError(f"Missing training column: {column}")

    movies = load_movies()
    ts = algo.trainset

    user_map = {_norm_id(ts.to_raw_uid(i)): i for i in ts.all_users()}
    item_map = {_norm_id(ts.to_raw_iid(i)): i for i in ts.all_items()}

    inner = np.array([item_map.get(_norm_id(i), -1) for i in movies["id"]], dtype=int)

    pop_counts = train["tmdb_id"].map(_norm_id).value_counts()
    pop_cnt = movies["id"].map(lambda i: pop_counts.get(_norm_id(i), 0)).to_numpy(dtype=float)
    pop_norm = np.log1p(pop_cnt) / max(np.log1p(pop_cnt.max()), 1.0)

    user_counts = train["userId"].value_counts()

    id_to_pos = {}
    for pos in range(len(movies) - 1, -1, -1):
        id_to_pos[int(movies["id"].iat[pos])] = pos

    return {
        "algo": algo,
        "train": train,
        "user_map": user_map,
        "inner": inner,
        "pop_cnt": pop_cnt,
        "pop_norm": pop_norm,
        "user_counts": user_counts,
        "id_to_pos": id_to_pos,
    }


def svd_predict(assets, user_id):
    """Predicted rating (0.5-5.0) for every movie, for one user."""
    algo = assets["algo"]
    ts = algo.trainset
    inner = assets["inner"]
    known = inner >= 0
    biased = getattr(algo, "biased", True)

    inner_u = assets["user_map"].get(_norm_id(user_id))

    est = np.full(len(inner), ts.global_mean if biased else 0.0, dtype=float)

    if biased and inner_u is not None:
        est += algo.bu[inner_u]
    if biased:
        est[known] += algo.bi[inner[known]]
    if inner_u is not None:
        est[known] += algo.qi[inner[known]] @ algo.pu[inner_u]

    return np.clip(est, 0.5, 5.0), inner_u is not None


def user_rated_positions(assets, user_id):
    train = assets["train"]
    ids = train.loc[train["userId"] == user_id, "tmdb_id"]
    id_to_pos = assets["id_to_pos"]
    return {id_to_pos[int(float(i))] for i in ids if int(float(i)) in id_to_pos}


# =========================================================
# POSTERS (column in CSV -> TMDB API -> placeholder)
# =========================================================
TMDB_IMG = "https://image.tmdb.org/t/p/"
POSTER_COLUMNS = ("poster_path", "poster_url", "poster")


@st.cache_resource
def poster_cache():
    """Plain dict {movie_id: poster_path_or_None}, shared across reruns."""
    return {}


def get_api_key():
    key = st.session_state.get("tmdb_key_input", "").strip()
    if key:
        return key
    try:
        key = st.secrets.get("TMDB_API_KEY", None)
        if key:
            return str(key)
    except Exception:
        pass
    return os.environ.get("TMDB_API_KEY")


def _fetch_poster_path(movie_id, api_key):
    try:
        import requests

        response = requests.get(
            f"https://api.themoviedb.org/3/movie/{int(movie_id)}",
            params={"api_key": api_key},
            timeout=6,
        )
        if response.status_code == 200:
            return response.json().get("poster_path") or ""
    except Exception:
        return None
    return None


def _column_poster(movie, size):
    for column in POSTER_COLUMNS:
        if column in movie.index:
            value = movie[column]
            if pd.notna(value) and str(value).strip():
                value = str(value).strip()
                if value.startswith("http"):
                    return value
                if not value.startswith("/"):
                    value = "/" + value
                return f"{TMDB_IMG}{size}{value}"
    return None


def resolve_posters(rows: pd.DataFrame, size="w342") -> dict:
    """Return {movie_id: url_or_None} for the given rows."""
    cache = poster_cache()
    api_key = get_api_key()
    out, todo = {}, []

    for _, movie in rows.iterrows():
        movie_id = int(movie["id"])
        url = _column_poster(movie, size)

        if url:
            out[movie_id] = url
        elif movie_id in cache:
            path = cache[movie_id]
            out[movie_id] = f"{TMDB_IMG}{size}{path}" if path else None
        else:
            todo.append(movie_id)

    todo = list(dict.fromkeys(todo))

    if todo and api_key:
        with ThreadPoolExecutor(max_workers=8) as pool:
            paths = list(pool.map(lambda i: _fetch_poster_path(i, api_key), todo))

        for movie_id, path in zip(todo, paths):
            if path is not None:  # None = request failed, retry next time
                cache[movie_id] = path
            out[movie_id] = f"{TMDB_IMG}{size}{path}" if path else None
    else:
        for movie_id in todo:
            out[movie_id] = None

    return out


# =========================================================
# HTML BUILDERS
# =========================================================
def year_of(movie):
    if "release_date" in movie.index and pd.notna(movie["release_date"]):
        text = str(movie["release_date"]).strip()
        if len(text) >= 4 and text[:4].isdigit():
            return text[:4]
    return ""


def genres_of(movie):
    genres = movie.get("genres_text", None)
    if genres is not None and pd.notna(genres) and str(genres).strip():
        return str(genres).replace("|", ", ")
    return ""


def img_tag(url):
    if not url:
        return ""
    return f'<img src="{esc(url)}" alt="" loading="lazy" referrerpolicy="no-referrer">'


def tile_html(movie, url, tag):
    return (
        '<div class="movie-tile"><div class="ph">🎬</div>'
        f"{img_tag(url)}"
        f'<div class="movie-overlay"><span>{esc(tag)}</span>'
        f"<h3>{esc(movie['original_title'])}</h3></div></div>"
    )


def marquee_html(sample, posters):
    tiles = []
    for _, movie in sample.iterrows():
        genres = genres_of(movie)
        tag = genres.split(",")[0].upper() if genres else "FEATURED"
        tiles.append(tile_html(movie, posters.get(int(movie["id"])), tag))

    track = "".join(tiles) * 2  # duplicated for a seamless infinite loop
    return H(f'<div class="filmstrip"><div class="marquee-track">{track}</div></div>')


def strip_html(items):
    return H(f'<div class="strip">{"".join(items)}</div>')


def spotlight_html(movie, url):
    bg = f"--bg:url('{esc(url)}');" if url else ""
    poster = f'<div class="spot-poster"><div class="ph">🎬</div>{img_tag(url)}</div>'

    genres = genres_of(movie)
    chips = "".join(f'<span class="gchip">{esc(g.strip())}</span>' for g in genres.split(",") if g.strip())

    year = year_of(movie)
    overview = movie.get("overview", None)
    overview = str(overview).strip() if overview is not None and pd.notna(overview) else ""
    if len(overview) > 420:
        overview = overview[:420].rsplit(" ", 1)[0] + "..."

    return H(
        f"""
        <div class="spot" style="{bg}">
            {poster}
            <div class="spot-info">
                <div class="kicker">NOW SPOTLIGHTING{' · ' + esc(year) if year else ''}</div>
                <h2>{esc(movie['original_title'])}</h2>
                <div>{chips}</div>
                <p>{esc(overview) if overview else 'No overview available.'}</p>
            </div>
        </div>
        """
    )


def cards_html(df, posters):
    cards = []

    for rank, (_, movie) in enumerate(df.iterrows(), start=1):
        url = posters.get(int(movie["id"]))
        overview = movie.get("overview", None)
        overview = str(overview).strip() if overview is not None and pd.notna(overview) else "No overview available."

        meta = " · ".join(x for x in [year_of(movie), genres_of(movie)] if x)
        frac = float(np.clip(movie["_frac"], 0, 1))

        cards.append(
            f"""
            <div class="rcard" style="--d:{min(rank, 12) * 0.06:.2f}s">
                <div class="poster-wrap">
                    <div class="ph">🎬</div>
                    {img_tag(url)}
                    <div class="rank">#{rank}</div>
                    <div class="pop"><p>{esc(overview)}</p></div>
                </div>
                <div class="rbody">
                    <div class="rtitle">{esc(movie['original_title'])}</div>
                    <div class="rmeta">{esc(meta)}</div>
                    <div class="bar"><i style="width:{frac * 100:.0f}%"></i></div>
                    <div class="rscore"><b>{esc(movie['_label'])}</b><span>{esc(movie['_extra'])}</span></div>
                </div>
            </div>
            """
        )

    return H(f'<div class="rec-grid">{"".join(cards)}</div>')


def section(title, sub=""):
    st.markdown(
        H(f'<div class="sec">{esc(title)}<small>{esc(sub)}</small></div>'),
        unsafe_allow_html=True,
    )


# =========================================================
# SIDEBAR - poster settings
# =========================================================
with st.sidebar:
    st.header("🖼️ Poster settings")
    st.caption(
        "Posters load from a `poster_path` column in movies.csv. If there "
        "isn't one, paste a free TMDB API key (themoviedb.org) to fetch them."
    )
    st.text_input("TMDB API key (optional)", type="password", key="tmdb_key_input")


# =========================================================
# LOAD EVERYTHING (content models are required)
# =========================================================
try:
    movies = load_movies()
    tfidf_matrix, tfidf_rich_matrix, embeddings, rich_embeddings = load_models()

    shapes = {
        "Original TF-IDF": tfidf_matrix.shape,
        "Rich TF-IDF": tfidf_rich_matrix.shape,
        "Original embeddings": embeddings.shape,
        "Rich embeddings": rich_embeddings.shape,
    }

    if any(len(s) != 2 for s in shapes.values()):
        raise ValueError("Every model artifact must be a 2D matrix.")

    row_counts = {name: s[0] for name, s in shapes.items()}
    row_counts["Movie dataset"] = len(movies)

    if len(set(row_counts.values())) != 1:
        raise ValueError(
            f"Artifact row counts do not match: {row_counts}. "
            "Regenerate the artifacts using the same dataset and movie ordering."
        )

except (FileNotFoundError, ValueError, OSError) as error:
    st.error(f"MovieMatch could not load its data: {error}")
    st.info("Ensure movies.csv and all four model artifacts are inside the artifacts folder.")
    st.stop()

# SVD is optional: if it can't load, the other tabs still work.
svd_assets, svd_error = None, None
try:
    svd_assets = load_svd_assets()
except Exception as error:  # surprise missing, bad pickle, missing file, ...
    svd_error = str(error)

METHODS = {
    "TF-IDF — Original Features": tfidf_matrix,
    "TF-IDF — Rich Features": tfidf_rich_matrix,
    "Sentence Transformer — Original Features": embeddings,
    "Sentence Transformer — Rich Features": rich_embeddings,
}
NORM_TITLES = normalized_titles()


# =========================================================
# HERO
# =========================================================
st.markdown(
    H(
        f"""
        <div class="hero">
            <div class="bulbs top"></div>
            <div class="hero-label">★ YOUR PERSONAL CINEMA ★</div>
            <div class="hero-title">MOVIEMATCH</div>
            <div class="hero-sub">Every movie leads to another. Discover your next obsession with
            recommendations powered by NLP, deep learning and collaborative filtering.</div>
            <div class="chips">
                <span class="chip"><b>{len(movies):,}</b> films</span>
                <span class="chip"><b>4</b> content engines</span>
                <span class="chip"><b>SVD</b> personalization</span>
                <span class="chip"><b>Hybrid</b> brain</span>
            </div>
            <div class="bulbs bottom"></div>
        </div>
        """
    ),
    unsafe_allow_html=True,
)


# =========================================================
# POSTER MARQUEE
# =========================================================
if "carousel_indices" not in st.session_state:
    pool_size = min(30, len(movies))
    st.session_state["carousel_indices"] = movies.sample(n=pool_size).index.tolist()

carousel_pool = movies.loc[st.session_state["carousel_indices"]]
carousel_posters = resolve_posters(carousel_pool)

with_posters = carousel_pool[carousel_pool["id"].map(lambda i: bool(carousel_posters.get(int(i))))]
carousel_sample = (with_posters if len(with_posters) >= 6 else carousel_pool).head(14)

section("NOW SHOWING", "hover to pause")
st.markdown(marquee_html(carousel_sample, carousel_posters), unsafe_allow_html=True)

if len(with_posters) == 0:
    st.warning(
        ""
    )


# =========================================================
# MOVIE PICKER + SPOTLIGHT
# =========================================================
valid_indices = movies.index[
    movies["original_title"].notna() & movies["original_title"].astype(str).str.strip().ne("")
].tolist()

if not valid_indices:
    st.error("No valid movie titles were found in the dataset.")
    st.stop()


def get_movie_label(index):
    movie = movies.iloc[index]
    title = str(movie["original_title"]).strip()
    year = year_of(movie)
    if year:
        title = f"{title} ({year})"
    return f"{title} · ID {movie['id']}"


section("PICK YOUR MOVIE", "the film you love")
selected_idx = st.selectbox(
    "🎥 Choose a movie",
    options=valid_indices,
    format_func=get_movie_label,
    key="selected_movie",
)

selected_movie = movies.iloc[selected_idx]
selected_poster = resolve_posters(movies.iloc[[selected_idx]], size="w500")
st.markdown(spotlight_html(selected_movie, selected_poster.get(int(selected_movie["id"]))), unsafe_allow_html=True)


# =========================================================
# RECOMMENDATION HELPERS
# =========================================================
def content_scores(idx, matrix):
    scores = cosine_similarity(matrix[idx : idx + 1], matrix).ravel().astype(float)
    scores[NORM_TITLES == NORM_TITLES[idx]] = -np.inf  # drop the movie itself + same-title rows
    scores[~np.isfinite(scores)] = -np.inf
    return scores


def top_positions(scores, k):
    order = np.argsort(-scores, kind="stable")
    return [int(i) for i in order if np.isfinite(scores[i])][:k]


def make_results(positions, frac, label, extra):
    df = movies.iloc[positions].copy()
    df["_frac"] = frac
    df["_label"] = label
    df["_extra"] = extra
    return df


def render_results(key):
    if key not in st.session_state:
        st.markdown(
            H(
                """
                <div class="empty"><div class="big">🎞️</div>
                <h3>Ready for your next movie?</h3>
                <p style="color:#b6b3c4;">Set your options above and hit the button.</p></div>
                """
            ),
            unsafe_allow_html=True,
        )
        return

    saved = st.session_state[key]
    df = saved["df"]

    section(saved["title"], saved["caption"])
    st.info(
        "Scores show how strongly the model favours each movie. "
        "They are not ratings or guarantees you'll enjoy it."
    )

    posters = resolve_posters(df, size="w342")
    st.markdown(cards_html(df, posters), unsafe_allow_html=True)

    with st.expander("📊 View as table / download CSV"):
        cols = [c for c in ["original_title", "release_date", "genres_text", "_label", "_extra"] if c in df.columns]
        table = df[cols].rename(columns={"_label": "Score", "_extra": "Details"}).reset_index(drop=True)
        st.dataframe(table, hide_index=True, **DATAFRAME_STRETCH)
        st.download_button(
            "⬇️ Download as CSV",
            data=table.to_csv(index=False),
            file_name=f"moviematch_{key}.csv",
            mime="text/csv",
            key=f"dl_{key}",
            **DOWNLOAD_STRETCH,
        )


def user_selector(prefix):
    """Pick a user from the most active ones, or type any userId."""
    counts = svd_assets["user_counts"]
    top_users = counts.head(1000).index.tolist()

    c1, c2 = st.columns([2, 1])
    with c1:
        chosen = st.selectbox(
            "👤 Pick a user (most active first)",
            options=top_users,
            format_func=lambda u: f"User {u} · {int(counts[u])} ratings",
            key=f"{prefix}_user",
        )
    with c2:
        typed = st.text_input("…or type a userId", key=f"{prefix}_user_typed")

    if typed.strip():
        match = counts.index[counts.index.map(_norm_id) == _norm_id(typed.strip())]
        if len(match):
            return match[0]
        st.warning("That userId isn't in the training data, using the picker instead.")

    return chosen


def taste_profile(user_id):
    train = svd_assets["train"]
    id_to_pos = svd_assets["id_to_pos"]

    liked = train.loc[train["userId"] == user_id].sort_values("rating", ascending=False)

    positions, ratings = [], []
    for tmdb_id, rating in zip(liked["tmdb_id"], liked["rating"]):
        pos = id_to_pos.get(int(float(tmdb_id)))
        if pos is not None and pos not in positions:
            positions.append(pos)
            ratings.append(rating)
        if len(positions) == 8:
            break

    if not positions:
        return

    rows = movies.iloc[positions]
    posters = resolve_posters(rows)
    tiles = [tile_html(m, posters.get(int(m["id"])), f"★ {r:g}") for (_, m), r in zip(rows.iterrows(), ratings)]

    st.caption("🎬 Taste profile: this user's top-rated films")
    st.markdown(strip_html(tiles), unsafe_allow_html=True)


def svd_unavailable():
    st.warning(
        "The SVD model couldn't be loaded, so this tab is disabled. "
        f"Reason: {svd_error}"
    )
    st.caption(
        "Needs `scikit-surprise` in requirements.txt, plus `ndcg_tuned_svd.pkl` "
        "and `train.csv` in the artifacts folder."
    )


# =========================================================
# TABS
# =========================================================
tab_sim, tab_svd, tab_hybrid = st.tabs(["🎞️ Similar Movies", "👤 For You (SVD)", "⚡ Hybrid"])


# ---------- TAB 1: content-based ----------
with tab_sim:
    section("SIMILAR MOVIES", "content-based")

    c1, c2 = st.columns([3, 1])
    with c1:
        method = st.selectbox(
            "🧠 Recommendation method",
            list(METHODS.keys()),
            key="sim_method",
            help=(
                "TF-IDF compares word-based text features. Sentence Transformers "
                "compare semantic meaning. Rich features use the expanded text."
            ),
        )
    with c2:
        top_n = st.slider("How many?", 5, 20, 10, 5, key="sim_n")

    if st.button("✨ Find Similar Movies", type="primary", key="sim_go", **BUTTON_STRETCH):
        with st.spinner("Searching for your next favorite movie..."):
            try:
                scores = content_scores(selected_idx, METHODS[method])
            except (ValueError, TypeError) as error:
                st.error(f"Could not calculate similarity: {error}")
                st.stop()

            scores[scores < 0] = -np.inf
            positions = top_positions(scores, top_n)

            if not positions:
                st.warning("No suitable recommendations were found. Try another movie or method.")
            else:
                vals = scores[positions]
                st.session_state["res_sim"] = {
                    "title": "MOVIES LIKE THIS",
                    "caption": f"{get_movie_label(selected_idx)} · {method}",
                    "df": make_results(
                        positions,
                        vals,
                        [f"{v * 100:.0f}% similar" for v in vals],
                        [f"cosine {v:.3f}" for v in vals],
                    ),
                }
                st.toast("🍿 Your watchlist is ready!")

    render_results("res_sim")


# ---------- TAB 2: SVD ----------
with tab_svd:
    section("PICKS FOR YOU", "collaborative filtering (SVD)")

    if svd_assets is None:
        svd_unavailable()
    else:
        st.caption(
            "Predicts what a specific user would rate each movie, using the "
            "tuned SVD model, no selected movie needed."
        )
        svd_user = user_selector("svd")
        taste_profile(svd_user)

        c1, c2, c3 = st.columns([2, 1, 1])
        with c1:
            svd_disc = st.slider(
                "🔭 Discovery boost (more popular or less popular?)",
                0.0, 1.0, 0.3, 0.05, key="svd_disc",
                help="0 = pure predicted rating. Higher values push popular, "
                     "heavily-rated movies down so lesser-known films can surface.",
            )
        with c2:
            svd_n = st.slider("How many?", 5, 20, 10, 5, key="svd_n")
        with c3:
            svd_hide = st.checkbox("Hide already-rated", value=True, key="svd_hide")

        if st.button("🎯 Recommend for this user", type="primary", key="svd_go", **BUTTON_STRETCH):
            with st.spinner("Predicting ratings..."):
                est, known_user = svd_predict(svd_assets, svd_user)
                adjusted = est - svd_disc * svd_assets["pop_norm"]

                if svd_hide:
                    for pos in user_rated_positions(svd_assets, svd_user):
                        adjusted[pos] = -np.inf

                positions = top_positions(adjusted, svd_n)
                pop_cnt = svd_assets["pop_cnt"]

                st.session_state["res_svd"] = {
                    "title": "PICKS FOR YOU",
                    "caption": f"User {svd_user} · discovery {svd_disc:.2f}"
                               + ("" if known_user else " · new user (item bias only)"),
                    "df": make_results(
                        positions,
                        (est[positions] - 0.5) / 4.5,
                        [f"★ {est[p]:.2f} predicted" for p in positions],
                        [f"{int(pop_cnt[p])} ratings" for p in positions],
                    ),
                }
                st.toast("🎯 Personalized picks are in!")

        render_results("res_svd")


# ---------- TAB 3: hybrid ----------
with tab_hybrid:
    section("HYBRID", "collaborative filtering + content similarity")

    if svd_assets is None:
        svd_unavailable()
    else:
        st.caption(
            "Combines the user's SVD-predicted ratings with content similarity "
            "to movies they rated highly (4 stars or above). No movie selection is needed."
        )

        hy_user = user_selector("hy")

        c1, c2 = st.columns(2)
        with c1:
            # The tested configuration performed best with 60% SVD + 40% content.
            svd_weight = st.slider(
                "⚖️ SVD weight",
                0.0, 1.0, 0.6, 0.05, key="hy_svd_weight",
                help="0.6 means 60% collaborative filtering and 40% content similarity.",
            )
        with c2:
            hy_n = st.slider("How many?", 5, 20, 10, 5, key="hy_n")

        hy_hide = st.checkbox("Hide already-rated movies", value=True, key="hy_hide")

        if st.button("⚡ Build my hybrid picks", type="primary", key="hy_go", **BUTTON_STRETCH):
            with st.spinner("Blending your rating patterns and movie content..."):
                # SVD predicted ratings for every movie in the catalogue.
                est, known_user = svd_predict(svd_assets, hy_user)

                # Only recommend movies that the fitted SVD model knows.
                candidate_mask = svd_assets["inner"] >= 0
                candidate_positions = np.flatnonzero(candidate_mask)

                # Remove movies this user has already rated.
                rated_positions = user_rated_positions(svd_assets, hy_user)
                if hy_hide and rated_positions:
                    candidate_positions = np.array(
                        [p for p in candidate_positions if p not in rated_positions],
                        dtype=int,
                    )

                # Build the user's content profile from movies rated >= 4.
                user_ratings = svd_assets["train"]
                liked_ids = user_ratings.loc[
                    (user_ratings["userId"] == hy_user)
                    & (user_ratings["rating"] >= 4),
                    "tmdb_id",
                ].tolist()

                id_to_pos = svd_assets["id_to_pos"]
                liked_positions = list(dict.fromkeys(
                    id_to_pos[int(float(movie_id))]
                    for movie_id in liked_ids
                    if int(float(movie_id)) in id_to_pos
                ))

                if len(candidate_positions) == 0:
                    st.warning("No eligible unseen movies were found for this user.")
                elif len(liked_positions) == 0:
                    st.warning(
                        "This user has no training movies rated 4 or above. "
                        "Showing SVD-only recommendations instead."
                    )
                    adjusted = est[candidate_positions]
                    min_svd, max_svd = adjusted.min(), adjusted.max()
                    svd_norm = (
                        (adjusted - min_svd) / (max_svd - min_svd)
                        if max_svd > min_svd else np.ones_like(adjusted)
                    )
                    blend = svd_norm
                    content_norm = np.zeros_like(blend)
                else:
                    # Embeddings are normalized, so their dot product is cosine similarity.
                    candidate_vectors = np.asarray(embeddings[candidate_positions])
                    liked_vectors = np.asarray(embeddings[liked_positions])
                    content_scores_user = (candidate_vectors @ liked_vectors.T).mean(axis=1)

                    # Min-max normalize both scores over the same candidate set.
                    c_min, c_max = content_scores_user.min(), content_scores_user.max()
                    content_norm = (
                        (content_scores_user - c_min) / (c_max - c_min)
                        if c_max > c_min else np.ones_like(content_scores_user)
                    )

                    svd_scores = est[candidate_positions]
                    s_min, s_max = svd_scores.min(), svd_scores.max()
                    svd_norm = (
                        (svd_scores - s_min) / (s_max - s_min)
                        if s_max > s_min else np.ones_like(svd_scores)
                    )

                    blend = (
                        svd_weight * svd_norm
                        + (1.0 - svd_weight) * content_norm
                    )

                if len(candidate_positions) > 0:
                    order = np.argsort(-blend, kind="stable")[:hy_n]
                    positions = candidate_positions[order]

                    st.session_state["res_hy"] = {
                        "title": "HYBRID PICKS",
                        "caption": (
                            f"User {hy_user} · {svd_weight:.0%} SVD + "
                            f"{1.0 - svd_weight:.0%} content"
                            + ("" if known_user else " · new user")
                        ),
                        "df": make_results(
                            positions,
                            blend[order],
                            [f"{score * 100:.0f}% hybrid" for score in blend[order]],
                            [
                                f"SVD ★{est[p]:.2f} · content {content_norm[i]:.2f}"
                                for i, p in zip(order, positions)
                            ],
                        ),
                    }
                    st.toast("⚡ Hybrid recommendations ready!")

        render_results("res_hy")


# =========================================================
# FOOTER
# =========================================================
st.markdown(
    H('<div class="footer">MOVIEMATCH · CONTENT-BASED · SVD · HYBRID · MADE FOR MOVIE LOVERS</div>'),
    unsafe_allow_html=True,
)