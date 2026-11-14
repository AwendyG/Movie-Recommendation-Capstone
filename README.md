# 🎬 MovieMatch: Hybrid Movie Recommender

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Open%20the%20App-FF4B4B?logo=streamlit&logoColor=white)](https://capstoneproject-8rshd6mcwd8wg8qettjnjr.streamlit.app/)

👉 **Try it live: [MovieMatch on Streamlit](https://capstoneproject-8rshd6mcwd8wg8qettjnjr.streamlit.app/)**

A movie recommendation system that combines **content-based filtering** (what a movie is about) with **collaborative filtering** (what similar users enjoyed), served through a **Streamlit** app.

Built as a capstone project on 3,959 movies from TMDB and 610 users from MovieLens.

A movie recommendation system that combines **content-based filtering** (what a movie is about) with **collaborative filtering** (what similar users enjoyed), served through a **Streamlit** app.

Built as a capstone project on 3,959 movies from TMDB and 610 users from MovieLens.

---

## Table of Contents
- [Overview](#overview)
- [Key Results](#key-results)
- [Data](#data)
- [Methodology](#methodology)
- [Evaluation](#evaluation)
- [Findings and Limitations](#findings-and-limitations)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Future Work](#future-work)
- [Author](#author)

---

## Overview

The project answers one question: **does personalisation beat simply recommending what is popular?**

It was built in two phases:

1. **Content-based recommender:** movies are described by their overview, genres, keywords, tagline and production companies, then compared using TF-IDF and sentence embeddings.
2. **Collaborative filtering and hybrid:** real user ratings are added, an SVD model learns user taste, and its scores are blended with content similarity.

## Key Results

Evaluated on 357 users, each with one held-out movie they rated 4.0 or higher. A "hit" means that movie appears in the user's top 10.

| Model | HitRate@10 | NDCG@10 | Median rank |
|---|---|---|---|
| Random (expected) | 0.32% | 0.0015 | 1,577 |
| **Most popular** | **8.40%** | **0.0404** | 273 |
| SVD (tuned for rating error) | 2.52% | 0.0096 | 626 |
| SVD (default settings) | 3.08% | 0.0139 | 467 |
| SVD (tuned for NDCG@10) | 4.48% | 0.0231 | 406 |
| Content-based only | 1.40% | 0.0093 | 1,096 |
| **Hybrid (60% SVD + 40% content)** | **6.72%** | **0.0353** | 394 |

**Rating prediction (RMSE, lower is better):** baseline 0.9469, SVD 0.9033, tuned SVD 0.8910.

The 60/40 hybrid is the best personalised model, with NDCG@10 about 53% higher than SVD alone. The "most popular" list still leads overall (see [Findings and Limitations](#findings-and-limitations)).

## Data

| Source | Used for |
|---|---|
| [TMDB 5000 Movies](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata) | Movie metadata: overview, genres, keywords, tagline, companies, runtime |
| [MovieLens `ml-latest-small`](https://grouplens.org/datasets/movielens/latest/) | 67,579 matched ratings from 610 users |

**Data preparation**
- Dropped the `homepage` column; removed rows missing `tagline`, `release_date` or `overview`.
- Median-imputed missing or zero `runtime`; parsed JSON columns (genres, keywords, companies) into lists.
- Filled 3 empty genre lists manually.
- Linked MovieLens to TMDB through `links.csv`: **3,206 of 3,959 movies (81%)** matched, with 0 ID mismatches. After normalising titles, 96.3% match and 99.9% of release years agree within one year.
- Kept movies with **5 or more ratings** (2,070 movies, 65,098 ratings).

## Methodology

### 1. Content-based filtering
- **TF-IDF:** 20,000 features, 1–2 word phrases, cosine similarity.
- **Sentence embeddings:** `all-MiniLM-L6-v2` (384 dimensions) on the combined movie text.
- **Richer metadata:** production companies added to the text, and both methods rebuilt.

### 2. Collaborative filtering
- **Split:** each user's most recent rating is held out for testing (610 tests); the rest is training (66,969 ratings).
- **Baseline:** global mean + user bias + movie bias, clipped to 0.5–5.0.
- **SVD** (Surprise library) with grid-search tuning, then a second search that chose settings by **NDCG@10** instead of RMSE (50 factors, regularisation 0.05, learning rate 0.002).

### 3. Hybrid recommender
```
final_score = α × SVD_score + (1 − α) × content_score
```
- `content_score`: mean embedding similarity to movies the user rated 4 or higher.
- Both scores are min-max scaled to 0–1 before blending.
- Tested α from 0 to 1; **α = 0.6** performed best.

## Evaluation

- **RMSE / MAE:** how far predicted ratings are from real ones.
- **HitRate@10:** share of users whose held-out liked movie appears in their top 10.
- **NDCG@10:** rewards liked movies that rank nearer the top.
- **Median rank:** typical position of the held-out movie (lower is better).
- **Reference points:** random and "most popular" recommenders.
- **Error analysis:** results split by movie popularity and user activity, rating-level error, and case studies of individual users.

## Findings and Limitations

- **Lower rating error did not mean better top-10 lists.** The RMSE-tuned SVD had the best RMSE but worse ranking than the default SVD, which is why the final model was tuned on NDCG@10.
- **Popularity bias.** Every model performed far better on well-rated movies. For the least-rated half of held-out movies, hit rates were close to 0%.
- **"Most popular" is a strong benchmark.** The dataset is small and skewed (median 8 ratings per movie), which favours popular titles. It beats the hybrid overall, so the hybrid is reported as the best *personalised* model.
- **Blend weight chosen on the test set.** α = 0.6 is the best of the values tested, not proven optimal; a separate validation split is needed to confirm it.
- **Modest test size.** 357 users means small differences should be read cautiously.
- **Coverage.** About 19% of catalog movies have no MovieLens ratings, so they rely on content similarity alone.

## Project Structure

```
Movie-Recommendation-Capstone/
├── Capstone.ipynb        # Full analysis: cleaning, models, evaluation, hybrid
├── app.py                # Streamlit app (rename to match your file)
├── requirements.txt
└── artifacts/            # Saved outputs created by the notebook
    ├── movies.csv
    ├── tfidf_matrix.npz
    ├── tfidf_rich_matrix.npz
    ├── embeddings.npy
    ├── rich_embeddings.npy
    ├── ndcg_tuned_svd.pkl
    └── train.csv
```

## Getting Started

### 1. Clone and install
```bash
git clone https://github.com/AwendyG/Movie-Recommendation-Capstone.git
cd Movie-Recommendation-Capstone
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Add the data
Download the datasets listed under [Data](#data) and update the file paths in the first cells of `Capstone.ipynb`.

### 3. Rebuild the artifacts (optional)
Run `Capstone.ipynb` from top to bottom. It saves the files in `artifacts/` that the app loads.

### 4. Launch the app
```bash
streamlit run app.py
```

### Requirements
`pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `scipy`, `sentence-transformers`, `scikit-surprise`, `joblib`, `streamlit`

## Future Work
- Add popularity-aware re-ranking so lesser-known movies get a fair chance.
- Train on a larger ratings dataset and add cold-start handling for new users.

## Author
**AwendyG**: [github.com/AwendyG](https://github.com/AwendyG)

## Acknowledgements
Movie data from [TMDB](https://www.themoviedb.org/) and ratings from [GroupLens MovieLens](https://grouplens.org/datasets/movielens/). This product uses the TMDB API but is not endorsed or certified by TMDB.
