# MovieMatch: Movie Recommendation Capstone

A content-based movie recommender. Pick a movie you like, choose a method, and MovieMatch returns a ranked list of similar movies with similarity scores. It comes with a Streamlit web app.

## How it works

Each movie's text (genres, overview and other fields) is turned into a numeric vector. Recommendations are the movies whose vectors are closest to the chosen movie's, measured with cosine similarity.

Four methods are available:

| Method | Representation | Text used |
|---|---|---|
| TF-IDF, Original Features | Sparse word weights | Original text features |
| TF-IDF, Rich Features | Sparse word weights | Expanded text features |
| Sentence Transformer, Original Features | Dense semantic embeddings | Original text features |
| Sentence Transformer, Rich Features | Dense semantic embeddings | Expanded text features |

To produce results, the app:

1. Computes cosine similarity between the selected movie and every movie.
2. Excludes the selected movie and any entries with the same title.
3. Ranks by score and returns the top 5, 10, 15 or 20.

Similarity scores show how closely the model matches a movie. They are not ratings or probabilities of enjoying it.

## Repository structure

```
.
├── Capstone.ipynb      # Data preparation, feature building, modeling and experiments
├── app.py              # Streamlit app (MovieMatch)
├── requirements.txt    # Python dependencies
└── artifacts/          # Precomputed data and models used by the app
    ├── movies.csv
    ├── tfidf_matrix.npz
    ├── tfidf_rich_matrix.npz
    ├── embeddings.npy
    └── rich_embeddings.npy
```

The four matrices and `movies.csv` must have the same number of rows, in the same movie order. The app checks this on startup and shows an error if they do not match.

## Getting started

```bash
git clone https://github.com/AwendyG/Movie-Recommendation-Capstone.git
cd Movie-Recommendation-Capstone
pip install -r requirements.txt
streamlit run app.py
```

Then open the local URL Streamlit prints (usually http://localhost:8501).

## Using the app

1. Choose a movie from the dropdown (titles show release year and ID to tell duplicates apart).
2. Choose a recommendation method and the number of results.
3. Click **Find Similar Movies**.
4. Browse the movie cards, or open **View all recommendations** to see a table and download it as CSV.

## Regenerating the artifacts

Run `Capstone.ipynb` to rebuild the data and models, and save the outputs into `artifacts/` using the same file names listed above.

## Tech stack

Python, pandas, NumPy, SciPy, scikit-learn (TF-IDF, cosine similarity), sentence-transformers (embeddings), Streamlit.

## Limitations and future work

- Content-based only, so it does not learn from user ratings or viewing history.
- Quality depends on the text available for each movie.
- Ideas: a hybrid score that blends methods, collaborative filtering with user ratings, and evaluation with user feedback.
