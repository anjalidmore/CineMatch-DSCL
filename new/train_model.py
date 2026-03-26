"""
CineMatch - Movie Recommendation Model Training Script
Loads the pre-processed data (from data cleaning notebook),
demonstrates cosine similarity recommendations with confidence scores.

NOTE: The actual training (CountVectorizer + cosine similarity) was
already performed in the 'data cleaning.ipynb' notebook, which produced
movies.pkl and similarity.pkl. This script loads those artifacts.
"""

import pandas as pd
import numpy as np
import pickle
import os

# ─── Configuration ────────────────────────────────────────────────────────────
MOVIES_CSV_PATH = "movies.csv"
SIMILARITY_PKL_PATH = "similarity.pkl"
TOP_N = 5

# Known TMDB genre names (used for genre extraction from tags)
KNOWN_GENRES = [
    "action", "adventure", "animation", "comedy", "crime", "documentary",
    "drama", "family", "fantasy", "history", "horror", "music", "mystery",
    "romance", "sciencefiction", "tvmovie", "thriller", "war", "western",
]
# ──────────────────────────────────────────────────────────────────────────────


def load_model():
    """Load the trained model outputs."""
    print("[1/3] Loading pre-trained model data...")
    movies = pd.read_csv(MOVIES_CSV_PATH)
    similarity = pickle.load(open(SIMILARITY_PKL_PATH, "rb"))
    print(f"       Loaded {len(movies)} movies")
    print(f"       Similarity matrix shape: {similarity.shape}")
    return movies, similarity


def extract_genres_from_tags(tags_str):
    """Extract genre names from the tags string using known genre list."""
    words = tags_str.lower().split()
    return [g for g in KNOWN_GENRES if g in words]


def recommend(movie_title, df, similarity, top_n=TOP_N):
    """
    Get top-N recommendations for a given movie.
    Returns list of (title, confidence_pct, genres) tuples.
    The cosine similarity score serves as the confidence probability.
    """
    try:
        movie_index = df[df["title"] == movie_title].index[0]
    except IndexError:
        print(f"  Movie '{movie_title}' not found!")
        return []

    distances = similarity[movie_index]
    movies_list = sorted(
        list(enumerate(distances)), reverse=True, key=lambda x: x[1]
    )[1 : top_n + 1]

    results = []
    for idx, score in movies_list:
        row = df.iloc[idx]
        confidence = round(score * 100, 2)
        genres = extract_genres_from_tags(str(row.get("tags", "")))
        results.append((row["title"], confidence, genres))

    return results


def main():
    print("=" * 60)
    print("  CineMatch - Model Training / Loading Pipeline")
    print("=" * 60)

    # Load pre-trained model
    movies_df, similarity = load_model()

    # Show dataset summary
    print(f"\n[2/3] Dataset summary:")
    print(f"       Total movies: {len(movies_df)}")
    print(f"       Columns: {list(movies_df.columns)}")

    # Demo recommendations with confidence scores
    print(f"\n[3/3] Testing recommendations with confidence scores...")
    print("=" * 60)

    test_movies = ["Avatar", "Batman Begins", "The Dark Knight Rises"]
    for movie in test_movies:
        print(f"\n  Recommendations for '{movie}':")
        results = recommend(movie, movies_df, similarity)
        for title, confidence, genres in results:
            bar = "█" * int(confidence // 5) + "░" * (20 - int(confidence // 5))
            genre_str = ", ".join(genres[:3]) if genres else "N/A"
            print(f"    {bar} {confidence:6.2f}%  {title}  [{genre_str}]")

    print("\n" + "=" * 60)
    print("  Model loaded and verified successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
