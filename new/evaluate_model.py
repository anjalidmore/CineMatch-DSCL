"""
CineMatch - Model Evaluation Script
Computes Accuracy, Precision, Recall, F1 Score using genre-overlap as ground truth.

Since the preprocessed data only has 'tags' (overview + genres + keywords combined),
we extract genres from the tags string using a known genre list.
"""

import pandas as pd
import numpy as np
import pickle

# ─── Configuration ────────────────────────────────────────────────────────────
MOVIES_CSV_PATH = "movies.csv"
SIMILARITY_PKL_PATH = "similarity.pkl"
METRICS_PKL_PATH = "metrics.pkl"
TOP_N = 5

# Known TMDB genre names (lowercased, spaces removed — matching notebook preprocessing)
KNOWN_GENRES = [
    "action", "adventure", "animation", "comedy", "crime", "documentary",
    "drama", "family", "fantasy", "history", "horror", "music", "mystery",
    "romance", "sciencefiction", "tvmovie", "thriller", "war", "western",
]
# ──────────────────────────────────────────────────────────────────────────────


def load_model():
    """Load the trained model outputs."""
    print("[1/4] Loading model data...")
    movies = pd.read_csv(MOVIES_CSV_PATH)
    similarity = pickle.load(open(SIMILARITY_PKL_PATH, "rb"))
    print(f"       Loaded {len(movies)} movies, similarity matrix {similarity.shape}")
    return movies, similarity


def extract_genres(tags_str):
    """Extract genre names from the combined tags string."""
    if not isinstance(tags_str, str):
        return set()
    words = tags_str.lower().split()
    return {g for g in KNOWN_GENRES if g in words}


def get_recommendations(movie_idx, similarity, top_n=TOP_N):
    """Get top-N recommended movie indices for a given movie index."""
    distances = similarity[movie_idx]
    movies_list = sorted(
        list(enumerate(distances)), reverse=True, key=lambda x: x[1]
    )[1 : top_n + 1]
    return [(idx, score) for idx, score in movies_list]


def compute_metrics(movies, similarity, top_n=TOP_N):
    """
    Compute evaluation metrics using genre overlap as ground truth.

    A recommendation is "relevant" if the recommended movie shares ≥1 genre
    with the query movie.
    """
    print("[2/4] Computing evaluation metrics...")

    total_recommendations = 0
    relevant_recommendations = 0
    precision_scores = []
    recall_scores = []
    confidence_scores = []

    n_movies = len(movies)
    sample_size = min(n_movies, 500)

    np.random.seed(42)
    sample_indices = np.random.choice(n_movies, sample_size, replace=False)

    # Pre-compute genre sets from tags for all movies
    genre_sets = {}
    for idx in range(n_movies):
        tags = movies.iloc[idx].get("tags", "")
        genre_sets[idx] = extract_genres(str(tags))

    for i, movie_idx in enumerate(sample_indices):
        if (i + 1) % 100 == 0:
            print(f"       Processing movie {i + 1}/{sample_size}...")

        query_genres = genre_sets[movie_idx]
        if not query_genres:
            continue

        recs = get_recommendations(movie_idx, similarity, top_n)

        relevant_count = 0
        for rec_idx, score in recs:
            rec_genres = genre_sets[rec_idx]
            if query_genres & rec_genres:  # shares ≥1 genre
                relevant_count += 1
                relevant_recommendations += 1
            total_recommendations += 1
            confidence_scores.append(score)

        # Precision: fraction of recommended that are relevant
        precision = relevant_count / top_n if top_n > 0 else 0
        precision_scores.append(precision)

        # Recall: fraction of all same-genre movies appearing in top-N
        all_relevant = sum(
            1
            for j in range(n_movies)
            if j != movie_idx and query_genres & genre_sets[j]
        )
        recall = relevant_count / all_relevant if all_relevant > 0 else 0
        recall_scores.append(recall)

    # Aggregate
    accuracy = (
        relevant_recommendations / total_recommendations
        if total_recommendations > 0
        else 0
    )
    avg_precision = np.mean(precision_scores) if precision_scores else 0
    avg_recall = np.mean(recall_scores) if recall_scores else 0
    f1 = (
        2 * avg_precision * avg_recall / (avg_precision + avg_recall)
        if (avg_precision + avg_recall) > 0
        else 0
    )
    avg_confidence = np.mean(confidence_scores) if confidence_scores else 0

    metrics = {
        "accuracy": round(float(accuracy), 4),
        "precision": round(float(avg_precision), 4),
        "recall": round(float(avg_recall), 4),
        "f1_score": round(float(f1), 4),
        "avg_confidence": round(float(avg_confidence), 4),
        "total_movies_evaluated": int(sample_size),
        "total_recommendations": int(total_recommendations),
    }
    return metrics


def save_metrics(metrics):
    """Save metrics to pickle file."""
    print("[3/4] Saving metrics...")
    pickle.dump(metrics, open(METRICS_PKL_PATH, "wb"))
    print(f"       Saved to {METRICS_PKL_PATH}")


def display_metrics(metrics):
    """Display metrics in a formatted table."""
    print("\n" + "=" * 60)
    print("  CineMatch - Model Evaluation Results")
    print("=" * 60)
    print(f"""
  ┌──────────────────────────────────────────────────────┐
  │  Accuracy             : {metrics['accuracy']:.4f}  ({metrics['accuracy']*100:.2f}%)           │
  │  Precision (avg)      : {metrics['precision']:.4f}  ({metrics['precision']*100:.2f}%)           │
  │  Recall (avg)         : {metrics['recall']:.4f}  ({metrics['recall']*100:.2f}%)            │
  │  F1 Score             : {metrics['f1_score']:.4f}  ({metrics['f1_score']*100:.2f}%)            │
  │  Avg Confidence Score : {metrics['avg_confidence']:.4f}  ({metrics['avg_confidence']*100:.2f}%)            │
  ├──────────────────────────────────────────────────────┤
  │  Movies Evaluated     : {metrics['total_movies_evaluated']:<28}│
  │  Total Recommendations: {metrics['total_recommendations']:<28}│
  └──────────────────────────────────────────────────────┘
""")


def main():
    movies, similarity = load_model()
    metrics = compute_metrics(movies, similarity)
    save_metrics(metrics)
    display_metrics(metrics)
    print("[4/4] Evaluation complete!")


if __name__ == "__main__":
    main()
