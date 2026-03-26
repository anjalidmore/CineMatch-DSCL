# CineMatch - Movie Recommendation System
## Detailed Project Report

---

## Table of Contents
1. [Data Collection](#1-data-collection)
2. [Data Preprocessing](#2-data-preprocessing)
3. [Model Development](#3-model-development)
4. [Model Training](#4-model-training)
5. [Classification / Prediction](#5-classification--prediction)
6. [Model Evaluation](#6-model-evaluation)

---

## 1. Data Collection

### Source
The dataset is sourced from **TMDB (The Movie Database)**, containing metadata for **4800+ movies**. The raw dataset includes 20 columns:

| Column | Description |
|--------|-------------|
| `budget` | Production budget (USD) |
| `genres` | JSON list of genre objects |
| `homepage` | Official movie website |
| `id` | TMDB unique identifier |
| `keywords` | JSON list of keyword objects |
| `original_language` | Original language code |
| `original_title` | Original title |
| `overview` | Plot summary text |
| `popularity` | TMDB popularity score |
| `production_companies` | JSON list of production companies |
| `production_countries` | JSON list of countries |
| `release_date` | Release date |
| `revenue` | Box office revenue (USD) |
| `runtime` | Duration in minutes |
| `spoken_languages` | JSON list of languages |
| `status` | Release status |
| `tagline` | Marketing tagline |
| `title` | Movie title |
| `vote_average` | Average user rating |
| `vote_count` | Total number of votes |

### Loading the Data

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")
df = pd.read_csv("movies.csv")
```

---

## 2. Data Preprocessing

### 2.1 Handling Missing Values
Null values were identified and dropped. After cleaning, all 20 columns have **zero null values**.

```python
df.isnull().sum()  # Verified: 0 nulls across all columns
```

### 2.2 Data Type Conversion
- `release_date` was converted to datetime format for proper handling.
- `overview` text was lowercased and stripped of extra whitespace.

```python
df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
df['overview'] = df['overview'].str.lower()
df['overview'] = df['overview'].str.strip()
```

### 2.3 Feature Selection
From 20 columns, only the most relevant features for content-based filtering were retained:

```python
movies = df[['id', 'title', 'overview', 'genres', 'keywords']]
```

### 2.4 JSON Parsing
The `genres` and `keywords` columns contained JSON strings. They were parsed into Python lists:

```python
import ast

def convert(text):
    L = []
    for i in ast.literal_eval(text):
        L.append(i['name'])
    return L

movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)
```

### 2.5 Text Normalization
- Overview was tokenized into word lists
- Multi-word genre/keyword names had spaces removed (e.g., "Science Fiction" -> "ScienceFiction")

```python
movies['overview'] = movies['overview'].apply(lambda x: x.split())
movies['genres'] = movies['genres'].apply(lambda x: [i.replace(" ", "") for i in x])
movies['keywords'] = movies['keywords'].apply(lambda x: [i.replace(" ", "") for i in x])
```

### 2.6 Tags Creation
A unified `tags` column was created by merging overview, genres, and keywords:

```python
movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords']
movies['tags'] = movies['tags'].apply(lambda x: " ".join(x))
```

### 2.7 Final Dataset
The final preprocessed dataset contains **1493 movies** with 3 columns:

| Column | Type | Description |
|--------|------|-------------|
| `id` | int | TMDB movie ID |
| `title` | str | Movie title |
| `tags` | str | Combined text features |

```python
new_df = movies[['id', 'title', 'tags']]
# Shape: (1493, 3)
```

### 2.8 Exploratory Data Analysis
A correlation heatmap was generated to understand relationships between numerical features:

```python
plt.figure(figsize=(8, 6))
corr = df[['budget', 'revenue', 'popularity', 'vote_average', 'vote_count', 'runtime']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title("Correlation Between Movie Features")
plt.show()
```

---

## 3. Model Development

### 3.1 Approach: Content-Based Filtering
CineMatch uses **content-based filtering**, which recommends movies based on the similarity of their content features (plot, genres, keywords) rather than user behavior (collaborative filtering).

### 3.2 Text Vectorization
The `tags` column is converted into numerical vectors using **CountVectorizer** from scikit-learn:

```python
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(new_df['tags']).toarray()
# Shape: (1493, 5000)
```

**Key parameters:**
- `max_features=5000`: Limits vocabulary to 5000 most frequent words
- `stop_words='english'`: Removes common English words (the, is, a, etc.)
- Output: A sparse matrix of shape **(1493 x 5000)** where each row is a movie and each column is a word frequency

### 3.3 Stemming
Before vectorization, **Porter Stemmer** (NLTK) was applied to reduce words to their root form, improving matching accuracy:

```python
import nltk
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

def stem(text):
    y = []
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)

new_df['tags'] = new_df['tags'].apply(stem)
```

**Examples:**
- "loved" -> "love"
- "loving" -> "love"
- "running" -> "run"

### 3.4 Similarity Computation
**Cosine Similarity** measures the cosine of the angle between two movie vectors:

```
cos(A, B) = (A . B) / (||A|| x ||B||)
```

- Score of **1.0** = identical content
- Score of **0.0** = completely unrelated

```python
from sklearn.metrics.pairwise import cosine_similarity

similarity = cosine_similarity(vectors)
# Shape: (1493, 1493)
```

This produces a **1493 x 1493 matrix** where `similarity[i][j]` gives the similarity score between movie `i` and movie `j`.

---

## 4. Model Training

### 4.1 Training Pipeline
The model training is performed in the `data cleaning.ipynb` notebook. The pipeline is:

1. Load raw TMDB dataset
2. Clean and preprocess data
3. Engineer features (tags column)
4. Apply stemming to tags
5. Vectorize using CountVectorizer (5000 features)
6. Compute cosine similarity matrix
7. Save artifacts

### 4.2 Model Serialization
The trained model artifacts are serialized using **Pickle** for fast loading:

```python
import pickle

new_df.to_csv("movies.csv", index=False)
pickle.dump(new_df, open('movies.pkl', 'wb'))
pickle.dump(similarity, open('similarity.pkl', 'wb'))
```

**Saved artifacts:**
| File | Size | Description |
|------|------|-------------|
| `movies.csv` | ~800 KB | Movie metadata (id, title, tags) |
| `similarity.pkl` | ~17 MB | 1493x1493 cosine similarity matrix |

### 4.3 Dynamic Learning (Runtime)
The Streamlit app implements **dynamic score adjustment** at runtime:

```python
# Liked movies boost similarity scores by 15%
for liked in prefs.get("liked", []):
    li = movies_df[movies_df["title"] == liked].index[0]
    scores += sim[li] * 0.15

# Disliked movies reduce similarity scores by 10%
for disliked in prefs.get("disliked", []):
    di = movies_df[movies_df["title"] == disliked].index[0]
    scores -= sim[di] * 0.10
```

User preferences are persisted to `user_prefs.json` between sessions.

---

## 5. Classification / Prediction

### 5.1 Recommendation Function
The core recommendation function ranks all movies by their similarity to the query movie and returns the top N:

```python
def recommend(movie):
    movie_index = new_df[new_df['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(
        list(enumerate(distances)), reverse=True, key=lambda x: x[1]
    )[1:6]
    for i in movies_list:
        print(new_df.iloc[i[0]].title)
```

### 5.2 Sample Predictions

**Query: "Batman Begins"**
| Rank | Recommended Movie | Match Score |
|------|-------------------|-------------|
| 1 | The Dark Knight | ~38% |
| 2 | The Dark Knight Rises | ~35% |
| 3 | Batman v Superman: Dawn of Justice | ~28% |
| 4 | Teenage Mutant Ninja Turtles | ~22% |
| 5 | The Taking of Pelham 1 2 3 | ~20% |

### 5.3 Confidence Scoring
The cosine similarity score is directly used as the **confidence probability**:

```python
def recommend(movie_title, df, similarity, top_n=5):
    movie_index = df[df["title"] == movie_title].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(
        list(enumerate(distances)), reverse=True, key=lambda x: x[1]
    )[1 : top_n + 1]

    results = []
    for idx, score in movies_list:
        confidence = round(score * 100, 2)  # Convert to percentage
        results.append((df.iloc[idx]["title"], confidence))
    return results
```

### 5.4 Watched Movie Handling
Movies marked as "watched" are visually greyed out in the UI but remain in the results for reference. This prevents the user from receiving recommendations they have already seen.

---

## 6. Model Evaluation

### 6.1 Evaluation Strategy
Since this is an **unsupervised recommendation system** (no explicit user ratings), a proxy evaluation strategy is used:

- **Ground Truth**: Genre overlap between query and recommended movies
- **Relevance Criterion**: A recommendation is "relevant" if it shares at least one genre with the query movie
- **Sample Size**: 500 randomly selected movies (seed=42)
- **Top-K**: 5 recommendations per query

### 6.2 Evaluation Code

```python
def compute_metrics(movies, similarity, top_n=5):
    total_recommendations = 0
    relevant_recommendations = 0
    precision_scores = []
    recall_scores = []

    sample_indices = np.random.choice(n_movies, 500, replace=False)

    for movie_idx in sample_indices:
        query_genres = genre_sets[movie_idx]
        recs = get_recommendations(movie_idx, similarity, top_n)

        relevant_count = 0
        for rec_idx, score in recs:
            rec_genres = genre_sets[rec_idx]
            if query_genres & rec_genres:  # shares >= 1 genre
                relevant_count += 1
                relevant_recommendations += 1
            total_recommendations += 1

        # Precision: relevant / recommended
        precision = relevant_count / top_n
        precision_scores.append(precision)

        # Recall: relevant / all same-genre movies
        all_relevant = sum(1 for j in range(n_movies) if j != movie_idx and query_genres & genre_sets[j])
        recall = relevant_count / all_relevant
        recall_scores.append(recall)

    accuracy = relevant_recommendations / total_recommendations
    avg_precision = np.mean(precision_scores)
    avg_recall = np.mean(recall_scores)
    f1 = 2 * avg_precision * avg_recall / (avg_precision + avg_recall)
```

### 6.3 Results

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **Accuracy** | **81.12%** | 81% of recommendations share a genre with the query |
| **Precision** | **81.12%** | Of all recommended movies, 81% are genre-relevant |
| **Recall** | **0.86%** | Of all same-genre movies in the dataset, 0.86% appear in top-5 |
| **F1 Score** | **1.69%** | Harmonic mean of precision and recall |
| **Avg Confidence** | **22.51%** | Average cosine similarity score of recommendations |
| **Movies Evaluated** | **500** | Random sample from 1493 total |
| **Total Recommendations** | **1970** | 500 queries x ~5 recommendations each |

### 6.4 Analysis

**High Precision (81.12%)**: The model is very good at recommending movies that are genre-relevant. 4 out of 5 recommendations typically share at least one genre with the query movie.

**Low Recall (0.86%)**: This is expected and **not a flaw**. With only 5 recommendations returned from a pool of potentially 200-400 same-genre movies, the recall will naturally be very low. For example, if a Drama has 350 other Drama movies in the dataset, retrieving 5 of them yields recall = 5/350 = 1.4%.

**The model is NOT underfitting or overfitting**:
- This is an unsupervised system with no train/test split
- High precision confirms the similarity metric is meaningful
- Low recall is inherent to top-K recommendation (small K, large candidate pool)

**Average Confidence (22.51%)**: This is the raw cosine similarity, which is naturally low for sparse high-dimensional vectors (5000 dimensions). A 22% similarity in this space indicates strong content overlap.

### 6.5 Comparison to Alternative Metrics

For recommendation systems, more appropriate metrics include:
- **NDCG (Normalized Discounted Cumulative Gain)**: Evaluates ranking quality
- **MAP (Mean Average Precision)**: Considers ranking order
- **Hit Rate**: Does the user find at least one relevant item?

The current evaluation using precision/recall with genre overlap provides a reasonable baseline for content-based filtering.

---

## Tech Stack

| Technology | Purpose |
|------------|---------|
| Python 3.12 | Core language |
| Pandas | Data manipulation |
| NumPy | Numerical operations |
| scikit-learn | CountVectorizer, cosine similarity |
| NLTK | Porter Stemmer for text processing |
| Streamlit | Web application framework |
| Pickle | Model serialization |
| Matplotlib / Seaborn | Data visualization (EDA) |

---

## Project Structure

```
CineMatch-DSCL/new/
|-- data cleaning.ipynb    # Full preprocessing + training pipeline
|-- train_model.py         # Standalone model loading + demo script
|-- evaluate_model.py      # Evaluation metrics computation
|-- app.py                 # Streamlit web application
|-- movies.csv             # Processed movie data (1493 x 3)
|-- similarity.pkl         # Cosine similarity matrix (1493 x 1493)
|-- metrics.pkl            # Saved evaluation metrics
|-- user_prefs.json        # User preferences (likes, dislikes, watched, watchlist)
|-- cinematch.log          # Application log file
|-- PROJECT_REPORT.md      # This report
```

---

*Report generated for CineMatch - Content-Based Movie Recommendation System*
