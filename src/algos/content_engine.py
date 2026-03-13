import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import os

def train_content_model(df):
    """
    Turns movie descriptions into a math matrix and saves it.
    """
    # 1. Create a 'soup' of keywords (Genres + Overview)
    # Ensure no NaN values
    df['genres'] = df['genres'].fillna('')
    df['overview'] = df['overview'].fillna('')
    df['soup'] = df['genres'] + " " + df['overview']

    # 2. Vectorize the text (TF-IDF)
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['soup'])

    # 3. Calculate Cosine Similarity
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    # 4. Save the model to /models folder for efficiency
    if not os.path.exists('models'):
        os.makedirs('models')
    
    joblib.dump(cosine_sim, 'models/content_sim.pkl')
    print("✅ Content Model saved to models/content_sim.pkl")
    return cosine_sim

def get_content_recommendations(title, df, cosine_sim):
    # Get the index of the movie that matches the title
    try:
        idx = df[df['title'] == title].index[0]
        # Get pairwise similarity scores
        sim_scores = list(enumerate(cosine_sim[idx]))
        # Sort by similarity
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        # Get top 5 (excluding the movie itself)
        movie_indices = [i[0] for i in sim_scores[1:6]]
        return df['title'].iloc[movie_indices].tolist()
    except IndexError:
        return ["Movie not found in database"]