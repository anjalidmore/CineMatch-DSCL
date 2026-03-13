import streamlit as st
import pickle
import pandas as pd

# Load data
import pandas as pd
movies = pd.read_csv("movies.csv")
similarity = pickle.load(open('similarity.pkl','rb'))

# Recommender function
def recommend(movie):

    movie_index = movies[movies['title'] == movie].index[0]
    distances = similarity[movie_index]

    movies_list = sorted(list(enumerate(distances)),
                         reverse=True,
                         key=lambda x: x[1])[1:6]

    recommended_movies = []

    for i in movies_list:
        recommended_movies.append(movies.iloc[i[0]].title)

    return recommended_movies


# Page config
st.set_page_config(page_title="Movie Recommender", layout="wide")


# Background styling
page_bg = """
<style>
.stApp {
background-image: url("https://images.unsplash.com/photo-1489599849927-2ee91cede3ba");
background-size: cover;
background-position: center;
}
</style>
"""

st.markdown(page_bg, unsafe_allow_html=True)


# Title
st.title("🎬 Movie Recommender System")

st.write("Find movies similar to your favourite ones!")


# Dropdown
selected_movie = st.selectbox(
"Select a movie",
movies['title'].values
)


# Button
if st.button("Recommend"):

    recommendations = recommend(selected_movie)

    st.subheader("Recommended Movies")

    col1, col2, col3, col4, col5 = st.columns(5)

    cols = [col1, col2, col3, col4, col5]

    for col, movie in zip(cols, recommendations):
        col.write(movie)