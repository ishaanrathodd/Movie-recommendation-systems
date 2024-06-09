import streamlit as st
import pandas as pd
from surprise import Reader, Dataset, SVD
from fuzzywuzzy import process
import numpy as np

# Load datasets
@st.cache_data()  # Cache the loaded data for better performance
def load_data():
    ratings = pd.read_csv('data/ratings.csv')
    movies = pd.read_csv('data/movies_metadata.csv', usecols=['id', 'title'])
    return ratings, movies

ratings, movies = load_data()

# Sample a smaller portion of the ratings data
ratings_sampled = ratings.sample(frac=0.1, random_state=42)  # Sample 10% of the data

# Preprocess movie IDs
movies['id'] = pd.to_numeric(movies['id'], errors='coerce') # convert to float because of NaN
movies = movies.dropna(subset=['id'])  # Drop NaN
movies['id'] = movies['id'].astype(int) # Can be successfully converted to int because no NaN present

# Create mappings between movie IDs and titles
id_to_title = dict(zip(movies['id' ], movies['title'].str.strip()))  
title_to_id = dict(zip(movies['title'].str.strip(), movies['id'])) 

# Load the dataset lazily
reader = Reader(rating_scale=(0.5, 5.0))
data = Dataset.load_from_df(ratings_sampled[['userId', 'movieId', 'rating']], reader)
trainset = data.build_full_trainset()

# Initialize SVD algorithm and train the model
svd = SVD()
svd.fit(trainset)

# Function to compute similarities based on latent factors
def compute_similarities(movie_id, trainset, svd, n_similar=10):
    movie_internal_id = trainset.to_inner_iid(movie_id)
    movie_factors = svd.qi[movie_internal_id]
    similarities = []

    for i in range(len(svd.qi)):
        if i == movie_internal_id:
            continue
        similarity = np.dot(movie_factors, svd.qi[i])
        similarities.append((i, similarity))

    # Sort by similarity
    similarities.sort(key=lambda x: x[1], reverse=True)

    # Get top N similar movies
    top_similar_movies = similarities[:n_similar]

    # Get movie titles, only include IDs that exist in id_to_title
    similar_movies = [(id_to_title[trainset.to_raw_iid(sim[0])], sim[1]) for sim in top_similar_movies if trainset.to_raw_iid(sim[0]) in id_to_title]

    return similar_movies

# Streamlit UI
st.title('Movie Recommendation System')

# Dropdown menu for movie title
movie_title = st.selectbox('Select a movie title:', list(title_to_id.keys()))

# Function to find the closest matching movie title
def find_matching_movie(title):
    return process.extractOne(title, title_to_id.keys())[0]

selected_movie_id = title_to_id.get(movie_title, None)
if selected_movie_id:
    pass
else:
    # Try to find a close match
    closest_match = find_matching_movie(movie_title)
    selected_movie_id = title_to_id.get(closest_match, None)
    if selected_movie_id:
        pass
    else:
        st.write(f"Movie '{movie_title}' not found or not in the training set.")

# Streamlit button to get recommendations
if st.button('Recommend Similar Movies'):
    if selected_movie_id:
        similar_movies = compute_similarities(selected_movie_id, trainset, svd)
        if similar_movies:
            st.write(f"Top similar movies to '{movie_title}':")
            for i in range(len(similar_movies)):
                title, similarity = similar_movies[i]
                st.write(f"{i+1}. {title} (Similarity: {similarity:.4f})")
        else:
            st.write(f"No similar movies found for '{movie_title}'.")
    else:
        st.write(f"Movie '{movie_title}' not found or not in the training set.")
