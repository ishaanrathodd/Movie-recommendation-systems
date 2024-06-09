import streamlit as st
import pandas as pd
import numpy as np
from surprise import Reader, Dataset, SVD
from fuzzywuzzy import process
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from ast import literal_eval
from nltk.stem.snowball import SnowballStemmer

# Load datasets
@st.cache_data()
def load_data():
    credits = pd.read_csv('data/credits.csv')
    keywords = pd.read_csv('data/keywords.csv')
    links_small = pd.read_csv('data/links_small.csv')
    md = pd.read_csv('data/movies_metadata.csv')
    ratings = pd.read_csv('data/ratings_small.csv')
    return credits, keywords, links_small, md, ratings

credits, keywords, links_small, md, ratings = load_data()

# Preprocess data
md['genres'] = md['genres'].fillna('[]').apply(literal_eval).apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
md['year'] = pd.to_datetime(md['release_date'], errors='coerce').apply(lambda x: str(x).split('-')[0] if pd.notnull(x) else np.nan)
links_small = links_small.dropna(subset=['tmdbId'])  # Drop rows with NaN tmdbId
links_small['tmdbId'] = links_small['tmdbId'].astype('int')

def convert_int(x):
    try:
        return int(x)
    except:
        return np.nan

md['id'] = md['id'].apply(convert_int)
md = md.drop([19730, 29503, 35587])
md['id'] = md['id'].astype('int')

# Merge datasets
keywords['id'] = keywords['id'].astype('int')
credits['id'] = credits['id'].astype('int')
md['id'] = md['id'].astype('int')
md = md.merge(credits, on='id')
md = md.merge(keywords, on='id')
smd = md[md['id'].isin(links_small['tmdbId'])]

# Parse data
smd['cast'] = smd['cast'].apply(literal_eval)
smd['crew'] = smd['crew'].apply(literal_eval)
smd['keywords'] = smd['keywords'].apply(literal_eval)
smd['cast_size'] = smd['cast'].apply(lambda x: len(x))
smd['crew_size'] = smd['crew'].apply(lambda x: len(x))

# Extract director
def get_director(x):
    for i in x:
        if i['job'] == 'Director':
            return i['name']
    return np.nan

smd['director'] = smd['crew'].apply(get_director)

# Process cast, director, and keywords
smd['tagline'] = smd['tagline'].fillna('')
smd['description'] = smd['overview'] + ' ' + smd['tagline']
smd['description'] = smd['description'].fillna('')
smd['cast'] = smd['cast'].apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
smd['cast'] = smd['cast'].apply(lambda x: x[:3] if len(x) >= 3 else x)
smd['keywords'] = smd['keywords'].apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
smd['cast'] = smd['cast'].apply(lambda x: [str.lower(i.replace(" ", "")) for i in x])
smd['director'] = smd['director'].astype('str').apply(lambda x: str.lower(x.replace(" ", "")))
smd['director'] = smd['director'].apply(lambda x: [x, x, x])

# Filter keywords
stemmer = SnowballStemmer('english')
s = smd.apply(lambda x: pd.Series(x['keywords']), axis=1).stack().reset_index(level=1, drop=True)
s.name = 'keyword'
s = s.value_counts()
s = s[s > 1]

def filter_keywords(x):
    words = []
    for i in x:
        if i in s:
            words.append(i)
    return words

smd['keywords'] = smd['keywords'].apply(filter_keywords)
smd['keywords'] = smd['keywords'].apply(lambda x: [stemmer.stem(i) for i in x])
smd['keywords'] = smd['keywords'].apply(lambda x: [str.lower(i.replace(" ", "")) for i in x])

# Create soup
smd['soup'] = smd['keywords'] + smd['cast'] + smd['director'] + smd['genres']
smd['soup'] = smd['soup'].apply(lambda x: ' '.join(map(str, x)))  # Convert dictionaries to strings

# Build TF-IDF matrix
tfidf = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), stop_words='english')
tfidf_matrix = tfidf.fit_transform(smd['soup'])

# Compute cosine similarity
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
smd = smd.reset_index()
titles = smd['title']
indices = pd.Series(smd.index, index=smd['title'])

# Create mappings between movie IDs and titles
id_to_title = dict(zip(smd['id'], smd['title'].str.strip()))
title_to_id = dict(zip(smd['title'].str.strip(), smd['id']))

# Function to find the closest matching movie title
def find_matching_movie(title):
    return process.extractOne(title, title_to_id.keys())[0]

# Load the dataset lazily
reader = Reader(rating_scale=(0.5, 5.0))
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
trainset = data.build_full_trainset()

# Initialize SVD algorithm and train the model
svd = SVD()
svd.fit(trainset)

# Preprocess id_map for hybrid function
id_map = links_small[['movieId', 'tmdbId']]
id_map.columns = ['movieId', 'id']
id_map = id_map.merge(smd[['title', 'id']], on='id').set_index('title')
indices_map = id_map.set_index('id')

# Hybrid recommendation function
def hybrid(userId, title):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[int(idx)]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:26]
    movie_indices = [i[0] for i in sim_scores]
    movies = smd.iloc[movie_indices][['title', 'vote_count', 'vote_average', 'release_date', 'id']]
    movies['est'] = movies['id'].apply(lambda x: svd.predict(userId, indices_map.loc[x]['movieId']).est)
    movies = movies.sort_values('est', ascending=False)
    return movies.head(10).reset_index(drop=True)


def calculate_genre_percentage(userId):
    # Get movies rated by the user
    user_movies = ratings[ratings['userId'] == userId]
    user_movie_ids = user_movies['movieId'].tolist()
    
    # Get genres of user's rated movies
    user_genres = smd[smd['id'].isin(user_movie_ids)]['genres'].sum()
    
    # Count genre occurrences
    genre_counts = pd.Series(user_genres).value_counts(normalize=True) * 100
    
    return genre_counts


# Streamlit UI
st.title('Movie Recommendation System')

# Dropdown menu for movie title
movie_title = st.selectbox('Select a movie title:', titles)

# Print selected movie title and ID
selected_movie_id = title_to_id.get(movie_title, None)
if selected_movie_id:
    pass
else:
    closest_match = find_matching_movie(movie_title)
    selected_movie_id = title_to_id.get(closest_match, None)
    if selected_movie_id:
        st.write(f"Selected Movie: {closest_match} - ID: {selected_movie_id}")
    else:
        st.write(f"Movie '{movie_title}' not found or not in the training set.")

# User input for userId
user_id = st.number_input('Enter your User ID:', min_value=1, step=1)

# Calculate genre interests for the selected user
user_ratings = ratings[ratings['userId'] == user_id]
user_movies = smd[smd['id'].isin(user_ratings['movieId'])]
genre_counts = user_movies['genres'].apply(pd.Series).stack().value_counts(normalize=True)
genre_table = pd.DataFrame({'Genre': genre_counts.index, 'Interest': genre_counts.values})
genre_table['Interest'] = genre_table['Interest'].map(lambda x: f"{x:.2%}")


# Button to recommend movies

if st.button('Recommend Movies'):
    if selected_movie_id:
        hybrid_recommendations = hybrid(user_id, movie_title)
        if not hybrid_recommendations.empty:
            hybrid_recommendations = hybrid_recommendations.reset_index(drop=True)
            st.write(f"Top hybrid recommendations for user {user_id} and movie '{movie_title}':")
            for i, rec in hybrid_recommendations.iterrows():
                st.write(f"{i+1}. {rec['title']} (Release Date: {rec['release_date']}, "
                            f"Vote Count: {rec['vote_count']}, Vote Average: {rec['vote_average']:.2f}, "
                            f"Estimated Rating: {rec['est']:.2f})")
        else:
            st.write(f"No hybrid recommendations found for '{movie_title}'.")
    else:
        st.write(f"Movie '{movie_title}' not found or not in the training set.")

# Button to show genre interests
if st.button('Show Genre Interest'):
    genre_percentage = calculate_genre_percentage(user_id)
    genre_percentage.columns = ['Interest']
    st.write(f"\nGenre interest percentage for User {user_id}:")
    st.table(genre_percentage)