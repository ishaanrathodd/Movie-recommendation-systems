import streamlit as st
import pandas as pd
import numpy as np
from ast import literal_eval
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem.snowball import SnowballStemmer

# Load datasets
credits = pd.read_csv('data/credits.csv')
keywords = pd.read_csv('data/keywords.csv')
links_small = pd.read_csv('data/links_small.csv')
md = pd.read_csv('data/movies_metadata.csv')
ratings = pd.read_csv('data/ratings_small.csv')

# Preprocess data
md['genres'] = md['genres'].fillna('[]').apply(literal_eval).apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
md['year'] = pd.to_datetime(md['release_date'], errors='coerce').apply(lambda x: str(x).split('-')[0] if pd.notnull(x) else np.nan)
links_small = links_small[links_small['tmdbId'].notnull()]['tmdbId'].astype('int')

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
smd = md[md['id'].isin(links_small)]

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
smd['soup'] = smd['soup'].apply(lambda x: ' '.join(x))

# Join movie descriptions and taglines to the soup (because of different data type)
smd['soup'] = smd.apply(lambda x: x['soup'] + ' ' + ' '.join(x['tagline'].split()) + ' ' + ' '.join(x['description'].split()), axis=1)

# Build Count matrix
count = CountVectorizer(analyzer='word', ngram_range=(1, 2), stop_words='english')
count_matrix = count.fit_transform(smd['soup'])

# Compute cosine similarity
cosine_sim = cosine_similarity(count_matrix, count_matrix)
smd = smd.reset_index()
titles = smd['title']
indices = pd.Series(smd.index, index=smd['title'])


def weighted_rating(x, m, C):
    v = x['vote_count']
    R = x['vote_average']
    return (v/(v+m) * R) + (m/(m+v) * C)

# Recommendation function
def recommendations(title):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:26]
    movie_indices = [i[0] for i in sim_scores]
    
    movies = smd.iloc[movie_indices][['title', 'vote_count', 'vote_average', 'year']]
    vote_counts = movies[movies['vote_count'].notnull()]['vote_count'].astype('int')
    vote_averages = movies[movies['vote_average'].notnull()]['vote_average'].astype('int')
    C = vote_averages.mean()
    m = vote_counts.quantile(0.60)
    qualified = movies[(movies['vote_count'] >= m) & (movies['vote_count'].notnull()) & 
                       (movies['vote_average'].notnull())]
    qualified['vote_count'] = qualified['vote_count'].astype('int')
    qualified['vote_average'] = qualified['vote_average'].astype('int')
    qualified['wr'] = qualified.apply(weighted_rating, args=(m, C), axis=1)
    qualified = qualified.sort_values('wr', ascending=False).head(10)
    return qualified

# Streamlit UI
st.title('Movie Recommendation System')

# Dropdown menu for movie title
movie_title = st.selectbox('Select a movie title:', titles)

if movie_title:
    recommendations = recommendations(movie_title)
    st.write(f'Recommendations for **{movie_title}**:')
    for i in range(len(recommendations)):
        rec = recommendations.iloc[i]
        st.write(f"{i+1}. {rec.title} (Year: {rec.year}, Vote Count: {rec.vote_count}, Vote Average: {rec.vote_average})")