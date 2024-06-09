# Movie Recommendation System


<img width="1710" alt="image" src="https://github.com/ishaanrathodd/Movie-recommendation-systems/assets/90642857/edfd1d57-45e1-4186-8ea5-6b995ba16c55">


This repository contains implementations of various movie recommendation algorithms including Content-Based, Collaborative Filtering, and Hybrid methods. These algorithms are implemented using Python and popular libraries such as scikit-learn, Surprise, and Streamlit.

## Overview

Movie recommendation systems are widely used to suggest movies to users based on their preferences. This repository provides implementations of different recommendation algorithms:

- **Content-Based Filtering**: Recommends movies similar to a given movie based on their features such as genres, cast, and keywords.
- **Collaborative Filtering**: Recommends movies to a user based on the preferences of similar users or items.
- **Hybrid Filtering**: Combines content-based and collaborative filtering methods to provide personalized recommendations.

## Algorithms

### Content-Based Filtering

Content-based filtering recommends items similar to what the user has liked in the past. It uses item features to make recommendations. In this repository, content-based filtering is implemented using:

- **CountVectorizer**: Converts text data (movie descriptions, genres, cast, etc.) into a matrix of token counts.
- **TF-IDF Vectorizer**: Converts text data into TF-IDF (Term Frequency-Inverse Document Frequency) vectors to reflect the importance of words in documents.

### Collaborative Filtering

Collaborative filtering recommends items based on the preferences of similar users or items. In this repository, collaborative filtering is implemented using:

- **Singular Value Decomposition (SVD)**: Decomposes the user-item interaction matrix to find latent factors representing user preferences and item features.

### Hybrid Filtering

Hybrid filtering combines content-based and collaborative filtering methods to provide more accurate and personalized recommendations. In this repository, hybrid filtering is implemented by combining content-based and collaborative filtering algorithms.

## Types of dataset

- The full dataset: This dataset consists of 26,000,000 ratings and 750,000 tag applications applied to 45,000 movies by 270,000 users. Includes tag genome data with 12 million relevance scores across 1,100 tags.
- The small dataset: This dataset comprises of 100,000 ratings and 1,300 tag applications applied to 9,000 movies by 700 users.

## Dataset
- Download the dataset using [this link](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset?resource=download)
- Put the csv files inside `data` folder.
- Following files are present in the dataset:
  ```
  credits.csv
  keywords.csv
  links.csv
  links_small.csv
  movies_metadata.csv
  ratings.csv
  ratings_small.csv
  ```

## Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/movie-recommendation-system.git

2. Install the required dependencies:
   
   ```bash
   pip install -r requirements.txt

## Usage

1. Run the Streamlit app:

   ```bash
   streamlit run app.py

2. Your code would start its execution in your default browser

## Resources

- Matrix factorization: https://www.youtube.com/watch?v=ZspR5PZemcs
- Cosine Similarity: https://www.youtube.com/watch?v=e9U0QAFbfLI
- Resources for SVD:
  - http://nicolas-hug.com/blog/matrix_facto_1
  - http://nicolas-hug.com/blog/matrix_facto_2
  - http://nicolas-hug.com/blog/matrix_facto_3
  - http://nicolas-hug.com/blog/matrix_facto_4
  - http://sifter.org/simon/journal/20061211.html

## Credits

Special thanks to [Jalaj Thanaki](https://github.com/jalajthanaki) for the codebase. I've built a Streamlit wrapper around it and incorporated personal optimizations.
