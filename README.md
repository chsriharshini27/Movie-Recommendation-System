# Movie-Recommendation-System

Content-Based Filtering:

Theory: Recommends items similar to those the user liked in the past. It uses item features and compares them to the user's preferences.
Implementation: Compute the similarity between items using features such as genre, director, actors, etc.

#code
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Example: Movie plot similarity
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['plot'])
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

def get_recommendations(title, cosine_sim=cosine_sim):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    movie_indices = [i[0] for i in sim_scores]
    return movies['title'].iloc[movie_indices]


Collaborative Filtering:

Theory: Recommends items based on the preferences of similar users. It can be user-based or item-based.
Implementation: Use techniques like user-item matrices, cosine similarity, and matrix factorization (e.g., SVD).

#code

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

user_ratings = ratings.pivot(index='user_id', columns='movie_id', values='rating').fillna(0)
user_sim = cosine_similarity(user_ratings)
user_sim_df = pd.DataFrame(user_sim, index=user_ratings.index, columns=user_ratings.index)

def get_user_recommendations(user_id, user_sim_df=user_sim_df):
    similar_users = user_sim_df[user_id].sort_values(ascending=False)
    similar_users = similar_users[1:11]
    recommendations = ratings[ratings['user_id'].isin(similar_users.index)]
    recommendations = recommendations.groupby('movie_id')['rating'].mean().sort_values(ascending=False)
    return recommendations.head(10).index.tolist()


Hybrid Methods:

Theory: Combines content-based and collaborative filtering to leverage the strengths of both.
Implementation: Weighted hybrid, switching hybrid, or blending multiple models.

Data Collection:

Source movie data from databases like IMDb, TMDb, or MovieLens.
User interaction data: ratings, watch history, etc.

Data Cleaning:

Handle missing values, normalize ratings, and preprocess text data (e.g., genres).

Feature Engineering:

Extract features like genre, director, actors, release year, etc.
Create user profiles based on their interaction history.

Evaluation:

from sklearn.metrics import mean_squared_error
from math import sqrt

rmse = sqrt(mean_squared_error(actual_ratings, predicted_ratings))
