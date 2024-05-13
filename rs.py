import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

movies = pd.read_csv('ml-1m/ml-1m/movies.dat', sep='::',
                     engine='python', header=None, names=['movie_id', 'title', 'genre'], encoding='latin1')

ratings = pd.read_csv('ml-1m/ml-1m/ratings.dat', sep='::',
                      engine='python', header=None, names=['user_id', 'movie_id', 'rating', 'timestamp'])

ratings['user_id'] = pd.factorize(ratings['user_id'])[0]

movie_ratings = pd.merge(movies, ratings, on='movie_id')

user_movie_matrix = movie_ratings.pivot_table(index='user_id', columns='title', values='rating')

user_movie_matrix = user_movie_matrix.fillna(0)

user_similarity = cosine_similarity(user_movie_matrix)
def recommend_movies(user_id, num_recommendations=5):
    user_ratings = user_movie_matrix.loc[user_id].values.reshape(1, -1)
    similarity_scores = user_similarity[user_id]
    similar_users = similarity_scores.argsort()[::-1][1:]  # 除去用户本身
    recommendations = []
    for similar_user in similar_users:
        similar_user_ratings = user_movie_matrix.iloc[similar_user].values.reshape(1, -1)
        non_zero_indices = user_ratings != 0
        predicted_ratings = similar_user_ratings[non_zero_indices] * similarity_scores[similar_user]
        for idx, rating in enumerate(predicted_ratings):
            recommendations.append((idx, rating))
    recommendations = sorted(recommendations, key=lambda x: x[1], reverse=True)[:num_recommendations]
    recommended_movie_indices = [idx for idx, _ in recommendations]
    recommended_movies = user_movie_matrix.columns[recommended_movie_indices]
    return recommended_movies

user_id = 0
recommendations = recommend_movies(user_id)
print(f"Recommend to the user{user_id} :")
for idx, movie in enumerate(recommendations, 1):
    print(f"{idx}. {movie}")
