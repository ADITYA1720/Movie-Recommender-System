import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import math

ratings = pd.read_csv('ratings.csv')
movies = pd.read_csv('movies.csv')

# generate genre embeddings for each movie
movies['genres'] = movies['genres'].apply(lambda x: x.split("|"))
from collections import Counter
genre_frequency = Counter(g for genres in movies['genres'] for g in genres)

all_genres = set(g for G in movies['genres'] for g in G)
for g in all_genres:
    movies[g] = movies.genres.transform(lambda x: int(g in x))

merged_data = pd.merge(movies, ratings, on='movieId')

sorted_data = merged_data.drop(columns=['genres', 'title', 'timestamp'])
sorted_data = sorted_data.sort_values(by=['userId', 'movieId'], axis=0)
feature_map = sorted_data.drop(columns=['movieId'])

# spli the data into train and test part
train_data, test_data = train_test_split(feature_map, test_size=0.5, stratify=feature_map['userId'], random_state=42)
X_train, y_train = train_data.iloc[:,:-2], train_data.iloc[:,-1]
X_test, y_test = test_data.iloc[:,:-2], test_data.iloc[:,-1]

# train an XGBoost model

# '''
# implemented a GridSearchCV pipeline to get the best parameters for the model
# xgb = XGBRegressor()
# parameters_grid = { 'learning_rate' : [0.01, 0.5, 0.1], 'max_depth' : [5, 10], 'min_child_weight' : [3,5,7], 'n_estimators':[100,200,300,400,500]}
# model = GridSearchCV(estimator=xgb, param_grid=parameters_grid)
# model.fit(X_train, y_train)

# '''

xgb = XGBRegressor(objective='reg:squarederror',
                        learning_rate=0.1,
                        max_depth=5,
                        min_child_weight=7,
                        n_estimators=500)
xgb.fit(X_train, y_train)
preds = xgb.predict(X_test)

user_id = st.number_input("Enter User ID")
# if user_id:
#     try:
#         user_input_int = int(user_id)
#         st.write("Integer Input:", user_input_int)
#     except ValueError:
#         st.write("Invalid input. Please enter a valid integer.")
# else:
#     st.write("Please enter a value.")

filtered_df = merged_data[merged_data['userId'] == user_id]
filtered_df = filtered_df[['movieId', 'title','rating','userId']]
result = pd.merge(train_data.iloc[:,:-1], filtered_df, on='userId', how='inner')
result = pd.merge(train_data.iloc[:,:-1], filtered_df, on='userId', how='inner')
st.write(result[['movieId', 'title','rating']])

def create_X(df):
    M = df['userId'].nunique()
    N = df['movieId'].nunique()

    user_mapper = dict(zip(np.unique(df["userId"]), list(range(M))))
    movie_mapper = dict(zip(np.unique(df["movieId"]), list(range(N))))

    user_inv_mapper = dict(zip(list(range(M)), np.unique(df["userId"])))
    movie_inv_mapper = dict(zip(list(range(N)), np.unique(df["movieId"])))

    user_index = [user_mapper[i] for i in df['userId']]
    item_index = [movie_mapper[i] for i in df['movieId']]

    X = csr_matrix((df["rating"], (user_index,item_index)), shape=(M,N))

    return X, user_mapper, movie_mapper, user_inv_mapper, movie_inv_mapper

X, user_mapper, movie_mapper, user_inv_mapper, movie_inv_mapper = create_X(ratings)

#Get Top Rated Movies
sorted_df = ratings[ratings['userId'] == int(user_id)].sort_values(by='rating', ascending=False)
top_rated_movies = sorted_df[:5]
movieId_list = top_rated_movies['movieId']
# definition to get the most similar movie to a given movie using the generated genre embdding

def find_similar_movies(movie_id, X, movie_mapper, movie_inv_mapper, k, metric='cosine'):
    X = X.T
    neighbour_ids = []

    movie_ind = movie_mapper[movie_id]
    movie_vec = X[movie_ind]
    if isinstance(movie_vec, (np.ndarray)):
        movie_vec = movie_vec.reshape(1,-1)
    # use k+1 since kNN output includes the movieId of interest
    kNN = NearestNeighbors(n_neighbors=k+1, algorithm="brute", metric=metric)
    kNN.fit(X)
    neighbour = kNN.kneighbors(movie_vec, return_distance=False)
    for i in range(0,k):
        n = neighbour.item(i)
        neighbour_ids.append(movie_inv_mapper[n])
    neighbour_ids.pop(0)
    return neighbour_ids

similar_movies_index=[]
for id in movieId_list:
  similar_movies_index.extend(find_similar_movies(id, X, movie_mapper, movie_inv_mapper, k=2))

# method to get the genre embedding for the fetched most similar movies
def get_genre_embedding(movie_id):

    genres = movies[movies['movieId'] == movie_id]['genres'].tolist()
    embedding = [1 if g in genres[0] else 0 for g in all_genres]
    return embedding
  
def get_recommendation(similar_movies_index):
  movie_titles = dict(zip(movies['movieId'], movies['title']))
  recommendations = []
  for id in similar_movies_index:
    similar_movies = find_similar_movies(id, X, movie_mapper, movie_inv_mapper, metric='cosine', k=6)
    movie_title = movie_titles[id]
  
    embedding = get_genre_embedding(id)
    predicted_rating = xgb.predict(np.array(get_genre_embedding(1)).reshape(1,-1))
    recommendations.append((movie_title, predicted_rating[0]))
  return recommendations
st.write("========================================================")
st.write("Recommended Movies and Predicted Ratings for {user_id} are")
my_recommendations = get_recommendation(similar_movies_index)
for i,tup in enumerate(my_recommendations):
   st.write(f"{i+1}. {tup[0]} ")
   st.write(f"Predicted Rating:{tup[1]}")

## Evaluate error on the data
st.write("========================================================")
st.write("The Root Mean Square Error is")
error=0
for pred_val, true_val in zip(preds, y_test):
  sq_diff = pow((pred_val - true_val),2) 
  error += sq_diff
rmse = math.sqrt(error)
st.write(f"RMSE : {rmse}")
