import pandas as pd
#import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from collections import Counter

# Load data
ratings = pd.read_csv('ratings.csv')
movies = pd.read_csv('movies.csv')

data = pd.merge(ratings, movies, on='movieId')

from sklearn.model_selection import train_test_split
train_data, test_data = train_test_split(data, test_size=0.5, stratify=data['userId'], random_state=42)

st.write("=========================Overview of Train Data=========================")

# Overview of the  Train data
st.write(train_data['userId'].value_counts().sort_index())
n_ratings_train = len(train_data)
n_movies_train = train_data['movieId'].nunique()
n_users_train = train_data['userId'].nunique()

# Display data overview
st.write(f"Number of ratings in Train Data: {n_ratings_train}")
st.write(f"Number of unique movie Ids in Train Data: {n_movies_train}")
st.write(f"Number of unique users in Train Data: {n_users_train}")
st.write(f"Average number of ratings per user in Train Data: {round(n_ratings_train/n_users_train, 2)}")
st.write(f"Average number of ratings per movie in Train Data: {round(n_ratings_train/n_movies_train, 2)}")


st.write("====================================================================")
st.write("=========================Overview of Test Data=========================")

# Overview of the  Test data
st.write(test_data['userId'].value_counts().sort_index())

n_ratings_test = len(test_data)
n_movies_test = test_data['movieId'].nunique()
n_users_test = test_data['userId'].nunique()

# Display data overview
st.write(f"Number of ratings in Train Data: {n_ratings_test}")
st.write(f"Number of unique movie Ids in Train Data: {n_movies_test}")
st.write(f"Number of unique users in Train Data: {n_users_test}")
st.write(f"Average number of ratings per user in Train Data: {round(n_ratings_test/n_users_test, 2)}")
st.write(f"Average number of ratings per movie in Train Data: {round(n_ratings_test/n_movies_test, 2)}")

