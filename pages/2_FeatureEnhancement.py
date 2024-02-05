# Movie-Recommender-System
import streamlit as st
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
import seaborn as sns

ratings = pd.read_csv('ratings.csv')
movies = pd.read_csv('movies.csv')

st.write("1. Extracted all different types of Genres")
movies['genres'] = movies['genres'].apply(lambda x: x.split("|"))
st.write(movies.head())
from collections import Counter
genre_frequency = Counter(g for genres in movies['genres'] for g in genres)
st.write(f"There are {len(genre_frequency)} genres.")
st.write(genre_frequency)

st.write("2. Applied One Hot Encoding to Genres")
genres = set(g for G in movies['genres'] for g in G)

for g in genres:
    movies[g] = movies.genres.transform(lambda x: int(g in x))
st.write(movies.head())

st.write("3. Merged the data on Movie ID")
data = pd.merge(movies, ratings, on='movieId')
st.write(data.head())
